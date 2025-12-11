import asyncio
import inspect
import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from .extraction_agent import ExtractionAgent
from ..tools.searchersv2.models import Idea, SearchResults, SearchQuery, Source, SourceType
from ..tools.searchersv2.paper_searcher import PaperSearcher
from ..tools.searchersv2.web_searcher import WebSearcher
from ..tools.searchersv2.github_searcher_web import GithubWebSearcher
from ..tools.querygenv2.query_generator import QueryGenerator
from ..tools.querygenv2.reranker import rerank_articles_two_stage
from ..tools.enricher import (
    enrich_papers_with_extraction,
    enrich_web_with_reports,
    enrich_code_with_reports,
)

logger = logging.getLogger(__name__)


class ResearchAgentV3(BaseAgent):
    """
    轻量 orchestrator：查询 → 搜索 → 富化(web/code) → 重排 → 富化(paper) →（可选）refine。
    输出 SearchResults，富化结果与重排分数写入 Source.metadata。
    注意：先对 web/github 做报告以补足描述，再重排，论文抽取仅作用于重排后的结果。
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ResearchAgentV3"
        self.temperature = config.get("temperature", 0.7)
        self.top_k = config.get("top_k", 10)
        self.enable_refine = config.get("enable_refine", False)

        self.max_results_per_query = config.get("max_results_per_query", 8)
        self.enable_paper_filtering = config.get("enable_paper_filtering", False)
        self.paper_batch_size = config.get("paper_batch_size", 8)
        self.web_max_results = config.get("web_max_results", 3)
        self.github_max_results = config.get("github_max_results", 3)

        self.query_generator = QueryGenerator()

        self.paper_searcher = PaperSearcher(
            max_results_per_query=self.max_results_per_query,
            enable_filtering=False,
            batch_size=self.paper_batch_size
        )
        self.web_searcher = WebSearcher(max_results_per_query=self.web_max_results)
        self.github_searcher = GithubWebSearcher(max_results_per_query=self.github_max_results)

        extraction_config = config.get(
            "extraction_config",
            {
                "name": "ExtractionAgent",
                "model_provider": config.get("model_provider", "default"),
                "extract_temperature": config.get("extract_temperature", 0.3),
                "_global_config": config.get("_global_config", {}),
            },
        )
        self.extraction_agent = ExtractionAgent(model, extraction_config)

    async def execute(self, idea: Idea, params: Optional[Dict[str, Any]] = None) -> SearchResults:
        params = params or {}
        depth = params.get("depth", 0) if self.enable_refine else 0
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}.execute:{frame.f_lineno}] Starting research_agentv3 pipeline (depth={depth})")

        # 1) 生成初始查询
        initial_queries = self.query_generator.generate(idea)
        Q = initial_queries
        self._log_queries(Q)

        # 初始化累积的 sources
        all_papers: List[Source] = []
        all_web_pages: List[Source] = []
        all_github_repos: List[Source] = []

        idea_text = idea.get_full_text()

        # 迭代循环 depth 次（当 enable_refine=True 且 depth>0 时）
        # 如果 depth=0，执行一次搜索但不迭代
        iterations = depth if depth > 0 else 1
        for iteration in range(iterations):
            is_last_iteration = (iteration == iterations - 1)
            is_refine_iteration = (depth > 0 and not is_last_iteration)
            frame = inspect.currentframe()
            if depth > 0:
                logger.info(f"[{self.__class__.__name__}.execute:{frame.f_lineno}] Iteration {iteration + 1}/{depth}")
            else:
                logger.info(f"[{self.__class__.__name__}.execute:{frame.f_lineno}] Single search (no iteration)")

            # 2) 搜索（paper / web / github），使用 Q
            new_papers, new_web_pages, new_github_repos = await self._run_search(idea, Q, params)

            # 3) 先富化 web，补充可供重排的描述信息
            new_web_pages = await self._run_enrich_web(idea_text, new_web_pages, params)

            # 4) 合并新的 sources 到历史 sources
            all_papers.extend(new_papers)
            all_web_pages.extend(new_web_pages)
            all_github_repos.extend(new_github_repos)

            # 5) 对合并后的 sources 进行重排（基于富化后的描述）
            all_papers, all_web_pages, all_github_repos = await self._run_rerank(
                idea.basic_idea or "", all_papers, all_web_pages, all_github_repos
            )

            # 6) 对重排后的论文和代码结果做富化
            all_papers, all_github_repos = await self._run_enrich_paper_and_code(
                idea_text, all_papers, all_github_repos, params
            )

            # 7) 如果不是最后一轮且 enable_refine=True，进行 refine 生成新的查询
            if is_refine_iteration and self.enable_refine:
                Q = self._run_refine(idea, all_papers, all_web_pages, all_github_repos, Q)
                self._log_refined_queries(Q)

        # 8) 返回最终结果
        refined_queries = SearchQuery()
        if self.enable_refine and depth > 0:
            refined_queries = Q

        return SearchResults(
            idea=idea,
            queries=initial_queries,  # 返回初始查询
            papers=all_papers,
            github_repos=all_github_repos,
            web_pages=all_web_pages,
            kaggle_results=[],
            scholar_results=[],
            refined_queries=refined_queries,
        )

    async def _run_search(
        self, idea: Idea, queries: SearchQuery, params: Dict[str, Any]
    ) -> (List[Source], List[Source], List[Source]):
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._run_search:{frame.f_lineno}] Starting search")
        before = params.get("before")
        after = params.get("after")
        title = params.get("title")

        # paper 返回 (Source, q_idx)
        paper_pairs = self.paper_searcher.search(
            queries.paper_queries, basic_idea=idea.basic_idea or "", before=before, after=after
        )
        papers: List[Source] = []
        for src, q_idx in paper_pairs:
            if src.metadata is None:
                src.metadata = {}
            src.metadata["query_index"] = q_idx
            if q_idx is not None and q_idx < len(queries.paper_queries):
                src.metadata["query"] = queries.paper_queries[q_idx]
            papers.append(src)

        web_pairs = (
            self.web_searcher.search(queries.web_queries)
            # self.web_searcher.search(queries.web_queries, before=before, after=after)
            if queries.web_queries
            else []
        )
        web_pages: List[Source] = []
        for src, q_idx in web_pairs:
            if src.metadata is None:
                src.metadata = {}
            src.metadata["query_index"] = q_idx
            if q_idx is not None and q_idx < len(queries.web_queries):
                src.metadata["query"] = queries.web_queries[q_idx]
            web_pages.append(src)

        github_repos: List[Source] = []
        if queries.github_queries:
            repo_pairs = self.github_searcher.search(queries.github_queries)
            for src, q_idx in repo_pairs:
                if src.metadata is None:
                    src.metadata = {}
                src.metadata["query_index"] = q_idx
                if q_idx is not None and q_idx < len(queries.github_queries):
                    src.metadata["query"] = queries.github_queries[q_idx]
                github_repos.append(src)

        if title:
            papers = self._filter_by_title(papers, title)
            web_pages = self._filter_by_title(web_pages, title)
            github_repos = self._filter_by_title(github_repos, title)

        self._log_sources("paper_search", papers)
        self._log_sources("web_search", web_pages)
        self._log_sources("github_search", github_repos)

        return papers, web_pages, github_repos

    def _filter_by_title(self, sources: List[Source], original_title: str) -> List[Source]:
        """
        过滤掉包含原始论文标题子串的搜索结果。
        
        对于 paper: 检查 title 和 description
        对于 web & code: 检查 title、page_raw_text 和 description
        
        如果任何一个字段包含原始标题的长度 >= 0.6 * len(original_title) 的子串，则过滤掉该 source。
        
        Args:
            sources: 要过滤的源列表
            original_title: 原始论文的标题
            
        Returns:
            过滤后的源列表
        """
        if not original_title or not sources:
            return sources
        
        # 计算最小子串长度
        min_substring_len = int(0.6 * len(original_title))
        if min_substring_len < 1:
            return sources
        
        # 生成所有可能的子串（长度 >= min_substring_len）
        title_lower = original_title.lower()
        substrings = []
        for i in range(len(title_lower) - min_substring_len + 1):
            for j in range(i + min_substring_len, len(title_lower) + 1):
                substrings.append(title_lower[i:j])
        
        filtered_sources = []
        for source in sources:
            # 根据 source_type 决定检查哪些字段
            texts_to_check = []
            
            # 所有类型都检查 title 和 description
            if source.title:
                texts_to_check.append(source.title.lower())
            if source.description:
                texts_to_check.append(source.description.lower())
            
            # web 和 code 类型额外检查 page_raw_text
            if source.source_type in [SourceType.WEBPAGE, SourceType.CODE]:
                if source.page_raw_text:
                    texts_to_check.append(source.page_raw_text.lower())
            
            # 检查是否包含任何子串
            should_filter = False
            for text in texts_to_check:
                for substring in substrings:
                    if substring in text:
                        should_filter = True
                        break
                if should_filter:
                    break
            
            # 如果不包含任何子串，保留该 source
            if not should_filter:
                filtered_sources.append(source)
        
        return filtered_sources

    async def _run_enrich_web(
        self,
        idea_text: str,
        web_pages: List[Source],
        params: Dict[str, Any],
    ):
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_web:{frame.f_lineno}] Starting enrich web (web={len(web_pages)})"
        )
        web_temp = params.get("web_temperature", self.temperature)

        # web 报告
        web_pages = await enrich_web_with_reports(self._call_model, idea_text, web_pages, web_temp)
        self._log_enrich("web_enrich", web_pages, key="web_report")

        return web_pages

    async def _run_enrich_paper_and_code(
        self,
        idea_text: str,
        papers: List[Source],
        github_repos: List[Source],
        params: Dict[str, Any],
    ):
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_paper_and_code:{frame.f_lineno}] Starting enrich paper/code (papers={len(papers)}, github={len(github_repos)})"
        )
        code_temp = params.get("code_temperature", self.temperature)

        # paper 抽取
        papers = await enrich_papers_with_extraction(papers, self.extraction_agent)
        self._log_enrich("paper_enrich", papers, key="paper_extract")

        # code 报告
        github_repos = await enrich_code_with_reports(self._call_model, idea_text, github_repos, code_temp)
        self._log_enrich("code_enrich", github_repos, key="code_report")

        return papers, github_repos

    async def _run_rerank(
        self,
        basic_idea: str,
        papers: List[Source],
        web_pages: List[Source],
        github_repos: List[Source],
    ):
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._run_rerank:{frame.f_lineno}] Starting rerank with basic_idea")
        papers_ranked = self._rerank_single("papers", basic_idea, papers)
        web_ranked = self._rerank_single("web", basic_idea, web_pages)
        github_ranked = self._rerank_single("github", basic_idea, github_repos)
        return papers_ranked, web_ranked, github_ranked

    def _rerank_single(self, label: str, basic_idea: str, sources: List[Source]) -> List[Source]:
        frame = inspect.currentframe()
        if not sources:
            logger.info(f"[{self.__class__.__name__}._rerank_single:{frame.f_lineno}] {label}: no sources to rerank")
            return []

        items = []
        for idx, src in enumerate(sources):
            # 按 label 提取 article：paper 是 description，web 和 code 是 page_raw_text
            if label == "papers":
                text = src.description or src.title
            elif label in ["web", "github"]:
                text = src.page_raw_text or src.description or src.title
            else:
                text = self._source_text(src)
                if not text:
                    text = src.description or src.title
            items.append((idx, src, f"[idx:{idx}] {text}"))

        article_list = [item[2] for item in items]
        try:
            reranked = rerank_articles_two_stage(
                core_article=basic_idea,
                article_list=article_list,
                top_k=self.top_k,
            )
        except Exception as e:
            logger.warning(f"[{self.__class__.__name__}._rerank_single:{frame.f_lineno}] Rerank failed: {e}")
            return sources

        ranked_sources: List[Source] = []
        for content, emb_score, rerank_score in reranked:
            idx = self._extract_idx(content)
            if idx is None or idx >= len(items):
                continue
            src = items[idx][1]
            if src.metadata is None:
                src.metadata = {}
            src.metadata["rerank"] = {
                "embedding_score": emb_score,
                "reranker_score": rerank_score,
            }
            ranked_sources.append(src)

        self._log_rerank(label, ranked_sources)
        return ranked_sources

    def _run_refine(
        self,
        idea: Idea,
        papers: List[Source],
        web_pages: List[Source],
        github_repos: List[Source],
        queries: SearchQuery,
    ) -> SearchQuery:
        refined = SearchQuery()

        paper_scored = self._collect_scored_sources(papers)
        if paper_scored and queries.paper_queries:
            top_sources = [s for _, s in paper_scored]
            similarity_scores = [float(score) for score, _ in paper_scored]
            source_queries = [s.metadata.get("query", "") if s.metadata else "" for s in top_sources]
            refined.paper_queries = self.query_generator.refine_paper_queries(
                basic_idea=idea.basic_idea,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=queries.paper_queries,
            )

        web_scored = self._collect_scored_sources(web_pages)
        if web_scored and queries.web_queries:
            top_sources = [s for _, s in web_scored]
            similarity_scores = [float(score) for score, _ in web_scored]
            source_queries = [s.metadata.get("query", "") if s.metadata else "" for s in top_sources]
            refined.web_queries = self.query_generator.refine_web_queries(
                basic_idea=idea.basic_idea,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=queries.web_queries,
            )

        github_scored = self._collect_scored_sources(github_repos)
        if github_scored and queries.github_queries:
            top_sources = [s for _, s in github_scored]
            similarity_scores = [float(score) for score, _ in github_scored]
            source_queries = [s.metadata.get("query", "") if s.metadata else "" for s in top_sources]
            refined.github_queries = self.query_generator.refine_github_queries(
                basic_idea=idea.basic_idea,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=queries.github_queries,
            )

        return refined

    def _log_queries(self, queries: SearchQuery) -> None:
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._log_queries:{frame.f_lineno}] Generated queries | paper=%d web=%d github=%d scholar=%d kaggle=%d",
            len(queries.paper_queries),
            len(queries.web_queries),
            len(queries.github_queries),
            len(queries.scholar_queries),
            len(queries.kaggle_queries),
        )
        logger.info(
            f"[{self.__class__.__name__}._log_queries:{frame.f_lineno}] Queries detail | paper=%s | web=%s | github=%s",
            queries.paper_queries,
            queries.web_queries,
            queries.github_queries,
        )

    def _log_sources(self, stage: str, sources: List[Source], limit: int = 5) -> None:
        frame = inspect.currentframe()
        if not sources:
            logger.info(f"[{self.__class__.__name__}._log_sources:{frame.f_lineno}] %s: 0 results", stage)
            return
        samples = []
        for idx, src in enumerate(sources[:limit]):
            q = src.metadata.get("query") if src.metadata else None
            samples.append(f"{idx+1}. {src.title or 'N/A'} (query={q})")
        logger.info(f"[{self.__class__.__name__}._log_sources:{frame.f_lineno}] %s: %d results. Top%d: %s", stage, len(sources), limit, " | ".join(samples))

    def _log_enrich(self, stage: str, sources: List[Source], key: str, limit: int = 3) -> None:
        frame = inspect.currentframe()
        total = len(sources)
        with_key = sum(1 for s in sources if s.metadata and key in s.metadata)
        logger.info(f"[{self.__class__.__name__}._log_enrich:{frame.f_lineno}] %s: %d items, %d with %s", stage, total, with_key, key)
        if with_key == 0:
            return
        samples = []
        for src in sources:
            if not src.metadata or key not in src.metadata:
                continue
            content = src.metadata.get(key)
            text = ""
            if isinstance(content, dict):
                text = content.get("summary") or content.get("report_content") or str(content)
            else:
                text = str(content)
            text = (text[:180] + "...") if text and len(text) > 200 else text
            samples.append(f"{src.title or 'N/A'} -> {text}")
            if len(samples) >= limit:
                break
        if samples:
            logger.info(f"[{self.__class__.__name__}._log_enrich:{frame.f_lineno}] %s samples: %s", stage, " | ".join(samples))

    def _log_rerank(self, label: str, sources: List[Source], limit: int = 5) -> None:
        frame = inspect.currentframe()
        if not sources:
            logger.info(f"[{self.__class__.__name__}._log_rerank:{frame.f_lineno}] rerank[%s]: 0 items", label)
            return
        top = []
        for src in sources[:limit]:
            rerank_meta = src.metadata.get("rerank") if src.metadata else {}
            score = None
            if isinstance(rerank_meta, dict):
                score = rerank_meta.get("reranker_score")
            top.append(f"{src.title or 'N/A'} (score={score})")
        logger.info(f"[{self.__class__.__name__}._log_rerank:{frame.f_lineno}] rerank[%s]: total=%d, top%d=%s", label, len(sources), limit, " | ".join(top))

    def _log_refined_queries(self, refined: SearchQuery) -> None:
        frame = inspect.currentframe()
        has_any = refined.paper_queries or refined.web_queries or refined.github_queries
        if not has_any:
            logger.info(f"[{self.__class__.__name__}._log_refined_queries:{frame.f_lineno}] Refine: no new queries")
            return
        logger.info(
            f"[{self.__class__.__name__}._log_refined_queries:{frame.f_lineno}] Refine queries | paper=%s | web=%s | github=%s",
            refined.paper_queries,
            refined.web_queries,
            refined.github_queries,
        )

    def _collect_scored_sources(self, sources: List[Source]) -> List[tuple]:
        scored = []
        for src in sources:
            rerank_meta = src.metadata.get("rerank") if src.metadata else None
            score = None
            if isinstance(rerank_meta, dict):
                score = rerank_meta.get("reranker_score")
            if score is not None:
                scored.append((float(score), src))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: self.top_k]

    def _source_text(self, source: Source) -> str:
        meta = source.metadata or {}
        if "paper_extract" in meta:
            extract = meta["paper_extract"]
            if isinstance(extract, dict):
                parts = []
                for k, v in extract.items():
                    if isinstance(v, str):
                        parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        parts.append(f"{k}: {' '.join([str(i) for i in v])}")
                if parts:
                    return "\n".join(parts)
            return str(extract)
        if "web_report" in meta:
            rep = meta["web_report"]
            if isinstance(rep, dict):
                return rep.get("report_content") or rep.get("summary") or str(rep)
            return str(rep)
        if "code_report" in meta:
            rep = meta["code_report"]
            if isinstance(rep, dict):
                return rep.get("report_content") or rep.get("summary") or str(rep)
            return str(rep)
        if source.page_raw_text:
            return source.page_raw_text
        if source.repo_context:
            return source.repo_context
        return source.description or ""

    @staticmethod
    def _extract_idx(text: str) -> Optional[int]:
        if not text.startswith("[idx:"):
            return None
        try:
            prefix = text.split("]", 1)[0]
            idx_str = prefix.replace("[idx:", "")
            return int(idx_str)
        except Exception:
            return None


__all__ = ["ResearchAgentV3"]

