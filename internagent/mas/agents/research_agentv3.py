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
from ..tools.querygenv2.readpage import read_page
from sentence_transformers import SentenceTransformer, CrossEncoder
from ..tools.enricher import (
    enrich_papers_with_extraction,
    enrich_web_with_reports,
    enrich_code_with_rawtext,
    enrich_code_with_repo,
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

        # 加载 BGE 模型（全局单例，避免重复加载）
        embedding_model_name = config.get("embedding_model_name", "BAAI/bge-base-en-v1.5")
        reranker_model_name = config.get("reranker_model_name", "BAAI/bge-reranker-base")
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}.__init__:{frame.f_lineno}] Loading BGE models: {embedding_model_name} and {reranker_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker_model = CrossEncoder(reranker_model_name)
        logger.info(f"[{self.__class__.__name__}.__init__:{frame.f_lineno}] BGE models loaded successfully")

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

        # 维护全局去重集合（基于 URL）
        seen_paper_urls = set()
        seen_web_urls = set()
        seen_github_urls = set()
        
        # 维护全局富化集合（基于 URL），避免重复富化
        enriched_web_urls = set()
        enriched_github_urls = set()

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

            # 2) 搜索（paper / web / github），使用 Q，并进行全局去重和 read_page
            new_papers, new_web_pages, new_github_repos = await self._run_search(
                idea, Q, params, seen_paper_urls, seen_web_urls, seen_github_urls
            )

            # 3) 合并新的 sources 到历史 sources
            all_papers.extend(new_papers)
            all_web_pages.extend(new_web_pages)
            all_github_repos.extend(new_github_repos)

            # 4) 对合并后的 sources 进行重排
            idea_full_text = idea.get_full_text()
            all_papers, all_web_pages, all_github_repos = await self._run_rerank(
                idea_full_text, all_papers, all_web_pages, all_github_repos
            )

            # 5) 富化 web 和 code（基于 page_raw_text），为后续 refine 提供更丰富的描述信息
            # 只对未富化的 sources 执行富化
            all_web_pages = await self._run_enrich_web(idea_text, all_web_pages, params, enriched_web_urls)
            all_github_repos = await self._run_enrich_code(idea_text, all_github_repos, params, enriched_github_urls)

            # 6) 如果不是最后一轮且 enable_refine=True，进行 refine 生成新的查询
            if is_refine_iteration and self.enable_refine:
                Q = self._run_refine(idea, all_papers, all_web_pages, all_github_repos, Q)
                self._log_refined_queries(Q)

        # 8) 对最终的 topk 结果进行富化（paper 抽取和 code 仓库分析）
        all_papers, all_github_repos = await self._run_enrich_paper_and_code_final(
            idea_text, all_papers, all_github_repos, params
        )

        # 9) 返回最终结果
        refined_queries = SearchQuery()
        if self.enable_refine and depth > 0:
            refined_queries = Q

        # 10) 获取future papers
        future_papers = await self._get_future_papers(idea, initial_queries, params)
        all_papers = all_papers + future_papers

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


    async def _get_future_papers(self, idea: Idea, initial_queries: SearchQuery, params: Dict[str, Any]) -> List[Source]:
        """
        获取future papers（时间戳 >= before 的论文）
        
        使用独立的搜索流程，与常规搜索保持独立：
        - 使用独立的 seen_paper_urls 集合进行去重
        - 使用 initial_queries.paper_queries 进行搜索
        - 设置 after=before 来获取未来时间的论文
        """
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Starting future papers search")
        
        # 1. 使用 initial_queries.paper_queries 进行搜索
        seen_paper_urls = set()  # 独立的去重集合
        # 创建只包含 paper_queries 的 SearchQuery 对象
        Q = SearchQuery(paper_queries=initial_queries.paper_queries)
        logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Starting search with {len(Q.paper_queries)} queries")
        self._log_queries(Q)
        
        all_papers: List[Source] = []
        before = params.get("before")
        depth = params.get("depth", 0)
        
        if not before:
            logger.warning(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] No 'before' parameter provided, skipping future papers search")
            return []
        
        # 迭代循环 depth 次
        iterations = depth if depth > 0 else 1
        for iteration in range(iterations):
            is_last_iteration = (iteration == iterations - 1)
            is_refine_iteration = (depth > 0 and not is_last_iteration)
            frame = inspect.currentframe()
            if depth > 0:
                logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Iteration {iteration + 1}/{depth}")
            else:
                logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Single search (no iteration)")
            
            # 搜索：after=before 表示获取时间戳 >= before 的论文
            paper_pairs = self.paper_searcher.search(
                Q.paper_queries, 
                basic_idea=idea.basic_idea or "", 
                before=None, 
                after=before
            )
            papers = [p for p, _ in paper_pairs if p.url and p.url not in seen_paper_urls]
            for p in papers:
                if p.url:
                    seen_paper_urls.add(p.url)
            all_papers.extend(papers)
            
            if not is_last_iteration:
                # 2. 对 papers 进行重排
                idea_full_text = idea.get_full_text()
                all_papers, _, _ = await self._run_rerank(idea_full_text, all_papers, [], [])
                # 3. 生成新的查询
                if is_refine_iteration:
                    Q = self._run_refine(idea, all_papers, [], [], Q)
                    self._log_refined_queries(Q)
        
        # 4. 对 papers 进行富化
        idea_text = idea.get_full_text()
        all_papers, _ = await self._run_enrich_paper_and_code_final(idea_text, all_papers, [], params)
        
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Completed future papers search: found {len(all_papers)} papers (after={before})")
        self._log_sources("future_papers_search", all_papers)
        
        return all_papers

    async def _run_search(
        self, 
        idea: Idea, 
        queries: SearchQuery, 
        params: Dict[str, Any],
        seen_paper_urls: set,
        seen_web_urls: set,
        seen_github_urls: set,
    ) -> (List[Source], List[Source], List[Source]):
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._run_search:{frame.f_lineno}] Starting search")
        title = params.get("title")

        # paper 返回 (Source, q_idx)
        before = params.get("before")
        after = params.get("after")
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

        # 全局去重：过滤掉之前迭代中已见过的结果
        papers = [p for p in papers if p.url and p.url not in seen_paper_urls]
        web_pages = [w for w in web_pages if w.url and w.url not in seen_web_urls]
        github_repos = [g for g in github_repos if g.url and g.url not in seen_github_urls]

        # 更新 seen 集合
        for p in papers:
            if p.url:
                seen_paper_urls.add(p.url)
        for w in web_pages:
            if w.url:
                seen_web_urls.add(w.url)
        for g in github_repos:
            if g.url:
                seen_github_urls.add(g.url)

        # 对 web 和 github 执行 read_page（在全局去重之后）
        web_pages = await self._enrich_with_readpage(web_pages, "web")
        github_repos = await self._enrich_with_readpage(github_repos, "github")

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

    async def _enrich_with_readpage(self, sources: List[Source], source_type: str) -> List[Source]:
        """
        对 sources 执行 read_page，填充 page_raw_text 和相关元数据。
        
        Args:
            sources: 要富化的 Source 列表
            source_type: 源类型（"web" 或 "github"）
            
        Returns:
            富化后的 Source 列表
        """
        if not sources:
            return sources
        
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._enrich_with_readpage:{frame.f_lineno}] Starting read_page for {source_type} (count={len(sources)})"
        )
        
        enriched_sources = []
        for src in sources:
            if not src.url:
                enriched_sources.append(src)
                continue
            
            try:
                page_data = read_page(src.url)
                if isinstance(page_data, dict):
                    # Extract raw text from the dict returned by read_page
                    src.page_raw_text = page_data.get("raw", "")
                    # Also store metadata if available
                    if "md" in page_data and page_data["md"]:
                        md = page_data["md"]
                        if isinstance(md, dict):
                            src.page_title = md.get("title")
                            src.page_headings = md.get("headings", [])
                            src.page_links = md.get("links", [])
                else:
                    src.page_raw_text = str(page_data) if page_data else ""
            except Exception as e:
                logger.warning(f"Failed to read page for {source_type} {src.url}: {e}")
                src.page_raw_text = ""
            
            enriched_sources.append(src)
        
        logger.info(
            f"[{self.__class__.__name__}._enrich_with_readpage:{frame.f_lineno}] Completed read_page for {source_type}"
        )
        return enriched_sources

    async def _run_enrich_web(
        self,
        idea_text: str,
        web_pages: List[Source],
        params: Dict[str, Any],
        enriched_web_urls: set,
    ):
        frame = inspect.currentframe()
        # 过滤出未富化的 web_pages
        to_enrich = [wp for wp in web_pages if wp.url and wp.url not in enriched_web_urls]
        if not to_enrich:
            logger.info(
                f"[{self.__class__.__name__}._run_enrich_web:{frame.f_lineno}] All web pages already enriched (total={len(web_pages)})"
            )
            return web_pages
        
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_web:{frame.f_lineno}] Starting enrich web (total={len(web_pages)}, to_enrich={len(to_enrich)})"
        )
        web_temp = params.get("web_temperature", self.temperature)

        # 只对未富化的 web_pages 执行富化
        enriched_pages = await enrich_web_with_reports(self._call_model, idea_text, to_enrich, web_temp)
        self._log_enrich("web_enrich", enriched_pages, key="web_report")
        
        # 更新 enriched_web_urls 集合
        for wp in enriched_pages:
            if wp.url:
                enriched_web_urls.add(wp.url)

        return web_pages

    async def _run_enrich_code(
        self,
        idea_text: str,
        github_repos: List[Source],
        params: Dict[str, Any],
        enriched_github_urls: set,
    ):
        frame = inspect.currentframe()
        # 过滤出未富化的 github_repos
        to_enrich = [gr for gr in github_repos if gr.url and gr.url not in enriched_github_urls]
        if not to_enrich:
            logger.info(
                f"[{self.__class__.__name__}._run_enrich_code:{frame.f_lineno}] All github repos already enriched (total={len(github_repos)})"
            )
            return github_repos
        
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_code:{frame.f_lineno}] Starting enrich code (total={len(github_repos)}, to_enrich={len(to_enrich)})"
        )
        code_temp = params.get("code_temperature", self.temperature)

        # 只对未富化的 github_repos 执行富化
        enriched_repos = await enrich_code_with_rawtext(self._call_model, idea_text, to_enrich, code_temp)
        self._log_enrich("code_enrich", enriched_repos, key="code_report")
        
        # 更新 enriched_github_urls 集合
        for gr in enriched_repos:
            if gr.url:
                enriched_github_urls.add(gr.url)

        return github_repos

    async def _run_enrich_paper_and_code_final(
        self,
        idea_text: str,
        papers: List[Source],
        github_repos: List[Source],
        params: Dict[str, Any],
    ):
        """
        对最终的 topk 结果进行富化：
        - papers: 使用 extraction_agent 抽取 PDF
        - github_repos: 使用 enrich_code_with_repo 基于仓库上下文生成报告
        """
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_paper_and_code_final:{frame.f_lineno}] Starting final enrich paper/code (papers={len(papers)}, github={len(github_repos)})"
        )
        code_temp = params.get("code_temperature", self.temperature)

        # paper 抽取
        papers = await enrich_papers_with_extraction(papers, self.extraction_agent)
        self._log_enrich("paper_enrich_final", papers, key="paper_extract")

        # code 报告（基于 repo_context/readme）
        github_repos = await enrich_code_with_repo(self._call_model, idea_text, github_repos, code_temp)
        self._log_enrich("code_enrich_final", github_repos, key="code_report")

        return papers, github_repos

    async def _run_rerank(
        self,
        idea_full_text: str,
        papers: List[Source],
        web_pages: List[Source],
        github_repos: List[Source],
    ):
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._run_rerank:{frame.f_lineno}] Starting rerank with idea_full_text")
        papers_ranked = self._rerank_single("papers", idea_full_text, papers)
        web_ranked = self._rerank_single("web", idea_full_text, web_pages)
        github_ranked = self._rerank_single("github", idea_full_text, github_repos)
        return papers_ranked, web_ranked, github_ranked

    def _rerank_single(self, label: str, idea_full_text: str, sources: List[Source]) -> List[Source]:
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
                text = src.description or src.page_raw_text or src.title
            else:
                text = self._source_text(src)
                if not text:
                    text = src.description or src.title
            items.append((idx, src, f"[idx:{idx}] {text}"))

        article_list = [item[2] for item in items]
        
        try:
            reranked = rerank_articles_two_stage(
                core_article=idea_full_text,
                article_list=article_list,
                top_k=self.top_k,
                source_type=label,
                enable_llm_scoring=True,
                embedding_model=self.embedding_model,
                reranker_model=self.reranker_model,
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

        idea_full_text = idea.get_full_text()
        paper_scored = self._collect_scored_sources(papers)
        if paper_scored and queries.paper_queries:
            top_sources = [s for _, s in paper_scored]
            similarity_scores = [float(score) for score, _ in paper_scored]
            source_queries = [s.metadata.get("query", "") if s.metadata else "" for s in top_sources]
            refined.paper_queries = self.query_generator.refine_paper_queries(
                idea_full_text=idea_full_text,
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
                idea_full_text=idea_full_text,
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
                idea_full_text=idea_full_text,
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

