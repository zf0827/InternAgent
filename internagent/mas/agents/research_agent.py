"""
Research Agent for InternAgent

Performs deep research search across multiple platforms (papers, code, web, scholar)
and returns comprehensive search results for a given research idea.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchers import (
    Idea, SearchQuery, SearchResults, Source, ResourceNode, ResourceTree,
    PaperSearcher, CodeSearcher, WebSearcher
)
from ..tools.query_generation import QueryGenerator, DeepQueryGenerator, read_page
from ..tools.file_tree_utils import get_repo_tree
from ..tools.filter_utils import filter_search_results

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Agent that performs deep research across multiple platforms.
    
    Searches papers (arXiv, Semantic Scholar), code (GitHub, Kaggle),
    web pages, and Google Scholar to gather comprehensive research resources.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ResearchAgent"
        
        # Load search parameters from config
        self.max_iters = config.get("max_iters", 3)
        self.max_results_per_source = config.get("max_results_per_source", 3)
        self.enable_code_search = config.get("enable_code_search", True)
        self.enable_web_search = config.get("enable_web_search", True)
        self.enable_scholar_search = config.get("enable_scholar_search", True)
        self.paper_sources = config.get("paper_sources", ["arxiv", "semantic_scholar"])
        self.search_timeout = config.get("search_timeout", 30)
        self.top_k_readpage = config.get("top_k_readpage", 3)  # Number of top results to enrich with page content
        
        # Filtering and file tree generation settings
        self.enable_filtering = config.get("enable_filtering", True)
        self.enable_file_tree = config.get("enable_file_tree", True)
        self.file_tree_max_level = config.get("file_tree_max_level", 3)
        self.filter_top_k_papers = config.get("filter_top_k_papers", 10)
        self.filter_top_k_code = config.get("filter_top_k_code", 6)
        self.filter_top_k_web = config.get("filter_top_k_web", 10)
        
        # Initialize query generators
        self.query_generator = QueryGenerator()
        self.deep_query_generator = DeepQueryGenerator()
        
        # Initialize searchers
        self.paper_searcher = PaperSearcher(
            max_results_per_query=self.max_results_per_source,
            sources=self.paper_sources
        )
        
        if self.enable_code_search:
            self.code_searcher = CodeSearcher(max_results_per_query=self.max_results_per_source)
        else:
            self.code_searcher = None
            
        if self.enable_web_search or self.enable_scholar_search:
            self.web_searcher = WebSearcher(max_results_per_query=self.max_results_per_source)
        else:
            self.web_searcher = None
        
        logger.info(f"Initialized ResearchAgent with max_iters={self.max_iters}, "
                   f"sources={self.paper_sources}, code_search={self.enable_code_search}")

    def _context_to_idea(self, context: Dict[str, Any]) -> Idea:
        """
        Convert context dictionary to Idea object.
        If context["idea"] is string -> return Idea(raw_text=context["idea"])
        If context["idea"] is dict -> return Idea(motivation=context["idea"]["motivation"]...)
        else build Idea from context["motivation"]...
        """
        # Extract part field from context (can be at top level or inside idea dict)
        part = context.get("part")
        
        idea_ctx = context.get("idea")
        if isinstance(idea_ctx, str):
            # If part is specified in context but idea is just a string, use part from context
            return Idea(
                motivation="",
                research_question="",
                method="",
                experimental_setting="",
                expected_results=None,
                raw_text=idea_ctx,
                part=part,
            )
        if isinstance(idea_ctx, dict):
            # part can be in idea_ctx dict or in context
            part_from_idea = idea_ctx.get("part")
            return Idea(
                motivation=idea_ctx.get("motivation", ""),
                research_question=idea_ctx.get("research_question", ""),
                method=idea_ctx.get("method", ""),
                experimental_setting=idea_ctx.get("experimental_setting", ""),
                expected_results=idea_ctx.get("expected_results"),
                raw_text=idea_ctx.get("raw_text"),
                part=part_from_idea if part_from_idea is not None else part,
            )

        motivation = context.get("motivation", "")
        research_question = context.get("research_question", "")
        method = context.get("method", "")
        experimental_setting = context.get("experimental_setting", "")
        expected_results: Optional[str] = context.get("expected_results")
        raw_text: Optional[str] = context.get("raw_text")

        if not any([motivation, research_question, method, experimental_setting, expected_results, raw_text]):
            goal = context.get("goal", {})
            if isinstance(goal, dict):
                desc = goal.get("description", "")
                domain = goal.get("domain", "")
                background = goal.get("background", "")
                constraints = goal.get("constraints", [])
                rt = desc
                if domain:
                    rt += f"\n\nDomain: {domain}"
                if background:
                    rt += f"\n\nBackground: {background}"
                if constraints:
                    rt += "\n\nConstraints:\n" + "\n".join([f"- {c}" for c in constraints])
                raw_text = rt
            else:
                raw_text = self._format_context(context)

        return Idea(
            motivation=motivation,
            research_question=research_question,
            method=method,
            experimental_setting=experimental_setting,
            expected_results=expected_results,
            raw_text=raw_text,
            part=part,
        )

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deep research search for a given idea.
        
        Args:
            context: Contains the research idea to search for
            params: Additional parameters (max_iters override, etc.)
            
        Returns:
            Dictionary containing search_results and params
        """
        idea = self._context_to_idea(context)
        max_iters = params.get("max_iters", self.max_iters)

        try:
            # Run search in a thread to avoid blocking
            results = await asyncio.to_thread(self._deep_search, idea, max_iters)
        except Exception as e:
            logger.error(f"Research search failed: {e}", exc_info=True)
            raise AgentExecutionError(str(e))

        return {
            "search_results": results.to_dict(),
            "params": params,
        }

    def _deep_search(self, idea: Idea, max_iters: int = 2) -> SearchResults:
        """
        Perform deep research tree search.
        
        Args:
            idea: Research idea to search for
            max_iters: Maximum depth of iterative search
            
        Returns:
            SearchResults object with all found resources
        """
        logger.info("=" * 80)
        logger.info("Starting deep research tree search")
        logger.info("=" * 80)

        # Generate initial queries
        base_queries = self._generate_queries(idea)
        if not base_queries:
            return SearchResults(idea=idea, queries=SearchQuery())

        self._log_queries("initial", base_queries)

        # Aggregate queries for final result
        agg_queries = SearchQuery(
            paper_queries=list(base_queries.paper_queries),
            github_queries=list(base_queries.github_queries),
            kaggle_queries=list(base_queries.kaggle_queries),
            web_queries=list(base_queries.web_queries),
            scholar_queries=list(base_queries.scholar_queries),
        )

        # Prepare goals for page reading
        goals_root = {
            "papers": self._goal_prompt_idea(idea, "paper"),
            "github_repos": self._goal_prompt_idea(idea, "repos"),
            "kaggle_results": self._goal_prompt_idea(idea, "kaggle"),
            "web_pages": self._goal_prompt_idea(idea, "web"),
            "scholar_results": self._goal_prompt_idea(idea, "scholar"),
        }

        # Perform initial search
        first_results = self._search_platforms(base_queries)
        first_results = self._enrich_with_readpage_by_type(first_results, goals_root, self.top_k_readpage)
        
        papers = list(first_results.get("papers", []))
        repos = list(first_results.get("github_repos", []))
        kaggle = list(first_results.get("kaggle_results", []))
        web = list(first_results.get("web_pages", []))
        scholar = list(first_results.get("scholar_results", []))

        self._log_results("initial", papers, repos, kaggle, web, scholar)

        # Build resource tree
        tree = ResourceTree()
        root = ResourceNode(
            depth=0,
            papers=self._dedup_by_title(papers),
            repos=self._dedup_by_url(repos),
            kaggle=self._dedup_by_url(kaggle),
            web=self._dedup_by_url(web),
            scholar=self._dedup_by_title(scholar),
            queries_used=base_queries,
            goals_by_type={
                "papers": goals_root["papers"],
                "repos": goals_root["github_repos"],
                "kaggle": goals_root["kaggle_results"],
                "web": goals_root["web_pages"],
                "scholar": goals_root["scholar_results"],
            },
        )
        root.source_ids = [s.id for s in (root.papers + root.repos + root.kaggle + root.web + root.scholar)]
        tree.add_node(root)
        logger.info(f"[ROOT] node_id={root.id} depth={root.depth}")
        logger.info(f"[ROOT] counts papers={len(root.papers)} repos={len(root.repos)} "
                   f"kaggle={len(root.kaggle)} web={len(root.web)} scholar={len(root.scholar)}")

        # Expand tree with deep queries
        max_depth = max_iters if isinstance(max_iters, int) else 2
        self._expand_node(tree, root, max_depth)

        # Collect all results from tree
        final_sr = SearchResults(
            idea=idea,
            queries=agg_queries,
            papers=[],
            github_repos=[],
            kaggle_results=[],
            web_pages=[],
            scholar_results=[],
            resource_tree=tree,
        )
        
        final_papers: List[Source] = []
        final_repos: List[Source] = []
        final_kaggle: List[Source] = []
        final_web: List[Source] = []
        final_scholar: List[Source] = []
        
        for n in tree.nodes.values():
            final_papers.extend(n.papers)
            final_repos.extend(n.repos)
            final_kaggle.extend(n.kaggle)
            final_web.extend(n.web)
            final_scholar.extend(n.scholar)
        
        final_sr.papers = self._dedup_by_title(final_papers)
        final_sr.github_repos = self._dedup_by_url(final_repos)
        final_sr.kaggle_results = self._dedup_by_url(final_kaggle)
        final_sr.web_pages = self._dedup_by_url(final_web)
        final_sr.scholar_results = self._dedup_by_title(final_scholar)
        
        logger.info("\n" + final_sr.summary())
        
        # Apply filtering if enabled
        if self.enable_filtering:
            logger.info("Applying relevance filtering...")
            try:
                final_sr = filter_search_results(
                    final_sr,
                    top_k_papers=self.filter_top_k_papers,
                    top_k_code=self.filter_top_k_code,
                    top_k_web=self.filter_top_k_web
                )
                logger.info("Filtering completed")
                logger.info("\n" + final_sr.summary())
            except Exception as e:
                logger.error(f"Filtering failed: {e}, continuing with unfiltered results")
        
        # Generate file trees for GitHub repos if enabled
        if self.enable_file_tree:
            logger.info("Generating file trees for GitHub repositories...")
            self._generate_file_trees(final_sr.github_repos)
        
        logger.info("=" * 80)
        return final_sr

    def _generate_queries(self, idea: Idea) -> Optional[SearchQuery]:
        """Generate initial search queries from idea."""
        try:
            return self.query_generator.generate(idea)
        except Exception as e:
            logger.error(f"Error generating base queries: {e}")
            return None

    def _generate_deep_queries(self, resources: List[Source]) -> Optional[SearchQuery]:
        """Generate deep queries from existing resources."""
        try:
            return self.deep_query_generator.generate(resources)
        except Exception as e:
            logger.error(f"Error generating deep queries: {e}")
            return None

    def _search_platforms(self, queries: SearchQuery) -> Dict[str, List[Source]]:
        """Search all configured platforms in parallel."""
        results = {
            "papers": [],
            "github_repos": [],
            "kaggle_results": [],
            "web_pages": [],
            "scholar_results": []
        }
        tasks = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            if queries.paper_queries:
                tasks.append(("papers", executor.submit(self.paper_searcher.search, queries.paper_queries)))
            if self.code_searcher and getattr(queries, "github_queries", None):
                tasks.append(("github_repos", executor.submit(self.code_searcher.search_github, queries.github_queries)))
            if self.code_searcher and getattr(queries, "kaggle_queries", None):
                tasks.append(("kaggle_results", executor.submit(self.code_searcher.search_kaggle, queries.kaggle_queries)))
            if self.web_searcher and getattr(queries, "web_queries", None) and self.enable_web_search:
                tasks.append(("web_pages", executor.submit(self.web_searcher.search_web, queries.web_queries)))
            if self.web_searcher and getattr(queries, "scholar_queries", None) and self.enable_scholar_search:
                tasks.append(("scholar_results", executor.submit(self.web_searcher.search_scholar, queries.scholar_queries)))
            
            for key, fut in tasks:
                try:
                    results[key] = fut.result(timeout=self.search_timeout)
                except Exception as e:
                    logger.error(f"Error in {key} search: {e}")
        
        return results

    def _enrich_with_readpage_by_type(self, results: Dict[str, List[Source]], 
                                     goals_by_key: Dict[str, str], 
                                     top_k: int = 2) -> Dict[str, List[Source]]:
        """Enrich top results with page content."""
        def _process(goal: str):
            def inner(src: Source) -> Source:
                if not getattr(src, "url", None):
                    return src
                try:
                    data = read_page(src.url, goal=goal or "", mode="raw+structured")
                    md = data.get("metadata", {}) or {}
                    st = data.get("structured", {}) or {}
                    src.page_title = md.get("title")
                    src.page_headings = md.get("headings")
                    src.page_links = md.get("links")
                    src.page_raw_text = data.get("raw_text")
                    src.page_structured_summary = st.get("summary")
                    src.page_structured_evidence = st.get("evidence")
                    src.page_structured_rational = st.get("rational")
                    src.page_extractor_goal = goal
                    src.page_read_source = "jina"
                except Exception as e:
                    logger.error(f"Exception while processing URL {src.url}: {str(e)}")
                    src.metadata["readpage_error"] = str(e)
                return src
            return inner

        for key in list(results.keys()):
            items = results.get(key, []) or []
            if not items:
                continue
            k = min(top_k, len(items))
            subset = items[:k]
            goal = goals_by_key.get(key, "")
            logger.info(f"Enriching {key} with goal: {goal[:50]}...")
            
            try:
                with ThreadPoolExecutor(max_workers=k) as ex:
                    enriched = list(ex.map(_process(goal), subset))
                results[key] = enriched + items[k:]
            except Exception as e:
                logger.error(f"Exception during enrichment for {key}: {str(e)}")
        
        return results

    def _expand_node(self, tree: ResourceTree, node: ResourceNode, max_depth: int) -> None:
        """Recursively expand tree node with deep queries."""
        if node.depth >= max_depth:
            return
        
        groups = node.top_k_by_type(3)
        contexts = [
            self.deep_query_generator.build_context(groups, 0, self.top_k_readpage),
            self.deep_query_generator.build_context(groups, 1, self.top_k_readpage),
            self.deep_query_generator.build_context(groups, 2, self.top_k_readpage),
        ]
        
        for idx, ctx in enumerate(contexts):
            q = self.deep_query_generator.generate_from_context(ctx)
            self._log_queries(f"node_{node.id}_ctx_{idx+1}", q)
            res = self._search_platforms(q)
            goals = {
                "papers": self._goal_prompt_queries(q, "paper"),
                "github_repos": self._goal_prompt_queries(q, "repos"),
                "kaggle_results": self._goal_prompt_queries(q, "kaggle"),
                "web_pages": self._goal_prompt_queries(q, "web"),
                "scholar_results": self._goal_prompt_queries(q, "scholar"),
            }
            res = self._enrich_with_readpage_by_type(res, goals, self.top_k_readpage)
            
            child = ResourceNode(
                depth=node.depth + 1,
                papers=self._dedup_by_title(res.get("papers", [])),
                repos=self._dedup_by_url(res.get("github_repos", [])),
                kaggle=self._dedup_by_url(res.get("kaggle_results", [])),
                web=self._dedup_by_url(res.get("web_pages", [])),
                scholar=self._dedup_by_title(res.get("scholar_results", [])),
                queries_used=q,
                goals_by_type={
                    "papers": goals["papers"],
                    "repos": goals["github_repos"],
                    "kaggle": goals["kaggle_results"],
                    "web": goals["web_pages"],
                    "scholar": goals["scholar_results"],
                },
            )
            child.source_ids = [s.id for s in (child.papers + child.repos + child.kaggle + child.web + child.scholar)]
            tree.add_node(child)
            tree.link_child(node.id, child.id)
            logger.info(f"[CHILD] parent={node.id} -> child={child.id} depth={child.depth}")
            self._log_results(f"node_{node.id}_child_{child.id}", 
                            child.papers, child.repos, child.kaggle, child.web, child.scholar)
            self._expand_node(tree, child, max_depth)

    def _goal_prompt_idea(self, idea: Idea, t: str) -> str:
        """Generate goal prompt for page reading based on idea."""
        base = idea.get_full_text()
        if t == "paper":
            return f"Extract methods, tasks, datasets, key findings, and evidence relevant to: {base}"
        if t == "repos":
            return f"Extract implementations, frameworks, dependencies, reproduction notes, and benchmark locations for: {base}"
        if t == "kaggle":
            return f"Extract dataset descriptions, features/labels, notebook highlights, and metrics for: {base}"
        if t == "web":
            return f"Extract engineering best practices, pitfalls, comparisons, and production notes for: {base}"
        if t == "scholar":
            return f"Extract surveys, trends, benchmarks, replication and meta-analysis leads for: {base}"
        return base

    def _goal_prompt_queries(self, queries: SearchQuery, t: str) -> str:
        """Generate goal prompt for page reading based on queries."""
        def join(xs: List[str]) -> str:
            return "; ".join(xs[:8]) if xs else ""
        
        if t == "paper":
            summary = join(queries.paper_queries)
            return f"Use these paper queries to focus methods, tasks, datasets, and evidence: {summary}"
        if t == "repos":
            summary = join(queries.github_queries)
            return f"Use these repository queries to focus implementations, frameworks, dependencies, and benchmarks: {summary}"
        if t == "kaggle":
            summary = join(queries.kaggle_queries)
            return f"Use these Kaggle queries to focus dataset structure, notebooks, features, labels, and metrics: {summary}"
        if t == "web":
            summary = join(queries.web_queries)
            return f"Use these web queries to focus tutorials, best practices, comparisons, and production notes: {summary}"
        if t == "scholar":
            summary = join(queries.scholar_queries)
            return f"Use these scholar queries to focus surveys, reviews, benchmarks, and replications: {summary}"
        return ""

    def _dedup_by_url(self, sources: List[Source]) -> List[Source]:
        """Deduplicate sources by URL."""
        seen = set()
        out = []
        for s in sources:
            key = getattr(s, "url", None) or ""
            if key and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _dedup_by_title(self, sources: List[Source]) -> List[Source]:
        """Deduplicate sources by normalized title."""
        seen = set()
        out = []
        for s in sources:
            key = ''.join((s.title or '').lower().split())
            if key and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _log_queries(self, label: str, queries: SearchQuery) -> None:
        """Log generated queries."""
        logger.info(f"[{label.upper()}] queries total={len(queries.get_all_queries())}")
        logger.info(f"[{label}] paper_queries: {self._format_list(queries.paper_queries)}")
        logger.info(f"[{label}] github_queries: {self._format_list(queries.github_queries)}")
        logger.info(f"[{label}] kaggle_queries: {self._format_list(queries.kaggle_queries)}")
        logger.info(f"[{label}] web_queries: {self._format_list(queries.web_queries)}")
        logger.info(f"[{label}] scholar_queries: {self._format_list(queries.scholar_queries)}")

    def _log_results(self, label: str, papers: List[Source], repos: List[Source], 
                    kaggle: List[Source], web: List[Source], scholar: List[Source]) -> None:
        """Log search results."""
        logger.info(f"[{label.upper()}] results papers={len(papers)} repos={len(repos)} "
                   f"kaggle={len(kaggle)} web={len(web)} scholar={len(scholar)}")

    def _format_list(self, items: List[str], k: int = 6) -> str:
        """Format list of items for logging."""
        if not items:
            return "[]"
        return " | ".join(items[:k])

    def _generate_file_trees(self, github_repos: List[Source]) -> None:
        """Generate file trees for GitHub repositories."""
        for repo in github_repos:
            # Skip if already has file tree
            if hasattr(repo, 'file_tree') and repo.file_tree:
                continue
            
            # Check if this is a GitHub repository
            if not hasattr(repo, 'platform') or repo.platform.value != 'github':
                continue
            
            if not hasattr(repo, 'url') or not repo.url:
                continue
            
            try:
                logger.info(f"Generating file tree for: {repo.url}")
                file_tree = get_repo_tree(repo.url, max_level=self.file_tree_max_level)
                if file_tree and not file_tree.startswith("Error"):
                    repo.file_tree = file_tree
                    logger.info(f"Successfully generated file tree for {repo.url}")
                else:
                    logger.warning(f"Failed to generate file tree for {repo.url}: {file_tree}")
                    repo.file_tree = f"Error: {file_tree}"
            except Exception as e:
                logger.error(f"Exception generating file tree for {repo.url}: {str(e)}")
                repo.file_tree = f"Error: {str(e)}"
