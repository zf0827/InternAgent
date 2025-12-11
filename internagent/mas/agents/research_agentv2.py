"""
Research Agent V2 for InternAgent

Simplified research agent that performs search across arXiv and web platforms.
"""

import logging
import asyncio
import os
import json
import dspy
import shutil
import tempfile
from typing import Dict, Any, Optional, List, Tuple

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchersv2.models import Idea, SearchQuery, SearchResults, Source, SourceType, Platform
from ..tools.querygenv2.query_generator import QueryGenerator, RefineGenerator
from ..tools.querygenv2.reranker import rerank_articles_two_stage
from ..tools.searchersv2.paper_searcher import PaperSearcher
from ..tools.searchersv2.web_searcher import WebSearcher
from ..tools.searchersv2.github_searcher import GitHubSearcher
from ..tools.searchersv2.paper_repo import extract_repos_from_papers

logger = logging.getLogger(__name__)

try:
    from repo_analysis import SimplePipeline
except ImportError:
    logger.warning("repo_analysis package not found. GitHub repo context generation will be disabled.")
    SimplePipeline = None


class ResearchAgentV2(BaseAgent):
    """
    Simplified Research Agent V2 that performs search across arXiv and web platforms.
    
    This agent:
    1. Takes an idea as input
    2. Generates optimized queries for arXiv and web
    3. Searches both platforms
    4. Returns SearchResults with papers and web pages
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ResearchAgentV2"
        
        # Load search parameters from config
        self.max_results_per_query = config.get("max_results_per_query", 8)
        self.enable_paper_filtering = config.get("enable_paper_filtering", True)
        self.paper_batch_size = config.get("paper_batch_size", 8)
        self.web_max_results = config.get("web_max_results", 3)
        self.topk_papers = config.get("topk_papers", 10)
        self.topk_web_pages = config.get("topk_web_pages", 10)
        # Initialize query generator
        self.query_generator = QueryGenerator()
        self.refine_generator = RefineGenerator()
        
        # Initialize searchers
        self.paper_searcher = PaperSearcher(
            max_results_per_query=self.max_results_per_query,
            enable_filtering=False,
            batch_size=self.paper_batch_size
        )
        self.web_searcher = WebSearcher(max_results_per_query=self.web_max_results)
        self.github_searcher = GitHubSearcher()
        
        logger.info(f"Initialized ResearchAgentV2 with paper_filtering={self.enable_paper_filtering}")
    
    def _context_to_idea(self, context: Dict[str, Any]) -> Idea:
        """
        Convert context dictionary to Idea object.
        assert context["idea"] is idea.to_dict()
        """
        idea_ctx = context.get("idea")
        assert isinstance(idea_ctx, dict)
        assert isinstance(idea_ctx["basic_idea_list"], list)
        assert isinstance(idea_ctx["basic_idea"], str)
        return Idea.from_dict(idea_ctx)
    
    def enrich_with_readpage(self, results: SearchResults) -> SearchResults:
        """
        Enrich search results with readpage text.
        For GitHub repos: download and use SimplePipeline to generate context
        Note: Web pages are now enriched directly in WebSearcher.search()
        """
        
        # Enrich GitHub repos with SimplePipeline context
        if SimplePipeline is not None:
            # Create a temporary directory for downloading repos
            temp_dir = tempfile.mkdtemp(prefix="github_repos_")
            logger.info(f"Created temporary directory for GitHub repos: {temp_dir}")
            
            try:
                for github_repo in results.github_repos:
                    try:
                        # Download the repository using SimplePipeline's static method
                        repo_path = SimplePipeline.download_github_repo(github_repo.url, temp_dir)
                        if repo_path and os.path.exists(repo_path):
                            # Use SimplePipeline to generate context
                            logger.info(f"Generating context for repository: {github_repo.url}")
                            pipeline = SimplePipeline(repo_path)
                            context = pipeline.get_context(max_tokens=8000, format='string')
                            github_repo.repo_context = context
                            logger.info(f"Successfully generated context for {github_repo.url}")
                        else:
                            logger.warning(f"Failed to download repository: {github_repo.url}")
                            github_repo.repo_context = None
                    except Exception as e:
                        logger.error(f"Failed to generate context for repository {github_repo.url}: {e}", exc_info=True)
                        github_repo.repo_context = None
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
        else:
            logger.warning("SimplePipeline not available. Skipping GitHub repo context generation.")
            for github_repo in results.github_repos:
                github_repo.repo_context = None
        
        return results
    
    def repos_in_papers(self, papers: List[Source]) -> List[Source]:
        """
        Extract additional GitHub repositories from papers by reading their PDF pages.
        For each paper, selects the most relevant GitHub repo based on n-gram overlap.
        
        Args:
            papers: List of paper sources to extract repos from
            
        Returns:
            List of GitHub repositories found in papers (one per paper)
        """
        return extract_repos_from_papers(papers)

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research search for a given idea.
        
        Args:
            context: Contains the research idea to search for
            params: Additional parameters (before date, after date, etc.)
            
        Returns:
            Dictionary containing search_results and params
        """
        idea = self._context_to_idea(context)
        before = params.get("before")
        after = params.get("after")
        
        try:
            # Run search in a thread to avoid blocking
            results = await asyncio.to_thread(self._search, idea, before, after)
        except Exception as e:
            logger.error(f"Research search failed: {e}", exc_info=True)
            raise AgentExecutionError(str(e))
        
        # add readpage text to search results
        results = self.enrich_with_readpage(results)

        return {
            "search_results": results.to_dict(),
            "params": params,
        }
    
    def _search(self, idea: Idea, before: Optional[str] = None, after: Optional[str] = None) -> SearchResults:
        """
        Perform search across arXiv and web platforms.
        
        Args:
            idea: Research idea to search for
            before: Optional date filter (YYYY-MM-DD format)
            after: Optional date filter (YYYY-MM-DD format)
            
        Returns:
            SearchResults object with all found resources
        """
        logger.info("=" * 80)
        logger.info("Starting research search (V2)")
        logger.info("=" * 80)
        
        # Generate queries
        queries = self.query_generator.generate(idea)
        if not queries.paper_queries and not queries.web_queries:
            logger.warning("No queries generated")
            return SearchResults(idea=idea, queries=queries)
        
        logger.info(f"Generated {len(queries.paper_queries)} paper queries and {len(queries.web_queries)} web queries")
        
        # Extract basic idea for filtering
        basic_idea = idea.basic_idea.strip()
        
        # Search papers with two-round search logic
        # First search: with before and after filters
        # Second search: with before=after, after=None (to get future papers)
        papers: List[Source] = []
        if queries.paper_queries:
            # First search: with before and after
            papers_first = self._search_papers_two_round(
                paper_queries=queries.paper_queries,
                basic_idea=basic_idea,
                before=before,
                after=after
            )
            
            # Second search: with after=before, after=None (to get future papers)
            papers_second: List[Source] = []
            if before and after:
                papers_second = self._search_papers_two_round(
                    paper_queries=queries.paper_queries,
                    basic_idea=basic_idea,
                    before=before,
                    after=after
                )
            
            # Merge results (deduplicate by normalized title)
            seen_titles = set()
            papers = []
            for paper in papers_first + papers_second:
                normalized_title = ''.join(paper.title.lower().split())
                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    papers.append(paper)
            
            logger.info(f"Merged papers: {len(papers_first)} + {len(papers_second)} (future) = {len(papers)} (unique)")

        # Search web
        web_pages: List[Source] = []
        if queries.web_queries:
            logger.info("Searching web...")
            web_pairs = self.web_searcher.search(
                queries=queries.web_queries,
                before=before,
                after=after,
            )
            for src, q_idx in web_pairs:
                if src.metadata is None:
                    src.metadata = {}
                src.metadata["query_index"] = q_idx
                if q_idx is not None and q_idx < len(queries.web_queries):
                    src.metadata["query"] = queries.web_queries[q_idx]
                web_pages.append(src)
            logger.info(f"Found {len(web_pages)} web pages")
        
            # rerank webpages using two-stage reranking
            if web_pages and basic_idea and self.topk_web_pages > 0:
                logger.info(f"Reranking {len(web_pages)} web pages to select top {self.topk_web_pages}")
                web_pages = self._rerank_web_pages(web_pages, basic_idea)
                logger.info(f"Selected top {len(web_pages)} web pages after reranking")
        
        # Search GitHub repositories
        github_repos: List[Source] = []
        logger.info("Searching GitHub repositories...")
        github_repos = self.github_searcher.search(idea=idea, before=before, after=after)
        logger.info(f"Found {len(github_repos)} GitHub repositories")
        
        # Extract additional github repos from papers
        paper_repos = self.repos_in_papers(papers)
        github_repos = github_repos + paper_repos
        logger.info(f"Found {len(paper_repos)} GitHub repositories from papers, total: {len(github_repos)}")

        # Build results
        results = SearchResults(
            idea=idea,
            queries=queries,
            papers=papers,
            github_repos=github_repos,
            kaggle_results=[],  # Placeholder
            web_pages=web_pages,
            scholar_results=[],  # Placeholder
        )
        
        logger.info("\n" + results.summary())
        logger.info("=" * 80)
        
        return results
    
    def _search_papers_two_round(
        self,
        paper_queries: List[str],
        basic_idea: str,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> List[Source]:
        """
        Perform two-round paper search with refinement.
        
        Round 1: Search -> Rerank -> TopK
        Refine: Generate refined queries based on top results
        Round 2: Search -> Merge -> Rerank -> Final TopK
        
        Args:
            paper_queries: List of initial paper queries
            basic_idea: Basic idea text for reranking
            before: Optional date filter
            after: Optional date filter
            
        Returns:
            List of top-k Source objects after two-round search
        """
        # TODO: WAY1: on first filter, save 5 future papers; on second filter, save 5 future papers;
        # TODO: WAY2: use 2 _search_papers_two_round w & w/o after

        logger.info("=" * 80)
        logger.info("Round 1: Initial paper search")
        logger.info("=" * 80)
        
        # Round 1: Search papers
        first_round_results: List[Tuple[Source, int]] = self.paper_searcher.search(
            queries=paper_queries,
            basic_idea=basic_idea,
            before=before,
            after=after,
        )
        
        logger.info(f"Found {len(first_round_results)} papers in first round")
        
        if not first_round_results:
            logger.warning("No papers found in first round")
            return []
        
        # Extract sources and create source-to-query mapping
        first_round_sources = [source for source, _ in first_round_results]
        source_to_query = {}
        for source, q_idx in first_round_results:
            normalized_title = ''.join(source.title.lower().split())
            if normalized_title not in source_to_query:
                source_to_query[normalized_title] = paper_queries[q_idx]
        
        # Round 1: Rerank and select topk
        topk_first_round, topk_similarity_scores, topk_source_queries = self._rerank_and_extract_topk(
            sources=first_round_sources,
            basic_idea=basic_idea,
            topk=self.topk_papers,
            source_to_query=source_to_query
        )
        
        logger.info(f"Selected top {len(topk_first_round)} papers after first round reranking")
        
        # Refine: Generate refined queries
        logger.info("=" * 80)
        logger.info("Refining queries based on top results")
        logger.info("=" * 80)
        
        try:
            refined_queries = self.refine_generator(
                basic_idea=basic_idea,
                top_sources=topk_first_round,
                similarity_scores=topk_similarity_scores,
                source_queries=topk_source_queries,
                original_queries=paper_queries
            )
            logger.info(f"Generated {len(refined_queries)} refined queries")
        except Exception as e:
            logger.error(f"Failed to generate refined queries: {e}", exc_info=True)
            refined_queries = []
        
        # Round 2: Search with refined queries (if any)
        if not refined_queries:
            logger.info("No refined queries generated, using first round results only")
            return topk_first_round
        
        logger.info("=" * 80)
        logger.info("Round 2: Search with refined queries")
        logger.info("=" * 80)
        
        second_round_results: List[Tuple[Source, int]] = self.paper_searcher.search(
            queries=refined_queries,
            basic_idea=basic_idea,
            before=before,
            after=after,
        )
        
        logger.info(f"Found {len(second_round_results)} papers in second round")
        
        # Merge results from both rounds (deduplicate)
        second_round_sources = [source for source, _ in second_round_results]
        all_sources = topk_first_round + second_round_sources
        
        seen_titles = set()
        unique_all_sources = []
        for source in all_sources:
            normalized_title = ''.join(source.title.lower().split())
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_all_sources.append(source)
        
        logger.info(f"Combined {len(unique_all_sources)} unique papers (from {len(topk_first_round)} + {len(second_round_sources)})")
        
        # Final rerank
        logger.info("=" * 80)
        logger.info("Final reranking")
        logger.info("=" * 80)
        
        final_papers, _, _ = self._rerank_and_extract_topk(
            sources=unique_all_sources,
            basic_idea=basic_idea,
            topk=self.topk_papers,
            source_to_query={}  # Not needed for final rerank
        )
        
        logger.info(f"Selected top {len(final_papers)} papers after final reranking")
        
        return final_papers
    
    def _rerank_and_extract_topk(
        self,
        sources: List[Source],
        basic_idea: str,
        topk: int,
        source_to_query: Dict[str, str]
    ) -> Tuple[List[Source], List[float], List[str]]:
        """
        Rerank sources and extract top-k with similarity scores and source queries.
        
        Args:
            sources: List of Source objects to rerank
            basic_idea: Basic idea text for reranking
            topk: Number of top results to return
            source_to_query: Mapping from normalized title to query string
            
        Returns:
            Tuple of (top_sources, similarity_scores, source_queries)
        """
        if not sources:
            return [], [], []
        
        # Collect abstracts and create mapping
        article_list = []
        abstract_to_source = {}
        
        for source in sources:
            abstract = source.description or ""
            if abstract:
                article_list.append(abstract)
                abstract_to_source[abstract] = source
        
        if not article_list:
            logger.warning("No sources with abstracts found for reranking")
            return sources[:topk], [0.0] * min(topk, len(sources)), [""] * min(topk, len(sources))
        
        # Perform two-stage reranking
        core_article = basic_idea.strip()
        rerank_top_k = max(topk, min(20, len(article_list)))
        
        try:
            reranked_results = rerank_articles_two_stage(
                core_article=core_article,
                article_list=article_list,
                top_k=rerank_top_k
            )
            
            # Extract top-k with scores and queries
            top_sources = []
            similarity_scores = []
            source_queries = []
            
            for article_abstract, embed_score, rerank_score in reranked_results[:topk]:
                if article_abstract in abstract_to_source:
                    source = abstract_to_source[article_abstract]
                    # Store scores in metadata
                    if not source.metadata:
                        source.metadata = {}
                    source.metadata["rerank_embed_score"] = embed_score
                    source.metadata["rerank_score"] = rerank_score
                    
                    top_sources.append(source)
                    similarity_scores.append(rerank_score)
                    
                    # Find corresponding query
                    normalized_title = ''.join(source.title.lower().split())
                    query = source_to_query.get(normalized_title, "")
                    source_queries.append(query)
            
            return top_sources, similarity_scores, source_queries
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback: return first topk
            fallback_sources = sources[:topk]
            fallback_scores = [0.0] * len(fallback_sources)
            fallback_queries = []
            for source in fallback_sources:
                normalized_title = ''.join(source.title.lower().split())
                query = source_to_query.get(normalized_title, "")
                fallback_queries.append(query)
            return fallback_sources, fallback_scores, fallback_queries
    
    def _rerank_papers(self, papers: List[Source], basic_idea: str) -> List[Source]:
        """
        Rerank papers using two-stage reranking and select top-k papers.
        
        Args:
            papers: List of Source objects (papers)
            basic_idea: Basic idea text used as query for reranking
            
        Returns:
            List of top-k Source objects after reranking
        """
        if not papers:
            return []
        
        # Collect abstracts and create mapping from abstract to Source
        article_list = []
        abstract_to_source = {}
        
        for paper in papers:
            abstract = paper.description or ""
            if abstract:  # Only include papers with abstracts
                article_list.append(abstract)
                abstract_to_source[abstract] = paper
        
        if not article_list:
            logger.warning("No papers with abstracts found for reranking")
            return papers[:self.topk_papers]  # Fallback to first topk
        
        # Use basic_idea as core_article for reranking
        core_article = basic_idea.strip()
        
        # Perform two-stage reranking
        # top_k should be at least topk_papers, but we can use more for better reranking
        rerank_top_k = max(self.topk_papers, min(20, len(article_list)))
        
        try:
            reranked_results = rerank_articles_two_stage(
                core_article=core_article,
                article_list=article_list,
                top_k=rerank_top_k
            )
            
            # Extract top-k papers based on reranking results
            top_papers = []
            for article_abstract, embed_score, rerank_score in reranked_results[:self.topk_papers]:
                if article_abstract in abstract_to_source:
                    source = abstract_to_source[article_abstract]
                    # Optionally store scores in metadata for debugging
                    if not source.metadata:
                        source.metadata = {}
                    source.metadata["rerank_embed_score"] = embed_score
                    source.metadata["rerank_score"] = rerank_score
                    top_papers.append(source)
            
            return top_papers
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback: return first topk papers
            return papers[:self.topk_papers]
    
    def _rerank_web_pages(self, web_pages: List[Source], basic_idea: str) -> List[Source]:
        """
        Rerank web pages using two-stage reranking and select top-k web pages.
        
        Args:
            web_pages: List of Source objects (web pages)
            basic_idea: Basic idea text used as query for reranking
            
        Returns:
            List of top-k Source objects after reranking
        """
        if not web_pages:
            return []
        
        # Collect descriptions and create mapping from description to Source
        article_list = []
        description_to_source = {}
        
        for web_page in web_pages:
            description = web_page.description or ""
            if description:  # Only include web pages with descriptions
                article_list.append(description)
                description_to_source[description] = web_page
        
        if not article_list:
            logger.warning("No web pages with descriptions found for reranking")
            return web_pages[:self.topk_web_pages]  # Fallback to first topk
        
        # Use basic_idea as core_article for reranking
        core_article = basic_idea.strip()
        
        # Perform two-stage reranking
        # top_k should be at least topk_web_pages, but we can use more for better reranking
        rerank_top_k = max(self.topk_web_pages, min(20, len(article_list)))
        
        try:
            reranked_results = rerank_articles_two_stage(
                core_article=core_article,
                article_list=article_list,
                top_k=rerank_top_k
            )
            
            # Extract top-k web pages based on reranking results
            top_web_pages = []
            for article_description, embed_score, rerank_score in reranked_results[:self.topk_web_pages]:
                if article_description in description_to_source:
                    source = description_to_source[article_description]
                    # Optionally store scores in metadata for debugging
                    if not source.metadata:
                        source.metadata = {}
                    source.metadata["rerank_embed_score"] = embed_score
                    source.metadata["rerank_score"] = rerank_score
                    top_web_pages.append(source)
            
            return top_web_pages
            
        except Exception as e:
            logger.error(f"Reranking web pages failed: {e}", exc_info=True)
            # Fallback: return first topk web pages
            return web_pages[:self.topk_web_pages]