"""
Query Generator Module

Uses dspy to generate structured queries from research ideas for different platforms.
"""

import dspy
import os
from typing import List, Optional
import logging

from ..searchers.models import Idea, SearchQuery

logger = logging.getLogger(__name__)


class QueryGenerationSignature(dspy.Signature):
    """
    Signature for generating search queries from a research idea.
    """
    idea_text = dspy.InputField(desc="Full research idea text: motivation, research question, method, experiment, data, evaluation. Extract core nouns/verbs and key terms.")
    paper_queries = dspy.OutputField(desc="Return 2-3 queries for academic papers (arXiv, Semantic Scholar, PubMed). Include task/method/dataset terms; prefer survey, benchmark, state-of-the-art, review. Comma-separated; each <= 12 words; no duplicates.")
    github_queries = dspy.OutputField(
    desc=(
        "Return 1-2 GitHub search queries for repositories implementing the idea. "
        "Each query will be used as the `q` parameter of GET /search/repositories. "
        "Keep each query SHORT and BROAD: 2-5 content words describing task, "
        "domain/dataset, and framework (e.g., segmentation, time series, graph, "
        "pytorch, tensorflow, implementation, official). Remove stopwords and "
        "avoid full sentences or long phrases. Prefer core nouns/short method names. "
        "You may optionally append simple qualifiers like `language:Python` or `stars:>50`. "
        "Output 1-2 concise, distinct queries, comma-separated, no duplicates, no quotes."
    ))

    kaggle_queries = dspy.OutputField(desc="Return 1-2 Kaggle queries targeting datasets/notebooks. Include dataset names, competition, notebook, kernel, EDA, baseline. Comma-separated; concise and distinct.")
    web_queries = dspy.OutputField(desc="Return 1-2 web queries. Use keywords like tutorial, comparison, best practices, production. Comma-separated; concise and distinct.")
    scholar_queries = dspy.OutputField(desc="Return 1-2 Google Scholar queries. Prefer scholarly keywords (survey, review, benchmark, meta-analysis, replication). Avoid engine-specific qualifiers not supported. Comma-separated; concise and distinct.")


class QueryGenerator(dspy.Module):
    """
    Generates platform-specific search queries from a research idea using dspy.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the query generator.

        Args:
            config: Configuration dictionary for the LLM (optional)
        """
        super().__init__()

        # Load configuration from environment variables if not provided
        if config is None:
            config = self._load_config_from_env()

        # Configure dspy LM instance (will be used in context manager)
        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized QueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise

        # Create the query generation module
        self.generate_queries = dspy.ChainOfThought(QueryGenerationSignature)

    def _load_config_from_env(self) -> dict:
        """Load LLM configuration from environment variables."""
        # Try DeepSeek first, fall back to OpenAI
        ds_api_key = os.getenv("DS_API_KEY")
        if ds_api_key:
            logger.info(f"Using DeepSeek API")
            return {
                "api_key": ds_api_key,
                "api_base": os.getenv("DS_API_BASE_URL"),
                "model": "openai/deepseek-v3"
            }
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            return {
                "api_key": openai_api_key,
                "api_base": os.getenv("OPENAI_API_BASE_URL"),
                "model": "openai/gpt-4o-mini"
            }
        
        raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")

    def forward(self, idea: Idea) -> SearchQuery:
        """
        Generate search queries from an idea.

        Args:
            idea: Research idea

        Returns:
            SearchQuery object with platform-specific queries
        """
        try:
            # Get full text of the idea
            idea_text = idea.get_full_text()

            # Generate queries using dspy with context manager
            logger.info("Generating queries for idea...")
            with dspy.settings.context(lm=self.lm):
                result = self.generate_queries(idea_text=idea_text)

            # Parse the comma-separated query strings
            paper_queries = self._parse_query_list(result.paper_queries)
            github_queries = self._parse_query_list(result.github_queries)
            kaggle_queries = self._parse_query_list(result.kaggle_queries)
            web_queries = self._parse_query_list(result.web_queries)
            scholar_queries = self._parse_query_list(result.scholar_queries)

            # Create SearchQuery object
            search_query = SearchQuery(
                paper_queries=paper_queries,
                github_queries=github_queries,
                kaggle_queries=kaggle_queries,
                web_queries=web_queries,
                scholar_queries=scholar_queries
            )

            logger.info(f"Generated {len(search_query.get_all_queries())} queries across all platforms")
            logger.debug(f"Paper queries: {paper_queries}")
            logger.debug(f"GitHub queries: {github_queries}")
            logger.debug(f"Kaggle queries: {kaggle_queries}")
            logger.debug(f"Web queries: {web_queries}")
            logger.debug(f"Scholar queries: {scholar_queries}")

            return search_query

        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            # Return a fallback query based on the research question
            return self._generate_fallback_queries(idea)

    def _parse_query_list(self, query_string: str) -> List[str]:
        """
        Parse comma-separated query string into a list.

        Args:
            query_string: Comma-separated query string

        Returns:
            List of cleaned query strings
        """
        if not query_string:
            return []

        # Split by comma and clean each query
        queries = [q.strip() for q in query_string.split(",")]

        # Remove empty queries and quotes
        queries = [q.strip('"').strip("'") for q in queries if q.strip()]

        return queries

    def _generate_fallback_queries(self, idea: Idea) -> SearchQuery:
        """
        Generate simple fallback queries if dspy fails.

        Args:
            idea: Research idea

        Returns:
            SearchQuery with basic queries
        """
        logger.warning("Using fallback query generation")

        # Extract key terms from research question and method
        rq = idea.research_question
        method = idea.method

        # Create basic queries
        paper_queries = [
            rq,
            f"{method} methodology"
        ]

        github_queries = [
            f"{method} implementation"
        ]

        kaggle_queries = [
            f"{rq} dataset",
            f"{method} notebook"
        ]

        web_queries = [
            rq
        ]

        scholar_queries = [
            rq
        ]

        return SearchQuery(
            paper_queries=paper_queries,
            github_queries=github_queries,
            kaggle_queries=kaggle_queries,
            web_queries=web_queries,
            scholar_queries=scholar_queries
        )

    def generate(self, idea: Idea) -> SearchQuery:
        """
        Convenience method to generate queries (alias for forward).

        Args:
            idea: Research idea

        Returns:
            SearchQuery object
        """
        return self(idea=idea)


def generate_queries(idea: Idea, config: Optional[dict] = None) -> SearchQuery:
    """
    Standalone function to generate queries from an idea.

    Args:
        idea: Research idea
        config: Optional configuration for the LLM

    Returns:
        SearchQuery object
    """
    generator = QueryGenerator(config)
    return generator.generate(idea)

