import dspy
import logging
import random
import os
from typing import List, Optional, Dict

from ..searchers.models import Source, SearchQuery

logger = logging.getLogger(__name__)


class DeepQuerySignature(dspy.Signature):
    idea = dspy.InputField(
        desc=(
            "The core research idea that we original supposed"
            "With this idea, we have searched some resources and we want to explore more into some interesting field"
            "The generated queries are related to this idea and explore aspects that would be interesting to investigate further after reviewing the provided context."
        ))
    context_snippets = dspy.InputField(desc="List each prior resource on a new line: title | 1-2 key concepts. These are divided into three categories: 1) arxiv papers, 2) github repos & kaggle, 3) google search & scholar. Use this context to expand into adjacent/complementary topics, but ensure all generated queries maintain a strong connection to the core idea.")
    paper_queries = dspy.OutputField(desc="Return 2-3 queries for academic papers (arXiv, Semantic Scholar, PubMed). CRITICAL REQUIREMENT: The core theme of each query MUST be closely related to the provided idea. Use synonyms/paraphrases, include task/method/dataset terms, prefer keywords like survey, benchmark, state of the art, review. Comma-separated; each query <= 12 words; no duplicates.")
    github_queries = dspy.OutputField(
        desc=(
            "Return 2-3 GitHub search queries for repositories implementing the idea. "
            "CRITICAL REQUIREMENT: The core theme of each query MUST be closely related to the provided idea. "
            "Each query will be used as the `q` parameter of GET /search/repositories. "
            "Keep each query SHORT and BROAD: 2-5 content words describing task, "
            "domain/dataset, and framework (e.g., segmentation, time series, graph, "
            "pytorch, tensorflow, implementation, official). Remove stopwords and "
            "avoid full sentences or long phrases. Prefer core nouns/short method names. "
            "You may optionally append simple qualifiers like `language:Python` or `stars:>50`. "
            "Output 1-2 concise, distinct queries, comma-separated, no duplicates, no quotes."
        ))
    kaggle_queries = dspy.OutputField(desc="Return 2-3 Kaggle queries targeting datasets/notebooks. CRITICAL REQUIREMENT: The core theme of each query MUST be closely related to the provided idea. Include dataset names, competition, notebook, kernel, EDA, baseline. Comma-separated; concise and distinct.")
    web_queries = dspy.OutputField(desc="Return 2-3 web queries. CRITICAL REQUIREMENT: The core theme of each query MUST be closely related to the provided idea. Use keywords like tutorial, comparison, best practices, production. Comma-separated; concise and distinct.")
    scholar_queries = dspy.OutputField(desc="Return 2-3 Google Scholar queries. CRITICAL REQUIREMENT: The core theme of each query MUST be closely related to the provided idea. Prefer scholarly keywords (survey, review, benchmark, meta-analysis, replication). Avoid engine-specific qualifiers that may not be supported. Comma-separated; concise and distinct.")


class DeepQueryGenerator(dspy.Module):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()

        if config is None:
            config = self._load_config_from_env()

        # Configure dspy LM instance (will be used in context manager)
        self.lm = dspy.LM(model=config.get("model", "deepseek-v3"), api_key=config["api_key"], api_base=config.get("api_base"))
        
        # Add system prompt to enforce strict adherence to idea relevance
        system_prompt = """You are a research query generator that follow these requirements:

1. CORE IDEA ADHERENCE: Every single query you generate should be related to the provided idea, not drifted too far.

2. CONTEXT INTEGRATION: Use the provided context (divided into three categories: arxiv papers, github repos & kaggle, google search & scholar) to identify interesting aspects that would be valuable to explore further.

3. DIVERSITY AND COVERAGE: The queries you generate should cover various aspects of the idea and resources, not only focus on one topic. These should be follow-up investigations that deepen understanding of the idea.

Remember: The goal is to generate queries that help explore the idea more deeply, using the context as inspiration but never losing sight of the core research theme."""
        
        self.generator = dspy.ChainOfThought(DeepQuerySignature)
        self.generator.system_prompt = system_prompt

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

    def forward(self, resources: List[Source], idea: str = "") -> SearchQuery:
        context = self._build_context(resources)
        with dspy.settings.context(lm=self.lm):
            result = self.generator(context_snippets=context)
        paper_queries = self._parse_query_list(getattr(result, "paper_queries", ""))
        github_queries = self._parse_query_list(getattr(result, "github_queries", ""))
        kaggle_queries = self._parse_query_list(getattr(result, "kaggle_queries", ""))
        web_queries = self._parse_query_list(getattr(result, "web_queries", ""))
        scholar_queries = self._parse_query_list(getattr(result, "scholar_queries", ""))
        return SearchQuery(
            paper_queries=paper_queries,
            github_queries=github_queries,
            kaggle_queries=kaggle_queries,
            web_queries=web_queries,
            scholar_queries=scholar_queries,
        )

    def _build_context(self, resources: List[Source]) -> str:
        lines = []
        for src in resources[:20]:
            title = (src.title or "").strip()
            desc = (src.description or "").strip()
            snippet = title
            if desc:
                snippet += f" | {desc[:140]}"
            lines.append(snippet)
        return "\n".join(lines)

    def build_context(self, groups: Dict[str, List[Source]], variant: int = 0, top_k: int = 2) -> str:
        parts: List[str] = []
        if variant == 0:
            parts.append("[context_type] arxiv")
            items = groups.get("papers", [])[:top_k]
            for s in items:
                title = (s.title or "").strip()
                summary = (s.page_structured_summary or "").strip()
                evidence = (s.page_structured_evidence or "").strip()
                desc = (s.description or "").strip()
                # merged = " ".join([x for x in [f"summary: {summary}", f"evidence: {evidence}", f"description: {desc}"] if x and not x.endswith(": ")])
                merged = " ".join([x for x in [f"summary: {summary}"] if x and not x.endswith(": ")])
                if not summary and not evidence and not desc:
                    fallback = (s.page_raw_text or "").strip()[:280]
                    merged = f"raw: {fallback}"
                parts.append(f"[paper] {title} | {merged}")
        elif variant == 1:
            parts.append("[context_type] github repos & kaggle")
            for t in ["repos", "kaggle"]:
                items = groups.get(t, [])[:top_k]
                for s in items:
                    title = (s.title or "").strip()
                    summary = (s.page_structured_summary or "").strip()
                    evidence = (s.page_structured_evidence or "").strip()
                    desc = (s.description or "").strip()
                    # merged = " ".join([x for x in [f"summary: {summary}", f"evidence: {evidence}", f"description: {desc}"] if x and not x.endswith(": ")])
                    merged = " ".join([x for x in [f"summary: {summary}", f"description: {desc}"] if x and not x.endswith(": ")])
                    if not summary and not evidence and not desc:
                        fallback = (s.page_raw_text or "").strip()[:280]
                        merged = f"raw: {fallback}"
                    parts.append(f"[{t}] {title} | {merged}")
        else:
            parts.append("[context_type] google search & scholar")
            for t in ["web", "scholar"]:
                items = groups.get(t, [])[:top_k]
                for s in items:
                    title = (s.title or "").strip()
                    summary = (s.page_structured_summary or "").strip()
                    evidence = (s.page_structured_evidence or "").strip()
                    rational = (s.page_structured_rational or "").strip()
                    # merged = " ".join([x for x in [f"summary: {summary}", f"evidence: {evidence}", f"rational: {rational}"] if x and not x.endswith(": ")])
                    merged = " ".join([x for x in [f"summary: {summary}"] if x and not x.endswith(": ")])
                    if not summary and not evidence and not rational:
                        fallback = (s.description or s.page_raw_text or "").strip()[:280]
                        merged = f"raw: {fallback}"
                    parts.append(f"[{t}] {title} | {merged}")
        logger.info(f"Build Context : {parts}")
        return "\n".join(parts)

    def _parse_query_list(self, query_string: str) -> List[str]:
        if not query_string:
            return []
        queries = [q.strip() for q in query_string.split(",")]
        queries = [q.strip('"').strip("'") for q in queries if q.strip()]
        # Randomly choose one to avoid focusing on first word
        return [random.choice(queries)] if queries else []

    def generate(self, resources: List[Source], idea: str = "") -> SearchQuery:
        return self.forward(resources=resources, idea=idea)

    def generate_from_context(self, context: str, idea: str = "") -> SearchQuery:
        with dspy.settings.context(lm=self.lm):
            result = self.generator(idea=idea, context_snippets=context)
        paper_queries = self._parse_query_list(getattr(result, "paper_queries", ""))
        github_queries = self._parse_query_list(getattr(result, "github_queries", ""))
        kaggle_queries = self._parse_query_list(getattr(result, "kaggle_queries", ""))
        web_queries = self._parse_query_list(getattr(result, "web_queries", ""))
        scholar_queries = self._parse_query_list(getattr(result, "scholar_queries", ""))
        return SearchQuery(
            paper_queries=paper_queries,
            github_queries=github_queries,
            kaggle_queries=kaggle_queries,
            web_queries=web_queries,
            scholar_queries=scholar_queries,
        )

