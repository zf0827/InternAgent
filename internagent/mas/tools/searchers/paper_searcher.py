"""
Paper Searcher Module

Searches for academic papers across multiple platforms (arXiv, Semantic Scholar, PubMed).
Uses utilities from the parent InternAgent project.
"""

import logging
import re
import os
from typing import List, Optional

# Use relative imports within internagent
from ..utils import (
    fetch_arxiv_papers,
    fetch_semantic_papers,
    fetch_pubmed_papers,
    paper_filter
)

from .models import Source, SourceType, Platform

logger = logging.getLogger(__name__)


class PaperSearcher:
    """
    Searches for academic papers across multiple platforms.
    """

    def __init__(self, max_results_per_query: int = 10, sources: Optional[List[str]] = None):
        """
        Initialize the paper searcher.

        Args:
            max_results_per_query: Maximum results per query (default: 10)
            sources: List of paper sources to search. Options: ["arxiv", "semantic_scholar", "pubmed"]
        """
        self.max_results = max_results_per_query or 10
        self.sources = sources or ["arxiv", "semantic_scholar"]

        logger.info(f"Initialized PaperSearcher with sources: {self.sources}, max_results: {self.max_results}")

    def search(self, queries: List[str], before: str = None) -> List[Source]:
        """
        Search for papers using multiple queries.

        Args:
            queries: List of search queries
            before: Optional date filter (YYYY-MM-DD format)

        Returns:
            List of Source objects
        """
        all_papers = []

        for query in queries:
            logger.info(f"Searching papers for query: {query}")

            try:
                # Search each source
                papers_by_source = self._search_all_sources(query, before)

                # Filter papers
                filtered_papers = paper_filter(papers_by_source)

                # Convert to Source objects
                for source_name, papers in filtered_papers.items():
                    for paper in papers:
                        source = self._convert_to_source(paper, source_name)
                        if source:
                            all_papers.append(source)

            except Exception as e:
                logger.error(f"Error searching papers for query '{query}': {e}")
                continue

        # Deduplicate by title
        unique_papers = self._deduplicate(all_papers)

        logger.info(f"Found {len(unique_papers)} unique papers")
        return unique_papers

    def _search_all_sources(self, query: str, before: str = None) -> dict:
        """
        Search all configured paper sources.

        Args:
            query: Search query
            before: Optional date filter

        Returns:
            Dictionary mapping source names to paper lists
        """
        results = {}

        if "arxiv" in self.sources:
            try:
                arxiv_papers = fetch_arxiv_papers(query, max_results=self.max_results, before=before)
                if arxiv_papers:
                    results["arXiv"] = arxiv_papers
                    logger.info(f"Found {len(arxiv_papers)} papers from arXiv")
            except Exception as e:
                logger.error(f"Error searching arXiv: {e}")

        if "semantic_scholar" in self.sources:
            try:
                s2_papers = fetch_semantic_papers(query, max_results=self.max_results, before=before)
                if s2_papers:
                    results["semantic_scholar"] = s2_papers
                    logger.info(f"Found {len(s2_papers)} papers from Semantic Scholar")
            except Exception as e:
                logger.error(f"Error searching Semantic Scholar: {e}")

        if "pubmed" in self.sources:
            try:
                pubmed_papers = fetch_pubmed_papers(query, max_results=self.max_results, before=before)
                if pubmed_papers:
                    results["pubmed"] = pubmed_papers
                    logger.info(f"Found {len(pubmed_papers)} papers from PubMed")
            except Exception as e:
                logger.error(f"Error searching PubMed: {e}")

        return results

    def _convert_to_source(self, paper: dict, source_name: str) -> Optional[Source]:
        """
        Convert a paper dictionary to a Source object.

        Args:
            paper: Paper dictionary from search API
            source_name: Name of the source (arxiv, semantic_scholar, pubmed)

        Returns:
            Source object or None if conversion fails
        """
        try:
            # Map source name to Platform enum
            platform_map = {
                "arXiv": Platform.ARXIV,
                "semantic_scholar": Platform.SEMANTIC_SCHOLAR,
                "pubmed": Platform.PUBMED
            }
            platform = platform_map.get(source_name, Platform.ARXIV)

            # Extract fields
            title = paper.get("title", "Untitled")
            url = paper.get("url", "")
            abstract = paper.get("abstract", "")
            authors = paper.get("authors", [])
            year = paper.get("year")
            citations = paper.get("citations")
            doi = paper.get("doi")

            # Handle DOI as URL if URL is not available
            if not url and doi:
                url = f"https://doi.org/{doi}"

            pdf_url = None
            s2_tldr = None
            s2_open_access_pdf = None
            arxiv_id = None
            if platform == Platform.SEMANTIC_SCHOLAR:
                s2_tldr = (paper.get("tldr", {}) or {}).get("text") if isinstance(paper.get("tldr"), dict) else paper.get("tldr")
                oap = paper.get("openAccessPdf")
                if isinstance(oap, dict):
                    s2_open_access_pdf = oap.get("url")
                    pdf_url = s2_open_access_pdf
            if platform == Platform.ARXIV:
                try:
                    candidate = url or ""
                    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", candidate)
                    if m:
                        arxiv_id = m.group(1) + (m.group(2) or "")
                except Exception:
                    pass
                if arxiv_id:
                    url = f"http://arxiv.org/html/{arxiv_id}"
                    pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

            return Source(
                title=title,
                url=url,
                source_type=SourceType.PAPER,
                platform=platform,
                description=abstract,
                authors=authors,
                year=year,
                citations=citations,
                metadata=paper,
                doi=doi,
                pdf_url=pdf_url,
                s2_tldr=s2_tldr,
                s2_open_access_pdf=s2_open_access_pdf,
                arxiv_id=arxiv_id,
                timestamp=str(year) if year else None
            )

        except Exception as e:
            logger.error(f"Error converting paper to Source: {e}")
            return None

    def _deduplicate(self, papers: List[Source]) -> List[Source]:
        """
        Remove duplicate papers based on title similarity.

        Args:
            papers: List of Source objects

        Returns:
            Deduplicated list of Source objects
        """
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            # Normalize title for comparison
            normalized_title = ''.join(paper.title.lower().split())

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)

        return unique_papers

