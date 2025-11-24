"""
Web Searcher Module

Searches the web and Google Scholar using the Serper API.
"""

import logging
import http.client
import json
import os
from typing import List, Optional

from .models import Source, SourceType, Platform

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    Searches the web and Google Scholar using Serper API.
    """

    def __init__(self, max_results_per_query: int = 10):
        """
        Initialize the web searcher.

        Args:
            max_results_per_query: Maximum results per query (default: 10)
        """
        self.max_results = max_results_per_query or 10
        self.api_key = os.getenv("SERPER_KEY_ID")

        if not self.api_key:
            logger.warning("Serper API key not found. Web search will be disabled.")

        logger.info(f"Initialized WebSearcher with max_results: {self.max_results}")

    def search_web(self, queries: List[str], before: str = None) -> List[Source]:
        """
        Search the web using multiple queries.

        Args:
            queries: List of search queries
            before: Optional date filter (YYYY-MM-DD format)

        Returns:
            List of Source objects
        """
        if not self.api_key:
            logger.warning("Cannot perform web search: API key not configured")
            return []

        all_results = []

        for query in queries:
            logger.info(f"Searching web for query: {query}")

            try:
                results = self._google_search(query, before)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching web for query '{query}': {e}")
                continue

        # Deduplicate by URL
        unique_results = self._deduplicate(all_results)

        logger.info(f"Found {len(unique_results)} unique web results")
        return unique_results

    def search_scholar(self, queries: List[str], before: str = None) -> List[Source]:
        """
        Search Google Scholar using multiple queries.

        Args:
            queries: List of search queries
            before: Optional date filter

        Returns:
            List of Source objects
        """
        if not self.api_key:
            logger.warning("Cannot perform scholar search: API key not configured")
            return []

        all_results = []

        for query in queries:
            logger.info(f"Searching Google Scholar for query: {query}")

            try:
                results = self._google_scholar_search(query, before)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching Google Scholar for query '{query}': {e}")
                continue

        # Deduplicate by title
        unique_results = self._deduplicate_by_title(all_results)

        logger.info(f"Found {len(unique_results)} unique scholar results")
        return unique_results

    def _google_search(self, query: str, before: str = None) -> List[Source]:
        """
        Perform Google search using Serper API.

        Args:
            query: Search query
            before: Optional date filter

        Returns:
            List of Source objects
        """
        conn = http.client.HTTPSConnection("google.serper.dev")

        # Check if query contains Chinese characters
        contains_chinese = any('\u4E00' <= char <= '\u9FFF' for char in query)

        tbs = None
        if before:
            try:
                b = str(before)
                mmddyyyy = f"{b[5:7]}/{b[8:10]}/{b[0:4]}"
                tbs = f"cdr:1,cd_max:{mmddyyyy}"
            except Exception:
                pass
        if contains_chinese:
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
                "num": self.max_results,
                **({"tbs": tbs} if tbs else {})
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
                "num": self.max_results,
                **({"tbs": tbs} if tbs else {})
            })

        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        # Retry logic
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                logger.warning(f"Attempt {i+1}/5 failed: {e}")
                if i == 4:
                    logger.error("Google search timeout after 5 attempts")
                    return []
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))

        sources = []

        try:
            if "organic" not in results:
                logger.warning(f"No results found for query: {query}")
                return []

            for idx, page in enumerate(results["organic"], 1):
                # Extract information
                title = page.get("title", "Untitled")
                link = page.get("link", "")
                snippet = page.get("snippet", "")
                date = page.get("date", "")
                source_name = page.get("source", "")

                # Create description
                description_parts = []
                if snippet:
                    description_parts.append(snippet)
                if date:
                    description_parts.append(f"Published: {date}")
                if source_name:
                    description_parts.append(f"Source: {source_name}")

                description = " | ".join(description_parts)

                source = Source(
                    title=title,
                    url=link,
                    source_type=SourceType.WEBPAGE,
                    platform=Platform.GOOGLE_SEARCH,
                    description=description,
                    metadata={
                        "date": date,
                        "source": source_name,
                        "snippet": snippet,
                        "rank": idx
                    },
                    timestamp=date or None,
                    web_source_name=source_name or None,
                    web_rank=idx
                )
                if before and date:
                    try:
                        # best-effort lexical compare for yyyy-mm-dd or similar
                        if len(date) >= 10 and date[0:4].isdigit():
                            if date > before:
                                continue
                    except Exception:
                        pass
                sources.append(source)

        except Exception as e:
            logger.error(f"Error parsing Google search results: {e}")

        return sources

    def _google_scholar_search(self, query: str, before: str = None) -> List[Source]:
        """
        Perform Google Scholar search using Serper API.

        Args:
            query: Search query
            before: Optional date filter

        Returns:
            List of Source objects
        """
        conn = http.client.HTTPSConnection("google.serper.dev")

        payload = json.dumps({
            "q": query,
            "num": self.max_results
        })

        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        # Retry logic
        for i in range(5):
            try:
                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                logger.warning(f"Attempt {i+1}/5 failed: {e}")
                if i == 4:
                    logger.error("Google Scholar search timeout after 5 attempts")
                    return []
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))

        sources = []

        try:
            if "organic" not in results:
                logger.warning(f"No scholar results found for query: {query}")
                return []

            for idx, page in enumerate(results["organic"], 1):
                # Extract information
                title = page.get("title", "Untitled")
                link = page.get("link", "")
                snippet = page.get("snippet", "")
                year = page.get("year")
                cited_by = page.get("citedBy")
                publication_info = page.get("publicationInfo", "")
                pdf_url = page.get("pdfUrl", "")

                # Create description
                description_parts = []
                if snippet:
                    description_parts.append(snippet)
                if publication_info:
                    description_parts.append(publication_info)

                description = " | ".join(description_parts)

                source = Source(
                    title=title,
                    url=link or "No link available",
                    source_type=SourceType.SCHOLAR,
                    platform=Platform.GOOGLE_SCHOLAR,
                    description=description,
                    year=year,
                    citations=cited_by,
                    metadata={
                        "publication_info": publication_info,
                        "pdf_url": pdf_url,
                        "snippet": snippet,
                        "rank": idx
                    },
                    pdf_url=pdf_url or None,
                    scholar_publication_info=publication_info or None,
                    scholar_rank=idx,
                    scholar_cited_by=cited_by
                )
                if before and year:
                    try:
                        by = int(str(before)[:4])
                        if int(year) > by:
                            continue
                    except Exception:
                        pass
                sources.append(source)

        except Exception as e:
            logger.error(f"Error parsing Google Scholar results: {e}")

        return sources

    def _deduplicate(self, sources: List[Source]) -> List[Source]:
        """
        Remove duplicate sources based on URL.

        Args:
            sources: List of Source objects

        Returns:
            Deduplicated list of Source objects
        """
        seen_urls = set()
        unique_sources = []

        for source in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)

        return unique_sources

    def _deduplicate_by_title(self, sources: List[Source]) -> List[Source]:
        """
        Remove duplicate sources based on title similarity.

        Args:
            sources: List of Source objects

        Returns:
            Deduplicated list of Source objects
        """
        seen_titles = set()
        unique_sources = []

        for source in sources:
            # Normalize title for comparison
            normalized_title = ''.join(source.title.lower().split())

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_sources.append(source)

        return unique_sources

