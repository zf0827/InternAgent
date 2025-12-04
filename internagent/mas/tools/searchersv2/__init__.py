"""
Searchers V2 Module

Provides search functionality for papers and web pages.
"""

from .paper_searcher import PaperSearcher, ChoosePaper, fetch_arxiv_papers_debug_api
from .web_searcher import WebSearcher
from .github_searcher import GitHubSearcher
from .models import Idea, SearchQuery, SearchResults, Source, SourceType, Platform

__all__ = [
    "PaperSearcher",
    "ChoosePaper",
    "fetch_arxiv_papers_debug_api",
    "WebSearcher",
    "GitHubSearcher",
    "Idea",
    "SearchQuery",
    "SearchResults",
    "Source",
    "SourceType",
    "Platform",
]

