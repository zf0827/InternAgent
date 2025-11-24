"""
Searchers Module for InternAgent

This module provides search functionality across multiple platforms:
- Paper search (arXiv, Semantic Scholar, PubMed)
- Code search (GitHub, Kaggle)
- Web search (Google, Google Scholar)
"""

from .models import Idea, SearchQuery, Source, SearchResults, SourceType, Platform, ResourceNode, ResourceTree
from .paper_searcher import PaperSearcher
from .code_searcher import CodeSearcher
from .web_searcher import WebSearcher

__all__ = [
    'Idea',
    'SearchQuery',
    'Source',
    'SearchResults',
    'SourceType',
    'Platform',
    'ResourceNode',
    'ResourceTree',
    'PaperSearcher',
    'CodeSearcher',
    'WebSearcher',
]

