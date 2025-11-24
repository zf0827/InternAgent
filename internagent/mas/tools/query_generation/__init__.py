"""
Query Generation Module for InternAgent

This module provides query generation functionality for research search:
- Initial query generation from research ideas
- Deep query generation from search results
- Web page reading and content extraction
"""

from .query_generator import QueryGenerator, generate_queries
from .deep_query_generator import DeepQueryGenerator
from .readpage import read_page, read_with_jina, extract_structured

__all__ = [
    'QueryGenerator',
    'generate_queries',
    'DeepQueryGenerator',
    'read_page',
    'read_with_jina',
    'extract_structured',
]

