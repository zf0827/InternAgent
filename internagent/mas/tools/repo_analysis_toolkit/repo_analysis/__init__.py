"""
Repo Analysis Toolkit
====================

A toolkit for analyzing code repositories and extracting contextual information.

Main Components:
- RepoAnalyzer: Static analysis and view generation (HCT, MCG, FCG)
- ImportanceScorer: Component importance scoring
- ContextBuilder: Context generation for core modules
- TaskMatcher: Task-repository relevance scoring
- RepoContextPipeline: Unified workflow
- FilterAndRankRepos: Multi-repository filtering and ranking

Quick Start:
    >>> from repo_analysis import SimplePipeline
    >>> pipeline = SimplePipeline('/path/to/repo')
    >>> context = pipeline.get_context()
    >>> key_modules = pipeline.get_key_modules()
"""

from .repo_analyzer import RepoAnalyzer
from .importance_scorer import ImportanceScorer
from .context_builder import ContextBuilder
from .task_matcher import TaskMatcher, create_task_dict
from .pipeline import RepoContextPipeline, SimplePipeline
from .multi_repo_filter import FilterAndRankRepos, quick_filter, print_ranking_results
from .config import (
    PipelineConfig,
    AnalysisConfig,
    ScoringConfig,
    ContextConfig,
    TaskMatchingConfig,
    DEFAULT_CONFIG,
    load_config_from_dict,
    load_config_from_json
)

__version__ = "0.1.0"

__all__ = [
    # Core components
    "RepoAnalyzer",
    "ImportanceScorer",
    "ContextBuilder",
    "TaskMatcher",
    
    # Pipelines
    "RepoContextPipeline",
    "SimplePipeline",
    
    # Multi-repo utilities
    "FilterAndRankRepos",
    "quick_filter",
    "print_ranking_results",
    
    # Task utilities
    "create_task_dict",
    
    # Configuration
    "PipelineConfig",
    "AnalysisConfig",
    "ScoringConfig",
    "ContextConfig",
    "TaskMatchingConfig",
    "DEFAULT_CONFIG",
    "load_config_from_dict",
    "load_config_from_json",
]
