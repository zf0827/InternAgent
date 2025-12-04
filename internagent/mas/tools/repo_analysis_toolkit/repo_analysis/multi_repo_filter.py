"""
Multi-Repo Filter - Filter and rank multiple repositories by relevance.

This module provides utilities for processing multiple repositories
and ranking them by their relevance to a specific task.
"""

import json
import logging
from typing import Dict, List, Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .pipeline import RepoContextPipeline
from .task_matcher import TaskMatcher, create_task_dict
from .utils import get_code_abs_token

logger = logging.getLogger(__name__)


class FilterAndRankRepos:
    """
    Filter and rank multiple repositories by task relevance.
    
    This class processes multiple repositories in parallel,
    extracts contexts, and ranks them by relevance to a task.
    """
    
    def __init__(self, llm_client: Optional[Callable] = None,
                 max_workers: int = 4, max_tokens_per_repo: int = 4000):
        """
        Initialize multi-repo filter.
        
        Args:
            llm_client: Optional LLM client for task matching
            max_workers: Number of parallel workers
            max_tokens_per_repo: Maximum tokens per repository context
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.max_tokens_per_repo = max_tokens_per_repo
        self.task_matcher = TaskMatcher(llm_client=llm_client)
    
    def process_single_repo(self, repo_path: str) -> Optional[Dict]:
        """
        Process a single repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary with repo information and context, or None if failed
        """
        try:
            pipeline = RepoContextPipeline(repo_path)
            pipeline.analyze()
            pipeline.score_importance(top_k=20)
            context = pipeline.build_context(max_tokens=self.max_tokens_per_repo)
            
            context_str = pipeline.context_builder.export_to_string()
            
            return {
                'repo_path': repo_path,
                'repo_name': Path(repo_path).name,
                'context': context_str,
                'context_dict': context,
                'key_modules': pipeline.key_modules
            }
        except Exception as e:
            logger.error(f"Error processing repo {repo_path}: {e}")
            return None
    
    def process_repos_parallel(self, repo_paths: List[str],
                               show_progress: bool = True) -> List[Dict]:
        """
        Process multiple repositories in parallel.
        
        Args:
            repo_paths: List of repository paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of processed repository information
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single_repo, repo_path): repo_path
                for repo_path in repo_paths
            }
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Processing repos")
            
            for future in iterator:
                repo_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {repo_path}: {e}")
        
        return results
    
    def filter_and_rank(self, task: Dict, repo_paths: List[str],
                       top_k: int = 5, use_llm: bool = None,
                       batch_size: int = 5) -> List[Dict]:
        """
        Filter and rank repositories by task relevance.
        
        Args:
            task: Task dictionary or description string
            repo_paths: List of repository paths
            top_k: Number of top repositories to return
            use_llm: Whether to use LLM (defaults to True if llm_client available)
            batch_size: Batch size for LLM scoring
            
        Returns:
            List of top-k repositories with scores, sorted by relevance
        """
        # Convert string to task dict if needed
        if isinstance(task, str):
            task = create_task_dict(task)
        
        # Determine whether to use LLM
        if use_llm is None:
            use_llm = self.llm_client is not None
        
        # Step 1: Process all repositories
        logger.info(f"Processing {len(repo_paths)} repositories...")
        repos = self.process_repos_parallel(repo_paths, show_progress=True)
        
        if not repos:
            logger.warning("No repositories processed successfully")
            return []
        
        # Step 2: Score repositories by task relevance
        logger.info(f"Scoring {len(repos)} repositories against task...")
        
        if use_llm and self.llm_client:
            # Use batch LLM scoring
            scored_repos = self.task_matcher.match_multiple_repos(
                task=task,
                repos=repos,
                use_llm=True,
                batch_size=batch_size
            )
        else:
            # Use heuristic scoring
            scored_repos = self.task_matcher.match_multiple_repos(
                task=task,
                repos=repos,
                use_llm=False
            )
        
        # Step 3: Sort by relevance score
        scored_repos.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Return top-k
        return scored_repos[:top_k]
    
    def save_results(self, results: List[Dict], output_file: str,
                    include_full_context: bool = False) -> None:
        """
        Save ranking results to JSON file.
        
        Args:
            results: List of scored repositories
            output_file: Path to output file
            include_full_context: Whether to include full context (can be large)
        """
        # Prepare data for export
        export_data = []
        for repo in results:
            repo_data = {
                'repo_path': repo.get('repo_path'),
                'repo_name': repo.get('repo_name'),
                'relevance_score': repo.get('relevance_score'),
                'scoring_method': repo.get('scoring_method'),
                'dimensions': repo.get('dimensions'),
                'key_modules': repo.get('key_modules', [])[:10]  # Limit to top 10
            }
            
            if include_full_context:
                repo_data['context'] = repo.get('context')
            
            export_data.append(repo_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    def load_repo_list(self, repo_list_file: str) -> List[str]:
        """
        Load repository paths from a file.
        
        Args:
            repo_list_file: Path to file containing repo paths (one per line)
            
        Returns:
            List of repository paths
        """
        with open(repo_list_file, 'r', encoding='utf-8') as f:
            repo_paths = [line.strip() for line in f if line.strip()]
        
        # Filter out non-existent paths
        valid_paths = []
        for path in repo_paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                logger.warning(f"Repository path does not exist: {path}")
        
        logger.info(f"Loaded {len(valid_paths)} valid repository paths")
        return valid_paths


def quick_filter(task_description: str, repo_paths: List[str],
                top_k: int = 5, llm_client: Optional[Callable] = None) -> List[Dict]:
    """
    Quick function to filter and rank repositories.
    
    Args:
        task_description: Task description string
        repo_paths: List of repository paths
        top_k: Number of top repositories to return
        llm_client: Optional LLM client
        
    Returns:
        List of top-k repositories with scores
    """
    filter_ranker = FilterAndRankRepos(llm_client=llm_client)
    task = create_task_dict(task_description)
    results = filter_ranker.filter_and_rank(task, repo_paths, top_k=top_k)
    return results


def print_ranking_results(results: List[Dict]) -> None:
    """
    Print ranking results in a readable format.
    
    Args:
        results: List of scored repositories
    """
    print("\n" + "=" * 80)
    print("REPOSITORY RANKING RESULTS")
    print("=" * 80 + "\n")
    
    for i, repo in enumerate(results, 1):
        print(f"{i}. {repo.get('repo_name', 'Unknown')}")
        print(f"   Path: {repo.get('repo_path', 'N/A')}")
        print(f"   Relevance Score: {repo.get('relevance_score', 0):.3f}")
        print(f"   Scoring Method: {repo.get('scoring_method', 'N/A')}")
        
        if 'dimensions' in repo and isinstance(repo['dimensions'], dict):
            print(f"   Dimensions:")
            for dim, score in repo['dimensions'].items():
                if isinstance(score, (int, float)):
                    print(f"     - {dim}: {score}")
        
        key_modules = repo.get('key_modules', [])
        if key_modules:
            print(f"   Top 3 Key Modules:")
            for j, module in enumerate(key_modules[:3], 1):
                print(f"     {j}. {module.get('path', 'N/A')} "
                     f"(score: {module.get('importance_score', 0):.2f})")
        
        print()
    
    print("=" * 80)

