"""
Task Matcher - Evaluate repository relevance to specific tasks.

This module provides functionality to match repositories with tasks using
LLM-based scoring or rule-based heuristics.
"""

import json
import logging
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class TaskMatcher:
    """
    Match repositories with tasks and score relevance.
    
    Supports two modes:
    1. LLM-based scoring (requires LLM client)
    2. Heuristic-based scoring (keyword matching, no LLM required)
    """
    
    # Scoring dimensions and their weights
    DIMENSION_WEIGHTS = {
        'Algorithm Match': 0.20,
        'Domain Applicability': 0.15,
        'Data Processing Capability': 0.15,
        'Model Implementation Quality': 0.15,
        'Experimental Results': 0.20,
        'Code Readability': 0.10,
        'Structure Organization': 0.05,
        'Scalability': 0.05
    }
    
    def __init__(self, llm_client: Optional[Callable] = None):
        """
        Initialize task matcher.
        
        Args:
            llm_client: Optional LLM client function that takes messages and returns JSON
                       Should have signature: func(messages: List[Dict], json_format: bool) -> Dict
        """
        self.llm_client = llm_client
    
    def match_single_repo(self, task: Dict, repo_context: str, 
                         use_llm: bool = True) -> Dict:
        """
        Match a single repository to a task.
        
        Args:
            task: Task description dictionary with keys like 'task_description', 'keywords'
            repo_context: Repository context string (from ContextBuilder)
            use_llm: Whether to use LLM for scoring (requires llm_client)
            
        Returns:
            Dictionary with relevance scores and dimensions
        """
        if use_llm and self.llm_client:
            return self._llm_based_matching(task, repo_context)
        else:
            return self._heuristic_based_matching(task, repo_context)
    
    def match_multiple_repos(self, task: Dict, repos: List[Dict],
                           use_llm: bool = True, batch_size: int = 5) -> List[Dict]:
        """
        Match multiple repositories to a task.
        
        Args:
            task: Task description dictionary
            repos: List of repository information dicts with 'context' key
            use_llm: Whether to use LLM for scoring
            batch_size: Number of repos to process in one LLM call
            
        Returns:
            List of repos with added scoring information
        """
        if use_llm and self.llm_client:
            return self._llm_batch_matching(task, repos, batch_size)
        else:
            results = []
            for repo in repos:
                repo_context = repo.get('context', '')
                scores = self._heuristic_based_matching(task, repo_context)
                repo.update(scores)
                results.append(repo)
            return results
    
    def _llm_based_matching(self, task: Dict, repo_context: str) -> Dict:
        """Use LLM to score repository relevance."""
        if not self.llm_client:
            raise ValueError("LLM client not configured")
        
        task_desc = task.get('task_description', str(task))
        
        system_prompt = """You are a professional code review expert skilled at analyzing the relevance of code repositories to specific tasks.

Your task: Carefully read the task description and repository context provided by the user. Determine whether the repository contains code that helps solve the task.

Evaluate the repository from the following dimensions (score 0 or 1):
1. Algorithm Match: Does the code algorithm match task requirements
2. Domain Applicability: Is the code applicable to the task domain
3. Data Processing Capability: Comprehensive data processing functions
4. Model Implementation Quality: High-quality model implementation
5. Code Readability: Clear and readable code
6. Structure Organization: Reasonable project structure
7. Experimental Results: Good experimental results
8. Scalability: Easy to extend

Additionally, provide an overall score (1-10) considering code quality, task matching, and implementation completeness.

Return ONLY valid JSON format:
{"Algorithm Match": 0 or 1, "Domain Applicability": 0 or 1, "Data Processing Capability": 0 or 1, "Model Implementation Quality": 0 or 1, "Code Readability": 0 or 1, "Structure Organization": 0 or 1, "Experimental Results": 0 or 1, "Scalability": 0 or 1, "Overall Score": 1-10}"""
        
        user_prompt = f"""Task Description:
{task_desc}

Repository Context:
{repo_context[:6000]}  

Evaluate the repository's relevance to the task."""
        
        try:
            response = self.llm_client(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                json_format=True
            )
            
            # Calculate weighted score
            dimension_score = sum(
                response.get(dim, 0) * weight 
                for dim, weight in self.DIMENSION_WEIGHTS.items()
            )
            
            overall_score = response.get("Overall Score", 0) / 10.0
            total_score = dimension_score * 0.6 + overall_score * 0.4
            
            return {
                'relevance_score': total_score,
                'dimensions': response,
                'scoring_method': 'llm'
            }
            
        except Exception as e:
            logger.error(f"LLM scoring error: {e}")
            return self._heuristic_based_matching(task, repo_context)
    
    def _llm_batch_matching(self, task: Dict, repos: List[Dict], 
                           batch_size: int) -> List[Dict]:
        """Use LLM to score multiple repositories in batches."""
        if not self.llm_client:
            raise ValueError("LLM client not configured")
        
        task_desc = task.get('task_description', str(task))
        results = []
        
        # Process in batches
        for i in range(0, len(repos), batch_size):
            batch = repos[i:i+batch_size]
            
            # Construct batch prompt
            repos_info = []
            for idx, repo in enumerate(batch):
                context = repo.get('context', '')[:4000]  # Limit context size
                repos_info.append(f"Repository {idx+1}:\n{context}\n")
            
            system_prompt = """You are a professional code review expert. Evaluate multiple repositories for their relevance to a specific task.

Score each repository from these dimensions (0 or 1):
1. Algorithm Match
2. Domain Applicability  
3. Data Processing Capability
4. Model Implementation Quality
5. Code Readability
6. Structure Organization
7. Experimental Results
8. Scalability

Also provide an overall score (1-10) for each repository.

Return ONLY valid JSON array:
[{"repo_index": 1, "Algorithm Match": 0/1, "Domain Applicability": 0/1, "Data Processing Capability": 0/1, "Model Implementation Quality": 0/1, "Code Readability": 0/1, "Structure Organization": 0/1, "Experimental Results": 0/1, "Scalability": 0/1, "Overall Score": 1-10}, ...]"""
            
            user_prompt = f"""Task: {task_desc}

Repositories:
{"".join(repos_info)}

Evaluate each repository."""
            
            try:
                response = self.llm_client(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    json_format=True
                )
                
                # Parse response and update repos
                if isinstance(response, list):
                    for score_info in response:
                        repo_idx = score_info.get("repo_index", 0) - 1
                        if 0 <= repo_idx < len(batch):
                            repo = batch[repo_idx]
                            
                            # Calculate score
                            dimension_score = sum(
                                score_info.get(dim, 0) * weight
                                for dim, weight in self.DIMENSION_WEIGHTS.items()
                            )
                            overall_score = score_info.get("Overall Score", 0) / 10.0
                            total_score = dimension_score * 0.6 + overall_score * 0.4
                            
                            repo['relevance_score'] = total_score
                            repo['dimensions'] = score_info
                            repo['scoring_method'] = 'llm_batch'
                
                results.extend(batch)
                
            except Exception as e:
                logger.error(f"LLM batch scoring error: {e}")
                # Fallback to heuristic
                for repo in batch:
                    context = repo.get('context', '')
                    scores = self._heuristic_based_matching(task, context)
                    repo.update(scores)
                results.extend(batch)
        
        return results
    
    def _heuristic_based_matching(self, task: Dict, repo_context: str) -> Dict:
        """
        Use heuristic rules for scoring (no LLM required).
        
        Based on keyword matching and simple rules.
        """
        task_desc = task.get('task_description', str(task))
        task_keywords = task.get('keywords', [])
        
        # Extract keywords from task description
        if not task_keywords and isinstance(task_desc, str):
            # Simple keyword extraction
            task_keywords = [
                word.lower().strip() 
                for word in task_desc.split() 
                if len(word) > 4 and word.isalnum()
            ]
        
        # Normalize for matching
        repo_lower = repo_context.lower()
        task_lower = task_desc.lower() if isinstance(task_desc, str) else ''
        
        # Keyword matching score
        keyword_matches = sum(1 for kw in task_keywords if kw.lower() in repo_lower)
        keyword_score = min(keyword_matches / max(len(task_keywords), 1), 1.0)
        
        # Check for common ML/data science indicators
        ml_keywords = ['train', 'model', 'neural', 'learning', 'dataset', 'accuracy', 
                      'loss', 'optimizer', 'epoch', 'batch']
        ml_score = sum(1 for kw in ml_keywords if kw in repo_lower) / len(ml_keywords)
        
        # Check for data processing indicators
        data_keywords = ['pandas', 'numpy', 'csv', 'dataframe', 'preprocess', 'transform']
        data_score = sum(1 for kw in data_keywords if kw in repo_lower) / len(data_keywords)
        
        # Check code organization
        structure_indicators = ['class', 'def ', 'import', '__init__']
        structure_score = sum(1 for ind in structure_indicators if ind in repo_lower) / len(structure_indicators)
        
        # Combine scores
        total_score = (
            keyword_score * 0.4 +
            ml_score * 0.3 +
            data_score * 0.2 +
            structure_score * 0.1
        )
        
        return {
            'relevance_score': total_score,
            'dimensions': {
                'keyword_match': keyword_score,
                'ml_indicators': ml_score,
                'data_processing': data_score,
                'code_structure': structure_score
            },
            'scoring_method': 'heuristic'
        }


def create_task_dict(description: str, keywords: List[str] = None, 
                     task_type: str = None) -> Dict:
    """
    Helper function to create a task dictionary.
    
    Args:
        description: Task description text
        keywords: Optional list of keywords
        task_type: Optional task type (e.g., 'ml', 'data_analysis', 'web')
        
    Returns:
        Task dictionary
    """
    return {
        'task_description': description,
        'keywords': keywords or [],
        'task_type': task_type
    }

