"""
Code Searcher Module

Searches for code repositories on GitHub and Kaggle.
"""

import logging
import os
import json
from pathlib import Path
from typing import List

# Use relative imports within internagent
from ..code_search import search_github_repos
from .models import Source, SourceType, Platform

logger = logging.getLogger(__name__)


class CodeSearcher:
    """
    Searches for code repositories and Kaggle resources.
    """

    def __init__(self, max_results_per_query: int = 5):
        """
        Initialize the code searcher.

        Args:
            max_results_per_query: Maximum results per query (default: 5)
        """
        self.max_results = max_results_per_query or 5

        logger.info(f"Initialized CodeSearcher with max_results: {self.max_results}")

    def search_github(self, queries: List[str], before: str = None, date_field: str = 'pushed') -> List[Source]:
        """
        Search GitHub repositories.

        Args:
            queries: List of search queries
            before: Optional date filter (YYYY-MM-DD format)
            date_field: Date field to filter on ('pushed' or 'created')

        Returns:
            List of Source objects
        """
        all_repos = []

        for query in queries:
            logger.info(f"Searching GitHub for query: {query}")

            try:
                repos_text = search_github_repos(query, limit=self.max_results, before=before, date_field=date_field)
                repos = self._parse_repos_text(repos_text)
                for repo in repos:
                    source = self._convert_to_source(repo)
                    if source:
                        all_repos.append(source)
            except Exception as e:
                logger.error(f"Error searching GitHub for query '{query}': {e}")
                continue

        # Deduplicate by URL
        unique_repos = self._deduplicate(all_repos)

        logger.info(f"Found {len(unique_repos)} unique repositories")
        return unique_repos

    def search_kaggle(self, queries: List[str], before: str = None) -> List[Source]:
        """
        Search Kaggle datasets and kernels.

        Args:
            queries: List of search queries
            before: Optional date filter

        Returns:
            List of Source objects
        """
        api = self._init_kaggle_api()
        if api is None:
            return []

        results: List[Source] = []
        for query in queries:
            try:
                logger.info(f"Searching Kaggle datasets for query: {query}")
                results.extend(self._search_kaggle_datasets(api, query, before))
            except Exception as e:
                logger.error(f"Error searching Kaggle datasets for '{query}': {e}")

            try:
                logger.info(f"Searching Kaggle kernels for query: {query}")
                results.extend(self._search_kaggle_kernels(api, query, before))
            except Exception as e:
                logger.error(f"Error searching Kaggle kernels for '{query}': {e}")

        return self._deduplicate(results)

    def _init_kaggle_api(self):
        """Initialize Kaggle API with authentication."""
        try:
            cfg_path = os.getenv("KAGGLE_CONFIG_DIR")
            if cfg_path:
                p = Path(cfg_path)
                cfg_dir = p if p.is_dir() else p.parent
                os.environ["KAGGLE_CONFIG_DIR"] = str(cfg_dir)
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            try:
                api.authenticate()
                return api
            except Exception:
                pass

            # Fallback: read kaggle.json and set env vars
            candidate = None
            if cfg_path:
                p = Path(cfg_path)
                candidate = p if p.is_file() else (p / "kaggle.json")
            else:
                home = Path.home()
                candidate = home / ".kaggle" / "kaggle.json"

            if candidate and candidate.exists():
                try:
                    with open(candidate, "r") as f:
                        data = json.load(f)
                    username = data.get("username")
                    key = data.get("key")
                    if username and key:
                        os.environ["KAGGLE_USERNAME"] = username
                        os.environ["KAGGLE_KEY"] = key
                        api.authenticate()
                        return api
                except Exception as e:
                    logger.error(f"Failed to load kaggle.json for env auth: {e}")

            # If still not authenticated, raise
            logger.error("Kaggle API authentication failed. Ensure kaggle.json or env vars are set.")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Kaggle API: {e}")
            return None

    def _search_kaggle_datasets(self, api, query: str, before: str = None) -> List[Source]:
        """Search Kaggle datasets."""
        datasets = []
        for name in ("dataset_list", "datasets_list"):
            func = getattr(api, name, None)
            if callable(func):
                try:
                    try:
                        datasets = func(search=query, sort_by="updated")
                    except TypeError:
                        datasets = func(search=query)
                    break
                except TypeError:
                    try:
                        try:
                            datasets = func(search=query, page_size=self.max_results, sort_by="updated")
                        except TypeError:
                            datasets = func(search=query, page_size=self.max_results)
                        break
                    except Exception:
                        continue
        sources: List[Source] = []
        for d in datasets[: self.max_results]:
            title = getattr(d, "title", None) or getattr(d, "ref", "dataset")
            ref = getattr(d, "ref", "")
            url = f"https://www.kaggle.com/datasets/{ref}" if ref else "https://www.kaggle.com/datasets"
            description = getattr(d, "subtitle", None) or getattr(d, "description", None)
            ts = getattr(d, "lastUpdated", None) or getattr(d, "updated", None) or getattr(d, "dateCreated", None)
            src = Source(
                title=title,
                url=url,
                source_type=SourceType.WEBPAGE,
                platform=Platform.KAGGLE,
                description=description,
                metadata={"ref": ref, "timestamp": ts},
                timestamp=str(ts) if ts else None,
                kaggle_ref=ref or None,
                kaggle_item_type="dataset",
                kaggle_subtitle=getattr(d, "subtitle", None)
            )
            if before and ts:
                try:
                    if str(ts) > str(before):
                        continue
                except Exception:
                    pass
            sources.append(src)
        return sources

    def _search_kaggle_kernels(self, api, query: str, before: str = None) -> List[Source]:
        """Search Kaggle kernels."""
        kernels = []
        for name in ("kernel_list", "kernels_list"):
            func = getattr(api, name, None)
            if callable(func):
                try:
                    try:
                        kernels = func(search=query, sort_by="dateRun")
                    except TypeError:
                        kernels = func(search=query)
                    break
                except TypeError:
                    try:
                        try:
                            kernels = func(search=query, page_size=self.max_results, sort_by="dateRun")
                        except TypeError:
                            kernels = func(search=query, page_size=self.max_results)
                        break
                    except Exception:
                        continue
        sources: List[Source] = []
        for k in kernels[: self.max_results]:
            title = getattr(k, "title", None) or getattr(k, "ref", "kernel")
            ref = getattr(k, "ref", "")
            # Kaggle uses /code for notebooks
            url = f"https://www.kaggle.com/code/{ref}" if ref else "https://www.kaggle.com/code"
            description = getattr(k, "subtitle", None) or getattr(k, "description", None)
            ts = getattr(k, "lastRunTime", None) or getattr(k, "updated", None)
            src = Source(
                title=title,
                url=url,
                source_type=SourceType.CODE,
                platform=Platform.KAGGLE,
                description=description,
                metadata={"ref": ref, "timestamp": ts},
                timestamp=str(ts) if ts else None,
                kaggle_ref=ref or None,
                kaggle_item_type="kernel",
                kaggle_subtitle=getattr(k, "subtitle", None)
            )
            if before and ts:
                try:
                    if str(ts) > str(before):
                        continue
                except Exception:
                    pass
            sources.append(src)
        return sources

    def _parse_repos_text(self, repos_text: str) -> List[dict]:
        """
        Parse the formatted repository string from search_github_repos.

        Args:
            repos_text: Formatted string containing repository information

        Returns:
            List of repository dictionaries
        """
        repos = []

        # Split by repository entries
        lines = repos_text.strip().split("\n")

        current_repo = {}
        for line in lines:
            line = line.strip()

            if line.startswith("Name:"):
                if current_repo:  # Save previous repo
                    repos.append(current_repo)
                current_repo = {"name": line.replace("Name:", "").strip()}

            elif line.startswith("Description:"):
                current_repo["description"] = line.replace("Description:", "").strip()

            elif line.startswith("Link:"):
                current_repo["link"] = line.replace("Link:", "").strip()

        # Add last repo
        if current_repo:
            repos.append(current_repo)

        return repos

    def _convert_to_source(self, repo: dict) -> Source:
        """
        Convert a repository dictionary to a Source object.

        Args:
            repo: Repository dictionary

        Returns:
            Source object
        """
        try:
            name = repo.get("name", "Unknown")
            link = repo.get("link", "")
            description = repo.get("description", "No description available")

            return Source(
                title=name,
                url=link,
                source_type=SourceType.CODE,
                platform=Platform.GITHUB,
                description=description,
                metadata=repo
            )

        except Exception as e:
            logger.error(f"Error converting repo to Source: {e}")
            return None

    def _deduplicate(self, repos: List[Source]) -> List[Source]:
        """
        Remove duplicate repositories based on URL.

        Args:
            repos: List of Source objects

        Returns:
            Deduplicated list of Source objects
        """
        seen_urls = set()
        unique_repos = []

        for repo in repos:
            if repo.url not in seen_urls:
                seen_urls.add(repo.url)
                unique_repos.append(repo)

        return unique_repos

