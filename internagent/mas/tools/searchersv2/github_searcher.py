"""
GitHub Searcher Module for V2

Searches GitHub repositories using deep_research agent.
"""

import logging
import json
import re
from typing import List, Optional

from deep_research import AutogenDeepSearchAgent

from .models import Source, SourceType, Platform
from .timestamp import get_repo_timestamps, is_date_in_range

logger = logging.getLogger(__name__)


class GitHubSearcher:
    """
    Searches GitHub repositories using deep_research agent.
    """
    
    def __init__(self):
        """
        Initialize the GitHub searcher.
        """
        self.agent = None
        logger.info("Initialized GitHubSearcher")
    
    def _get_agent(self) -> AutogenDeepSearchAgent:
        """
        Get or create the deep search agent instance.
        
        Returns:
            AutogenDeepSearchAgent instance
        """
        if self.agent is None:
            self.agent = AutogenDeepSearchAgent()
        return self.agent
    
    def _build_query(self, idea) -> str:
        """
        Build the search query from idea components.
        
        Args:
            idea: Idea object
            
        Returns:
            Formatted query string
        """
        query_template = """Please search for GitHub repositories that contain actual implementation code relevant to the following research idea:

### Basic Idea
{basic_idea}

### Methodology
{methodology}

### Experimental Setting
{experimental_setting}

---

Follow the systematic search process defined in your instructions:
1. Analyze the key technical components and the core essence of the research idea ,then extract search keywords
2. Design and execute multiple rounds of targeted searches, Use searching tool
3. Browse repository README files to verify the repository is 1. actually implemented 2. related to the research idea 3. has enough stars and well-maintained, Use browsing tool
    recommend query: "verify the repository is actually implemented and related to [core essence of the research idea]"
4. Check above requirements to exclude repos that do not meet the requirements
5. Select the top 8-12 repositories covering all required categories
6. Return results in the specified JSON format

Begin your search and analysis now.

---

### Final Output
Return a JSON array of 8–12 repositories, sorted by relevance descending (rank 1 = most useful).
Each object exactly:

{{
    "rank": 1,
    "repo_name": "owner/repo",
    "repo_url": "https://github.com/owner/repo",
    "category": "A",  // A, B, or C only
    "stars": 1450,
    "last_update": "2025-10",
    "why_relevant": "Short but precise reason (2–3 sentences max) explaining why this repo is directly useful and confirms it has real implementation code."
}}
"""
        
        basic_idea = idea.basic_idea.strip() if idea.basic_idea else ""
        methodology = idea.method.strip() if idea.method else ""
        experimental_setting = idea.experimental_setting.strip() if idea.experimental_setting else ""
        
        return query_template.format(
            basic_idea=basic_idea,
            methodology=methodology,
            experimental_setting=experimental_setting
        )
    
    def _parse_result_json(self, result_json: str) -> List[dict]:
        """
        Parse the JSON result from the agent.
        
        Args:
            result_json: JSON string from agent
            
        Returns:
            List of repository dictionaries
        """
        # Remove possible markdown code block formatting
        if "```json" in result_json:
            result_json = result_json.split("```json")[1].split("```")[0]
        elif "```" in result_json:
            result_json = result_json.split("```")[1].split("```")[0]
        
        # Remove any leading/trailing whitespace
        result_json = result_json.strip()
        
        try:
            repos = json.loads(result_json)
            if isinstance(repos, list):
                return repos
            elif isinstance(repos, dict) and "repos" in repos:
                return repos["repos"]
            else:
                logger.warning(f"Unexpected JSON format: {type(repos)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            logger.debug(f"Failed to parse JSON. Content: {result_json}")
            # Attempt to parse a list of objects if the string looks like one
            match = re.search(r'\[.*\]', result_json, re.DOTALL)
            if match:
                try:
                    repos = json.loads(match.group(0))
                    return repos if isinstance(repos, list) else []
                except:
                    pass
            return []
    
    def _json_to_sources(self, repos: List[dict]) -> List[Source]:
        """
        Convert JSON repository data to Source objects.
        
        Args:
            repos: List of repository dictionaries
            
        Returns:
            List of Source objects
        """
        sources = []
        
        for repo in repos:
            # Extract fields - handle both formats
            # Format 1: repo_name, repo_url, why_relevant (from query_repov2)
            # Format 2: title, url, page_summary (user mentioned)
            title = repo.get("title") or repo.get("repo_name", "Untitled Repository")
            url = repo.get("url") or repo.get("repo_url", "")
            description = repo.get("page_summary") or repo.get("why_relevant", "")
            
            # Additional metadata
            rank = repo.get("rank")
            category = repo.get("category")
            stars = repo.get("stars")
            last_update = repo.get("last_update")
            
            if not url:
                logger.warning(f"Skipping repo with no URL: {title}")
                continue
            
            # Get real timestamps from GitHub API
            timestamps = get_repo_timestamps(url)
            created_at = None
            updated_at = None
            pushed_at = None
            
            if timestamps:
                created_at = timestamps.get('created_at')
                updated_at = timestamps.get('updated_at')
                pushed_at = timestamps.get('pushed_at')
                # Format timestamps to YYYY-MM-DD format if available
                if updated_at:
                    try:
                        # Parse ISO format: 2024-01-15T10:30:00Z
                        last_update = updated_at[:10]  # Extract YYYY-MM-DD
                    except:
                        pass
            
            # Build description with additional info
            desc_parts = []
            if description:
                desc_parts.append(description)
            if category:
                desc_parts.append(f"Category: {category}")
            if stars:
                desc_parts.append(f"Stars: {stars}")
            if last_update:
                desc_parts.append(f"Last update: {last_update}")
            
            full_description = " | ".join(desc_parts) if desc_parts else None
            
            # Build metadata with timestamps
            metadata = {
                "rank": rank,
                "category": category,
                "stars": stars,
                "last_update": last_update,
                "repo_name": repo.get("repo_name", title),
            }
            
            # Add timestamp information if available
            if created_at:
                metadata["created_at"] = created_at
            if updated_at:
                metadata["updated_at"] = updated_at
            if pushed_at:
                metadata["pushed_at"] = pushed_at
            
            # Use updated_at as timestamp for Source object (extract YYYY-MM-DD)
            timestamp = None
            if updated_at:
                try:
                    timestamp = updated_at[:10]  # Extract YYYY-MM-DD from ISO format
                except:
                    timestamp = last_update
            else:
                timestamp = last_update
            
            source = Source(
                title=title,
                url=url,
                source_type=SourceType.CODE,
                platform=Platform.GITHUB,
                description=full_description,
                metadata=metadata,
                timestamp=timestamp,
            )
            sources.append(source)
        
        return sources
    
    def search(self, idea, before: Optional[str] = None, after: Optional[str] = None) -> List[Source]:
        """
        Search GitHub repositories for the given idea.
        
        Args:
            idea: Idea object to search for
            before: Optional date filter (YYYY-MM-DD format)
            after: Optional date filter (YYYY-MM-DD format)
            
        Returns:
            List of Source objects representing GitHub repositories
        """
        logger.info("Starting GitHub repository search")
        
        try:
            # Build query from idea
            query = self._build_query(idea)
            
            # Get agent and perform search
            agent = self._get_agent()
            logger.info("Calling deep_search agent...")
            result_json = agent.web_agent_answer(query)
            
            if not result_json:
                logger.warning("No results returned from deep_search agent")
                return []
            
            # Parse JSON result
            repos = self._parse_result_json(result_json)
            
            if not repos:
                logger.warning("No repositories found in parsed result")
                return []
            
            # Convert to Source objects
            sources = self._json_to_sources(repos)
            logger.info(f"Found {len(sources)} GitHub repositories")
            
            # Filter by date range if specified
            if before or after:
                filtered_sources = []
                for source in sources:
                    # Extract date from timestamp (YYYY-MM-DD format)
                    date_str = None
                    if source.timestamp:
                        # timestamp might be in YYYY-MM-DD or YYYY-MM format
                        if len(source.timestamp) >= 10:
                            date_str = source.timestamp[:10]
                        elif len(source.timestamp) >= 7:
                            # If only YYYY-MM, use first day of month
                            date_str = source.timestamp + "-01"
                    
                    if is_date_in_range(date_str, before=before, after=after):
                        filtered_sources.append(source)
                    else:
                        logger.debug(f"Filtered out repo {source.url} (date: {date_str}, range: {after} to {before})")
                
                logger.info(f"Filtered to {len(filtered_sources)} GitHub repositories within date range")
                return filtered_sources
            
            return sources
            
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {e}", exc_info=True)
            return []

