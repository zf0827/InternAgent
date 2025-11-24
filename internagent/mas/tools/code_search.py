import requests
from typing import Optional, List, Dict
import os
import json
import time

GITHUB_AI_TOKEN = os.getenv('GITHUB_AI_TOKEN', "Your_GITHUB_AI_TOKEN")

def search_github_repos(query, limit=5, before: Optional[str] = None, date_field: str = 'pushed'):
    repos = []
    per_page = 10
    page = 1
    searcher = GitHubSearcher()
    while len(repos) < limit:
        results = searcher.search_repos(query=query, per_page=per_page, page=page, before=before, date_field=date_field)
        if isinstance(results, dict) and 'items' in results:
            items = results.get('items', [])
            for item in items:
                formatted_repo = {
                    "name": f"{item['owner']['login']}/{item['name']}",
                    "author": item['owner']['login'],
                    "description": item.get('description'),
                    "link": item['html_url']
                }
                repos.append(formatted_repo)
                if len(repos) >= limit:
                    break
            if len(items) < per_page:
                break
            page += 1
        else:
            break
    return_str = """
    Here are some of the repositories I found on GitHub:
    """
    for repo in repos:
        return_str += f"""
        Name: {repo['name']}
        Description: {repo['description']}
        Link: {repo['link']}
        """
    return return_str

def search_github_code(repo_owner: str, 
                      repo_name: str, 
                      query: str, 
                      language: Optional[str] = None, 
                      per_page: int = 5, 
                      page: int = 1) -> List[Dict]:
    """
    Search GitHub code based on a keyword.
    
    Args:
        repo_owner: The owner of the repository
        repo_name: The name of the repository
        query: The keyword to search for
        language: The programming language to filter by, optional
        per_page: The number of results per page, optional
        page: The page number, optional
        
    Returns:
        List[Dict]: The search results list
    """
    searcher = GitHubSearcher()
    results = searcher.search_code(repo_owner, repo_name, query, language, per_page, page)
    # print(results)
    if 'items' not in results:
        return []
        
    # Extract useful information
    formatted_results = []
    for item in results['items']:
        response = requests.get(item['url'])
        if response.status_code == 200:
            download_url = response.json()['download_url']
            response = requests.get(download_url)
            if response.status_code == 200:
                content = response.text
            else:
                content = ""
        else:
            content = ""
        formatted_results.append({
            'name': item['name'],
            'path': item['path'],
            'url': item['html_url'],
            'repository': item['repository']['full_name'],
            'content_url': item['url'],
            'content': content
        })
    return json.dumps(formatted_results, indent=4)


class GitHubSearcher:
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub searcher
        
        Args:
            token: GitHub Personal Access Token, optional
        """
        self.session = requests.Session()
        if not token:
            token = os.getenv('GITHUB_AI_TOKEN')
        if token:
            self.session.headers.update({
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def search_code(self, 
                    repo_owner: str, 
                    repo_name: str, 
                    query: str, 
                    language: Optional[str] = None,
                    per_page: int = 5, 
                    page: int = 1) -> Dict:
        """搜索代码"""
        base_url = "https://api.github.com/search/code"
        
        # 构建查询
        q = f"repo:{repo_owner}/{repo_name} {query}"
        if language:
            q += f" language:{language}"
        
        params = {
            'q': q,
            'per_page': min(per_page, 100),  # 确保不超过最大限制
            'page': page
        }
        
        try:
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            
            # 处理速率限制
            self._handle_rate_limit(response.headers)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f"Request failed: {str(e)}",
                'items': []
            }
    
    def _handle_rate_limit(self, headers: Dict):
        """处理 API 速率限制"""
        if 'X-RateLimit-Remaining' in headers:
            remaining = int(headers['X-RateLimit-Remaining'])
            if remaining < 10:
                reset_time = int(headers['X-RateLimit-Reset'])
                sleep_time = reset_time - time.time()
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 5))  # 最多等待5秒

    def search_repos(self, query: str, per_page: int = 10, page: int = 1, before: Optional[str] = None, date_field: str = 'pushed') -> Dict:
        base_url = "https://api.github.com/search/repositories"
        q = query
        if before:
            q = f"{query} {date_field}:<{before}"
        params = {
            'q': q,
            'per_page': min(per_page, 100),
            'page': page
        }
        try:
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            self._handle_rate_limit(response.headers)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f"Request failed: {str(e)}",
                'items': []
            }
