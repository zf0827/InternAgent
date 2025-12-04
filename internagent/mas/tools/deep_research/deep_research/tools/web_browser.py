import os
import json
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import re

from typing_extensions import Annotated
from typing import List, Dict, Any


class WebBrowser:
    def __init__(self, max_browser_length=20000):
        self.search_engine = SerperSearchEngine()
        self.max_browser_length = max_browser_length

    async def searching(self, query: Annotated[str, "Query content to search for"]) -> str:
        try:
            result = await self.search_engine.engine_search(query, engine="google", search_num=10, web_parse=False)
            # 如果结果是空数组，返回更有意义的信息
            if result == "[]" or result.strip() == "[]":
                return json.dumps({
                    "status": "no_results",
                    "message": f"No search results found for query: {query}. Please try a different search query or adjust your search terms.",
                    "query": query,
                    "results": []
                }, ensure_ascii=False)
            return result
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error searching: {str(e)}",
                "query": query,
                "results": []
            }, ensure_ascii=False)

    async def browsing(self, query: Annotated[str, "Query string for content filtering"], url: Annotated[str, "URL of the webpage to browse"]) -> str:
        try:
            content = await self.browsing_url(url)
            # 如果返回的是错误消息，直接返回
            if isinstance(content, str) and content.startswith("Error:"):
                return json.dumps({"Input Query": query, "Search URL": url, "Search Result": content}, ensure_ascii=False)
            
            if len(content) > self.max_browser_length:
                return json.dumps({"Input Query": query, "Search URL": url, "Search Result": content[:self.max_browser_length]}, ensure_ascii=False)
            else:
                return json.dumps({"Input Query": query, "Search URL": url, "Search Result": content}, ensure_ascii=False)
        except Exception as e:
            import logging
            logging.error(f"Error in browsing method: {str(e)}", exc_info=True)
            return json.dumps({"Input Query": query, "Search URL": url, "Search Result": f"Error browsing URL: {str(e)}"}, ensure_ascii=False)

    async def browsing_url(self, url):
        if "r.jina.ai" not in url:
            url = "https://r.jina.ai/" + url

        headers = None
        if os.getenv("JINA_API_KEY"):
            headers = {
                "Authorization": "Bearer " + os.getenv("JINA_API_KEY", ""),
                "X-Engine": "direct",
                "X-Return-Format": "markdown",
                "X-Timeout": "10",
            }

        # 设置超时：总超时 30 秒，连接超时 10 秒，读取超时 20 秒
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()  # 检查 HTTP 状态码
                    content = await response.read()
        except asyncio.TimeoutError:
            return f"Error: Request timeout after 30 seconds for URL: {url}"
        except aiohttp.ClientError as e:
            return f"Error: Network error while browsing URL {url}: {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error while browsing URL {url}: {str(e)}"

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        content = await self.search_engine._clean_content(content)
        return content


class SerperSearchEngine:
    def __init__(self, chunk_size=4000, chunk_overlap=400):
        self.google_serper_url = "https://google.serper.dev/search"

    def google_search(self, query: Annotated[str, "The search query"], max_results: Annotated[int, "The maximum number of results to retrieve"] = 100) -> List[Dict[str, str]]:
        headers = {"X-API-KEY": os.environ.get("SERPER_API_KEY", ""), "Content-Type": "application/json"}
        all_results = []
        for _ in range(1):
            payload = {"q": query, "gl": "us", "hl": "en", "num": max_results}
            try:
                response = requests.post(self.google_serper_url, headers=headers, json=payload)
                response.raise_for_status()
                results = response.json()
                organic_results = [result for result in results.get("organic", [{}])]
                # 过滤掉空结果（没有 title 或 link 的）
                page_results = [
                    {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "link": r.get("link", "")} 
                    for r in organic_results 
                    if r.get("title") and r.get("link")
                ]
                all_results.extend(page_results)
            except Exception as e:
                # 记录错误但不中断，返回已收集的结果
                import logging
                logging.warning(f"Search error for query '{query}': {str(e)}")
                break
        return all_results

    async def engine_search(self, query, engine="google", search_num=10, web_parse=True, url_filter=None):
        results = self.google_search(query, max_results=search_num * 2)
        if url_filter:
            results = [res for res in results if res["link"] not in url_filter]
        results = results[: min(search_num, len(results))]
        
        # 如果结果为空，返回包含状态信息的结构
        if not results:
            empty_result = {
                "status": "no_results",
                "message": f"No search results found for query: {query}",
                "query": query,
                "results": []
            }
            return json.dumps(empty_result, ensure_ascii=False)
        
        if web_parse:
            enriched = await self._enrich_results_async(results)
            return json.dumps(enriched, ensure_ascii=False)
        return json.dumps(results, ensure_ascii=False)

    async def _parse_content_async(self, res):
        try:
            content = await WebBrowser().browsing(query="", url=res["link"])
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            res["content"] = await self._clean_content(content)
        except Exception:
            res["content"] = ""
        return res

    async def _enrich_results_async(self, results):
        tasks = [self._parse_content_async(res) for res in results]
        return await asyncio.gather(*tasks)

    async def _clean_content(self, content: str) -> str:
        content = re.sub(r"http[s]?://\S+", "", content)
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
        content = re.sub(r"<[^>]+>", "", content)
        content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", content)
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        content = "\n".join(line.strip() for line in content.split("\n") if line.strip())
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = "\n".join(line for line in content.split("\n") if len(line.split()) > 2)
        return content.strip()

