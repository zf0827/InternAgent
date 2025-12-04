import json
from typing import Annotated, Optional, Callable

from .web_browser import WebBrowser


class AgentToolLibrary:
    def __init__(self, llm_config=None, code_execution_config=None, tool_list=None, chat_history_provider: Optional[Callable[[], dict]] = None):
        if tool_list is None:
            tool_list = {"agent_coder": False, "deep_search": False}
        self.chat_history = {}
        self.chat_history_provider = chat_history_provider
        self.llm_config = llm_config
        self.code_execution_config = code_execution_config
        self.tool_list = tool_list
        if self.tool_list.get("deep_search"):
            self.web_browser = WebBrowser()

    def update_chat_history(self, chat_history):
        self.chat_history = chat_history

    def get_current_chat_history(self) -> dict:
        result = self.chat_history.copy()
        if self.chat_history_provider:
            try:
                dynamic_history = self.chat_history_provider()
                if dynamic_history:
                    if not isinstance(dynamic_history, dict):
                        dynamic_history = {"chat_history": dynamic_history}
                    self.chat_history = dynamic_history
                    return dynamic_history
            except Exception:
                pass
        return result

    async def searching(self, query: Annotated[str, "The query content to search for"]) -> str:
        return await self.web_browser.searching(query)

    async def browsing(self, query: Annotated[str, "The purpose of browsing this webpage"], url: Annotated[str, "The URL of the webpage to browse"]) -> str:
        return await self.web_browser.browsing(query, url)

