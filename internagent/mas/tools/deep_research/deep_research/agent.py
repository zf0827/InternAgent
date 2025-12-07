import json
import asyncio
from datetime import datetime
from textwrap import dedent
from copy import deepcopy

import tiktoken
import importlib

from .tools.agent_tool_library import AgentToolLibrary
from .utils.toolkits import register_toolkits
from .prompts.deepsearch_prompt import EXECUTOR_SYSTEM_PROMPT, DEEP_SEARCH_SYSTEM_PROMPT, DEEP_SEARCH_CONTEXT_SUMMARY_PROMPT, DEEP_SEARCH_RESULT_REPORT_PROMPT
from .config import get_llm_config
from .utils.tools_util import get_autogen_message_history
from .utils.ds_conversion import patch_agent_client, convert_dsml_tool_calls_to_openai_format


def get_researcher_system_message():
    return DEEP_SEARCH_SYSTEM_PROMPT.format(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class AutogenDeepSearchAgent:
    def __init__(self, llm_config=None, code_execution_config=None, return_chat_history=False, save_log=False, enable_xml_conversion_debug=False):
        self.llm_config = get_llm_config(service_type="deepsearch") if llm_config is None else llm_config
        self.code_execution_config = {"work_dir": "coding", "use_docker": False} if code_execution_config is None else code_execution_config
        self.return_chat_history = return_chat_history
        self.save_log = save_log
        self.enable_xml_conversion_debug = enable_xml_conversion_debug
        self.max_tool_messages_before_summary = 2
        self.current_tool_call_count = 0
        self.token_limit = 2000
        self.encoding = tiktoken.get_encoding("cl100k_base")

        AssistantAgent = self._import_assistant_agent()
        UserProxyAgent = self._import_user_proxy_agent()

        # Improved termination condition: exclude tool response messages to avoid false triggers from empty arrays, etc.
        def is_termination_msg_func(x):
            # If it's a tool response, should not trigger termination
            if x.get("role") == "tool":
                return False
            content = x.get("content", "")
            if not content:
                return False
            # Check if contains explicit termination marker
            if "<TERMINATE>" in content:
                return True
            # Check if it's "TERMINATE" or similar short termination message
            # But exclude JSON arrays and other tool return results
            if content.strip() == "TERMINATE" or (len(content.split("TERMINATE")[-1].strip()) < 5 and "TERMINATE" in content):
                return True
            return False

        self.researcher = AssistantAgent(
            name="researcher",
            system_message=get_researcher_system_message(),
            llm_config=self.llm_config,
            is_termination_msg=is_termination_msg_func,
        )

        self.executor = UserProxyAgent(
            name="executor",
            system_message=EXECUTOR_SYSTEM_PROMPT,
            human_input_mode="NEVER",
            llm_config=self.llm_config,
            code_execution_config=False,  # Disable code execution to avoid mistakenly executing JSON code blocks
            is_termination_msg=is_termination_msg_func,
        )

        self.agent_tool_library = AgentToolLibrary(
            llm_config=self.llm_config,
            code_execution_config=self.code_execution_config,
            tool_list={"agent_coder": False, "deep_search": True},
            chat_history_provider=self._get_researcher_chat_history,
        )

        self._register_tools()
        self._patch_agent_message_handlers()
        self._patch_llm_client_to_convert_xml_tool_calls()

    def _register_tools(self):
        register_toolkits([
            self.agent_tool_library.searching,
            self.agent_tool_library.browsing,
        ], self.researcher, self.executor)

    def _patch_agent_message_handlers(self):
        original_executor_receive = self.executor._process_received_message
        original_researcher_receive = self.researcher._process_received_message

        def executor_receive_with_summary(message, sender, silent):
            message_history = deepcopy(self.executor.chat_messages[self.researcher]) if self.researcher in self.executor.chat_messages else []
            if sender == self.researcher and len(message_history) > 1:
                if "tool_responses" in message_history[-1] and "tool_calls" in message_history[-2]:
                    self._summarize_tool_response(message_history, message)
                    self.current_tool_call_count += 1
            return original_executor_receive(message, sender, silent)

        def researcher_receive_with_summary(message, sender, silent):
            if sender == self.executor and self.current_tool_call_count >= self.max_tool_messages_before_summary:
                self.current_tool_call_count = 0
            return original_researcher_receive(message, sender, silent)

        self.executor._process_received_message = executor_receive_with_summary
        self.researcher._process_received_message = researcher_receive_with_summary

    def _patch_llm_client_to_convert_xml_tool_calls(self):
        """
        Patch llm_client to convert XML format tool calls
        
        This method intercepts completions returned from API endpoints and uses
        convert_dsml_tool_calls_to_openai_format from ds_conversion module to convert
        DeepSeek-V3.2 style DSML tool calls to OpenAI format.
        
        How it works:
        1. Intercept responses from client.create()
        2. Check if response contains XML format tool calls (in content)
        3. If present, use convert_dsml_tool_calls_to_openai_format for conversion
        4. Return converted completion (with standard format tool_calls)
        """
        # Patch researcher's client
        patch_agent_client(
            self.researcher,
            "researcher",
            convert_dsml_tool_calls_to_openai_format,
            self.enable_xml_conversion_debug
        )
        
        # Patch executor's client (if exists)
        patch_agent_client(
            self.executor,
            "executor",
            convert_dsml_tool_calls_to_openai_format,
            self.enable_xml_conversion_debug
        )

    def _summarize_tool_response(self, chat_history, current_message):
        tool_calls = chat_history[-2]["tool_calls"]
        tool_responses_list = chat_history[-1]["tool_responses"]

        try:
            del self.executor.chat_messages[self.researcher][-1]["content"]
            del self.researcher.chat_messages[self.executor][-2]["content"]
        except Exception:
            pass

        if not isinstance(tool_responses_list, list):
            tool_responses_list = [tool_responses_list]

        summary_list = []

        for tool_responses in tool_responses_list:
            if isinstance(tool_responses, (list, dict)):
                tool_responses = json.dumps(tool_responses)
            elif not isinstance(tool_responses, str):
                tool_responses = str(tool_responses)

            if isinstance(tool_calls, (list, dict)):
                tool_calls_str = json.dumps(tool_calls)
            else:
                tool_calls_str = str(tool_calls)

            token_count = len(self.encoding.encode(tool_responses))
            if token_count < self.token_limit:
                continue

            history_json = json.dumps(chat_history[:-2], ensure_ascii=False)
            response_summary = self._generate_summary_for_search_result(history_json, tool_responses)
            summary_list.append(response_summary)

        try:
            for idx, summary in enumerate(summary_list):
                self.executor.chat_messages[self.researcher][-1]["tool_responses"][idx]["content"] = summary
                self.researcher.chat_messages[self.executor][-2]["tool_responses"][idx]["content"] = summary
        except Exception:
            pass

    def _generate_summary_for_search_result(self, messages, tool_responses):
        summary_prompt = DEEP_SEARCH_CONTEXT_SUMMARY_PROMPT.format(tool_responses=tool_responses, messages=messages)
        OpenAIWrapper = self._import_openai_wrapper()
        client = OpenAIWrapper(**self.llm_config)
        messages_list = [{"role": "user", "content": summary_prompt}]
        response = client.create(messages=messages_list)
        return response.choices[0].message.content

    def _get_researcher_chat_history(self) -> dict:
        try:
            result = {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            if hasattr(self, "original_query"):
                result["original_query"] = self.original_query
            if hasattr(self.researcher, "chat_messages") and self.executor in self.researcher.chat_messages:
                chat_messages = self.researcher.chat_messages[self.executor]
                chat_messages = json.dumps(chat_messages, ensure_ascii=False)
                result["chat_history"] = chat_messages
            return result
        except Exception:
            return {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "error": "Failed to get chat history"}

    async def deep_search(self, query: str) -> str:
        if not hasattr(self, "researcher") or not hasattr(self, "executor"):
            self._ensure_agents()
        self.current_tool_call_count = 0
        self.original_query = query

        initial_message = dedent(f"""
        I need you to help me research the following question in depth:

        {query}
        """)

        self.agent_tool_library.update_chat_history({"original_query": self.original_query})
        self.researcher.update_system_message(get_researcher_system_message())

        chat_result = await self.executor.a_initiate_chat(
            self.researcher,
            message=initial_message,
            max_turns=30,
            summary_method="reflection_with_llm",
            summary_args={"summary_prompt": DEEP_SEARCH_RESULT_REPORT_PROMPT},
        )
        final_answer = self._extract_final_answer(chat_result)
        if self.return_chat_history:
            return final_answer, get_autogen_message_history(chat_result.chat_history)
        return final_answer

    def _extract_final_answer(self, chat_result) -> str:
        final_answer = chat_result.summary
        if isinstance(final_answer, dict):
            final_answer = final_answer.get("content", "")
        if final_answer is None:
            final_answer = ""
        final_answer = final_answer.strip().lstrip()
        messages = chat_result.chat_history
        final_content = messages[-1].get("content", "") if messages else ""
        if final_content:
            final_content = final_content.strip().lstrip()
        if final_answer == "":
            final_answer = final_content
        return final_answer

    def web_agent_answer(self, query: str) -> str:
        return asyncio.run(self.deep_search(query))

    async def run(self, query: str):
        self.return_chat_history = True
        final_answer, chat_result = await self.deep_search(query)
        return {"final_answer": final_answer, "trajectory": chat_result}

    def _import_assistant_agent(self):
        try:
            mod = importlib.import_module("autogen.agentchat.assistant_agent")
            return getattr(mod, "AssistantAgent")
        except Exception:
            mod = importlib.import_module("autogen")
            return getattr(mod, "AssistantAgent")

    def _import_user_proxy_agent(self):
        try:
            mod = importlib.import_module("autogen.agentchat.user_proxy_agent")
            return getattr(mod, "UserProxyAgent")
        except Exception:
            mod = importlib.import_module("autogen")
            return getattr(mod, "UserProxyAgent")

    def _import_openai_wrapper(self):
        try:
            mod = importlib.import_module("autogen.oai")
            return getattr(mod, "OpenAIWrapper")
        except Exception:
            raise ImportError("autogen OpenAIWrapper is required for summary generation")

    def _ensure_agents(self):
        AssistantAgent = self._import_assistant_agent()
        UserProxyAgent = self._import_user_proxy_agent()
        # Improved termination condition: exclude tool response messages to avoid false triggers from empty arrays, etc.
        def is_termination_msg_func(x):
            # If it's a tool response, should not trigger termination
            if x.get("role") == "tool":
                return False
            content = x.get("content", "")
            if not content:
                return False
            # Check if contains explicit termination marker
            if "<TERMINATE>" in content:
                return True
            # Check if it's "TERMINATE" or similar short termination message
            # But exclude JSON arrays and other tool return results
            if content.strip() == "TERMINATE" or (len(content.split("TERMINATE")[-1].strip()) < 5 and "TERMINATE" in content):
                return True
            return False

        self.researcher = AssistantAgent(
            name="researcher",
            system_message=get_researcher_system_message(),
            llm_config=self.llm_config,
            is_termination_msg=is_termination_msg_func,
        )
        self.executor = UserProxyAgent(
            name="executor",
            system_message=EXECUTOR_SYSTEM_PROMPT,
            human_input_mode="NEVER",
            llm_config=self.llm_config,
            code_execution_config=False,  # Disable code execution to avoid mistakenly executing JSON code blocks
            is_termination_msg=is_termination_msg_func,
        )
        self._register_tools()
        self._patch_agent_message_handlers()
        self._patch_llm_client_to_convert_xml_tool_calls()
