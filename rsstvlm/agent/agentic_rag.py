import asyncio
import json
import re
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole

from rsstvlm.agent.base_agent import BaseAgent
from rsstvlm.logger import agent_logger
from rsstvlm.services.mcp.mcp_client import MCPClient


class AgenticRAG(BaseAgent):
    def __init__(self):
        super().__init__()
        self.client = MCPClient()

    async def _setup_tools(self):
        """Asynchronously connect to the server and get the tools."""
        tools = await self.client.connect_to_server()
        return tools

    @classmethod
    async def create(cls):
        """Async factory to create and initialize an agent instance."""
        agent = cls()
        agent.tools = await agent._setup_tools()
        return agent

    async def stream(
        self,
        query: str,
    ):
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are an agentic RAG system that can use tools to answer user queries.",
            ),
            ChatMessage(role=MessageRole.USER, content=query),
        ]

        # align with QWEN's documentation
        available_tools = self._format_tools()

        final_chunks: list[str] = []

        try:
            response = self.llm_function.chat(
                messages=messages,
                tools=available_tools,
            )

            while True:
                assistant_message = response.message
                assistant_text = self._message_text(assistant_message)
                if assistant_text:
                    final_chunks.append(assistant_text)

                tool_calls = self._extract_tool_calls(response)
                if not tool_calls:
                    agent_logger.info(
                        "No tools been chosen, answer by LLM itself."
                    )
                    break

                agent_logger.info("Calling tools: %s", tool_calls)
                messages.append(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=assistant_text,
                        additional_kwargs={"tool_calls": tool_calls},
                    )
                )

                for tool_call in tool_calls:
                    tool_name, tool_args = self._parse_tool_call(tool_call)
                    if not tool_name:
                        continue

                    result_content = await self.client.call_tool(
                        tool_name, tool_args
                    )
                    result_text = self._render_tool_content(result_content)
                    final_chunks.append(
                        f"\n[Tool {tool_name} used with args: {self._format_args(tool_args)}]\n{result_text}"
                    )

                    messages.append(
                        ChatMessage(
                            role=MessageRole.USER,
                            content=result_text,
                        )
                    )
            response = self.llm.chat(messages=messages, tools=available_tools)
            agent_logger.info(
                "Final response: %s",
                response.raw.choices[0].message.model_extra.get(
                    "reasoning_content"
                ),
            )
        finally:
            await self.client.cleanup()

        return "\n".join(chunk for chunk in final_chunks if chunk)

    def _format_tools(self) -> list[dict[str, Any]]:
        """
        Format tools for Qwen function calling, refer to:
        https://qwen.readthedocs.io/zh-cn/latest/framework/function_call.html
        """
        formatted = []
        for tool in self.tools:
            raw_parameters = getattr(tool, "inputSchema", None)

            if hasattr(raw_parameters, "model_dump"):
                parameters = raw_parameters.model_dump(exclude_none=True)
            elif isinstance(raw_parameters, dict):
                parameters = raw_parameters
            else:
                parameters = {}

            if not isinstance(parameters, dict):
                parameters = {}

            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            parameters.setdefault("required", [])

            formatted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": getattr(tool, "description", "") or "",
                        "parameters": parameters,
                    },
                }
            )
        return formatted

    @staticmethod
    def _extract_tool_calls(response) -> list[Any]:
        tool_calls: list[Any] = []
        for source in (
            getattr(response, "additional_kwargs", None),
            getattr(response.message, "additional_kwargs", None),
        ):
            if not source:
                continue
            calls = source.get("tool_calls")
            if isinstance(calls, list):
                tool_calls.extend(calls)

        if tool_calls:
            return tool_calls

        raw_response = getattr(response, "raw", None)
        if not raw_response:
            return tool_calls

        choices = getattr(raw_response, "choices", []) or []
        for choice in choices:
            message = getattr(choice, "message", None)
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                tool_calls.extend(
                    AgenticRAG._parse_reasoning_tool_calls(reasoning_content)
                )
        return tool_calls

    @staticmethod
    def _parse_reasoning_tool_calls(
        reasoning_content: str,
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        call_index = 0
        for match in re.finditer(pattern, reasoning_content, flags=re.S):
            payload = match.group(1)
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue

            name = parsed.get("name")
            arguments = parsed.get("arguments", {})
            serialized_args = AgenticRAG._format_args(arguments)
            tool_calls.append(
                {
                    "id": f"call_{call_index}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serialized_args,
                    },
                }
            )
            call_index += 1
        return tool_calls

    @staticmethod
    def _parse_tool_call(tool_call: Any) -> tuple[str | None, dict[str, Any]]:
        function_call = {}
        if isinstance(tool_call, dict):
            function_call = tool_call.get("function", {}) or {}
        elif hasattr(tool_call, "function"):
            function_call = getattr(tool_call, "function", {}) or {}

        name = (
            function_call.get("name")
            if isinstance(function_call, dict)
            else None
        )
        raw_args = (
            function_call.get("arguments")
            if isinstance(function_call, dict)
            else {}
        )

        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                args = (
                    parsed if isinstance(parsed, dict) else {"input": parsed}
                )
            except json.JSONDecodeError:
                args = {"input": raw_args}
        else:
            args = {}

        return name, args

    @staticmethod
    def _render_tool_content(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [AgenticRAG._render_tool_content(item) for item in content]
            return "\n".join(filter(None, parts))

        if hasattr(content, "model_dump"):
            content_dict = content.model_dump()
        elif isinstance(content, dict):
            content_dict = content
        else:
            return str(content)

        content_type = content_dict.get("type")
        if content_type == "text":
            return content_dict.get("text", "")
        return json.dumps(content_dict)

    @staticmethod
    def _format_args(args: dict[str, Any]) -> str:
        try:
            return json.dumps(args, ensure_ascii=True)
        except (TypeError, ValueError):
            return str(args)

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        parts = []
        for block in getattr(message, "blocks", []) or []:
            if getattr(block, "block_type", None) == "text":
                parts.append(getattr(block, "text", ""))
        if parts:
            return "\n".join(filter(None, parts))
        return message.content or ""


async def main():
    agent = await AgenticRAG.create()
    await agent.stream("Hi")


if __name__ == "__main__":
    asyncio.run(main())
