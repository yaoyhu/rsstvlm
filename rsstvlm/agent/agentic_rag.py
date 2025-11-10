import asyncio
import json
import re
from typing import Any

from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

from rsstvlm.agent.base_agent import BaseAgent
from rsstvlm.logger import agent_logger
from rsstvlm.services.mcp.client import MCPClient


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

        while True:
            response: ChatResponse = self.llm_function.chat(
                messages=messages,
                tools=available_tools,
            )
            assistant_message = response.message

            messages.append(assistant_message)

            assistant_text = self._message_text(assistant_message)
            if assistant_text:
                final_chunks.append(assistant_text)

            # handle tool calls
            tool_call_blocks = [
                block
                for block in getattr(assistant_message, "blocks", [])
                if block.block_type == "tool_call"
            ]

            if not tool_call_blocks:
                agent_logger.info("No tools chosen, answer by LLM itself.")
                break

            for block in tool_call_blocks:
                tool_name = block.tool_name
                tool_kwargs_str = block.tool_kwargs

                agent_logger.info(
                    "Calling tool '%s' with args: %s",
                    tool_name,
                    tool_kwargs_str,
                )

                try:
                    tool_args = json.loads(tool_kwargs_str.strip())
                except json.JSONDecodeError:
                    agent_logger.error(
                        "Failed to parse tool arguments JSON: %s",
                        tool_kwargs_str,
                    )
                    # Append an error message for the LLM to see.
                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=f"Error: Could not parse arguments for tool {tool_name}.",
                            additional_kwargs={
                                "tool_call_id": block.tool_call_id
                            },
                        )
                    )
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
                        role=MessageRole.TOOL,
                        content=result_text,
                        additional_kwargs={"tool_call_id": block.tool_call_id},
                    )
                )

        agent_logger.info("Agent finished streaming.")

        return "\n".join(chunk for chunk in final_chunks if chunk)

    def _format_tools(self) -> list[dict[str, Any]]:
        """
        Reference:
            - https://qwen.readthedocs.io/zh-cn/latest/framework/function_call.html
            - mcp.types.Tool
        """
        formatted_tools = []
        for tool in self.tools:
            parameters = getattr(tool, "inputSchema", None)

            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}

            tool_name = getattr(tool, "name", None)
            tool_description = getattr(tool, "description", "") or ""

            if not tool_name:
                agent_logger.warning("Skipping a tool because it has no name.")
                continue

            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": parameters,
                    },
                }
            )
        return formatted_tools

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
