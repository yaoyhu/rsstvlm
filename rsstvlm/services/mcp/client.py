import asyncio

from fastmcp import Client
from llama_index.core.tools import FunctionTool

from rsstvlm.logger import mcp_logger


class MCPClient:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/mcp"

    def _mcp_tool_to_llamaindex(self, mcp_tool) -> FunctionTool:
        """Convert an MCP tool to a LlamaIndex FunctionTool."""

        async def tool_function(**kwargs) -> str:
            """Dynamically generated tool function that calls the MCP server."""
            try:
                result = await self.call_tool(mcp_tool.name, kwargs)
                return str(result)
            except Exception as e:
                return f"Error calling tool {mcp_tool.name}: {e!s}"

        tool_function.__name__ = mcp_tool.name
        tool_function.__doc__ = (
            mcp_tool.description or f"Tool: {mcp_tool.name}"
        )

        return FunctionTool.from_defaults(
            async_fn=tool_function,
            name=mcp_tool.name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        )

    async def connect_to_server(self) -> list[FunctionTool]:
        """Connect to the server and return LlamaIndex tools."""
        async with Client(self.base_url) as client:
            mcp_tools = await client.list_tools()
            mcp_logger.info(
                "Available tools: %s",
                [tool.name for tool in mcp_tools],
            )

            llamaindex_tools = [
                self._mcp_tool_to_llamaindex(tool) for tool in mcp_tools
            ]
            return llamaindex_tools

    async def call_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Connect, call a specific tool, and return the result."""
        async with Client(self.base_url) as client:
            result = await client.call_tool(tool_name, tool_args)
            return result.content

    def __getattr__(self, name):
        """Dynamically create a callable for any tool."""

        async def tool_caller(**kwargs):
            return await self.call_tool(name, kwargs)

        return tool_caller


async def main():
    client = MCPClient()
    tools = await client.connect_to_server()
    mcp_logger.info("LlamaIndex tools: %s", [t.metadata.name for t in tools])


if __name__ == "__main__":
    asyncio.run(main())
