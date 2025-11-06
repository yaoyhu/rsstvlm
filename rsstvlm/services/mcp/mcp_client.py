import asyncio
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rsstvlm.logger import mcp_logger


class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self):
        """Connect to an MCP server, return the tools"""
        server_module = "rsstvlm.services.mcp.mcp_server"
        server_params = StdioServerParameters(
            command="python", args=["-m", server_module], env=os.environ.copy()
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        mcp_logger.info(
            "Connected to MCP server with tools: %s",
            [tool.name for tool in tools],
        )
        return tools

    async def call_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Call a tool and get the result."""
        result = await self.session.call_tool(tool_name, tool_args)
        return result.content

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    def __getattr__(self, name):
        """Dynamically create a callable for any tool."""

        async def tool_caller(**kwargs):
            return await self.call_tool(name, kwargs)

        return tool_caller


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        # demo use only
        result1 = await client.ret_retrieval()
        result2 = await client.ret_images()

        print(f"PLOT return: {result2[0].text}")
        print(f"RAG return: {result1[0].text}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
