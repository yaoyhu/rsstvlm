import asyncio

from fastmcp import Client

from rsstvlm.logger import mcp_logger


class MCPClient:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/mcp"

    async def connect_to_server(self):
        """Connect to the server, list available tools, and disconnect."""
        async with Client(self.base_url) as client:
            tools = await client.list_tools()
            mcp_logger.info(
                "Available tools: %s",
                [tool.name for tool in tools],
            )
            return tools

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
    async with MCPClient() as client:
        await client.connect_to_server()


if __name__ == "__main__":
    asyncio.run(main())
