import asyncio

from rsstvlm.agent.base_agent import BaseAgent
from rsstvlm.services.mcp.mcp_client import MCPClient


class AgenticRAG(BaseAgent):
    def __init__(self):
        self.client = MCPClient()
        self.tools = []

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
        # messages = [{"role": "user", "content": query}]
        available_tools = [
            {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
            }
            for tool in self.tools
        ]

        print("Available tools:", available_tools)

        # Try to find build/index and query tools automatically
        build_tool = None
        query_tool = None
        for tool in self.tools:
            n = tool.name.lower()
            if "build" in n or "index" in n:
                build_tool = tool.name
            if "query" in n:
                query_tool = tool.name

        # Test inputs
        test_path = (
            "./tests/Aligner: Efficient Alignment by Learning to Correct.txt"
        )

        # Call build tool if available. Use safe kwargs and catch errors.
        if build_tool:
            print(f"Calling build tool: {build_tool} with file {test_path}")
            try:
                build_fn = getattr(self.client, build_tool)
                # try common parameter names; server-side may expect different names
                try:
                    res = await build_fn(file_path=test_path, exist=True)
                except TypeError:
                    # fallback to other common name
                    res = await build_fn(path=test_path)
                print("Build result:", res)
            except Exception as e:
                print("Build tool call failed:", e)
        else:
            print("No build/index tool found among available tools.")

        # Call query tool if available
        if query_tool:
            print(f"Calling query tool: {query_tool} with query: {query}")
            try:
                query_fn = getattr(self.client, query_tool)
                try:
                    qres = await query_fn(query_str=query)
                except TypeError:
                    qres = await query_fn(query=query)
                print("Query result:", qres)
            except Exception as e:
                print("Query tool call failed:", e)
        else:
            print("No query tool found among available tools.")

        await self.client.cleanup()


async def main():
    agent = await AgenticRAG.create()
    await agent.stream("Hi")


if __name__ == "__main__":
    asyncio.run(main())
