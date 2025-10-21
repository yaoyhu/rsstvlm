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
                "description": tool.description,
            }
            for tool in self.tools
        ]
        print("Available tools:", available_tools)
        await self.client.cleanup()


async def main():
    agent = await AgenticRAG.create()
    await agent.stream("Hi")


if __name__ == "__main__":
    asyncio.run(main())
