from mcp.server.fastmcp import FastMCP

from rsstvlm.services.tools.base_tool import MCPTools


class MCPServer:
    def __init__(self):
        self.mcp = FastMCP("Tools")
        self.tools = MCPTools()

        # TODO: add more tools
        self.mcp.add_tool(self.tools.rag.ret_retrieval)
        self.mcp.add_tool(self.tools.h5_plot.ret_images)

    def run(self):
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    print("🌐 Running MCP server...")
    mcp_server = MCPServer()
    mcp_server.run()
