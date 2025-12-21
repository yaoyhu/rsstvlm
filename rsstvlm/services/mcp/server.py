from fastmcp import FastMCP

mcp = FastMCP("USTC")


class MCPServer:
    def __init__(self):
        self._load_tools()

    def _load_tools(self):
        """
        Note:
            Re-deploy MCP server after modification: `sbatch scripts/mcp_server.sh`
        Refer:
            https://gofastmcp.com/patterns/decorating-methods#instance-methods
        """
        # neo4j-related tools
        from rsstvlm.services.graphrag.pipeline import GraphRAGPipeline

        pipeline = GraphRAGPipeline()
        mcp.tool(pipeline.query)
        mcp.tool(pipeline.hybrid_query)

        # others
        from rsstvlm.services.tools.plot import H5Plot

        h5plot = H5Plot()
        mcp.tool(h5plot.plot)
        mcp.tool(h5plot.structure)
        mcp.tool(h5plot.visual_explain)

    def run(self):
        """
        Run the MCP server persistently via streamable HTTP,
        for saving my debugging time...
        """
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    server = MCPServer()
    server.run()
