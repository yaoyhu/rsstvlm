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
        mcp.tool(pipeline.hybrid_query)

        # air matters
        from rsstvlm.services.tools.airmatters import AirMatters

        am = AirMatters()
        mcp.tool(am.current_air_condition)
        mcp.tool(am.place_search)
        mcp.tool(am.sub_places)
        mcp.tool(am.get_standard)
        mcp.tool(am.aqi_forecast)
        mcp.tool(am.history_air_condition)
        mcp.tool(am.nearby_place)
        mcp.tool(am.nearby_air_condition)
        mcp.tool(am.batch_air_condition)
        mcp.tool(am.map)
        mcp.tool(am.heatmap)

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
