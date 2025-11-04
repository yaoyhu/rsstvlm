class BaseTool:
    # A unique name for the tool
    NAME = "base_tool"
    # A brief description of what the tool does
    DESCRIPTION = "This is a base tool and should not be used directly."


class MCPTools:
    def __init__(self):
        from rsstvlm.services.graphrag.pipeline import GraphRAGPipeline

        self.graphrag = GraphRAGPipeline()
