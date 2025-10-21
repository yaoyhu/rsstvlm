class BaseTool:
    # A unique name for the tool
    NAME = "base_tool"
    # A brief description of what the tool does
    DESCRIPTION = "This is a base tool and should not be used directly."


class MCPTools:
    def __init__(self):
        from rsstvlm.services.rag.baseline_rag import BaselineRAG
        from rsstvlm.services.tools.plot import H5Plot

        # TODO: add web search
        self.h5_plot = H5Plot()
        self.rag = BaselineRAG()
        # self.add = AddTool()
