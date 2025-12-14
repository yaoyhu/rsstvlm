import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("rsstvlm.log"), logging.StreamHandler()],
)

# Suppress HTTP request logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Suppress MCP streamable HTTP logs
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

rag_logger = logging.getLogger("rag_logger")
agent_logger = logging.getLogger("agent_logger")
mcp_logger = logging.getLogger("mcp_logger")
