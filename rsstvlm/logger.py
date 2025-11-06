import logging

BOLD_BLUE = "\033[1;34m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format=f"{BOLD_BLUE}[%(filename)s:%(lineno)d:%(funcName)s]{RESET} %(message)s",
    handlers=[logging.FileHandler("rsstvlm.log"), logging.StreamHandler()],
)

rag_logger = logging.getLogger("rag_logger")
agent_logger = logging.getLogger("agent_logger")
mcp_logger = logging.getLogger("mcp_logger")
