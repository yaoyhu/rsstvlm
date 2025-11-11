import logging

BOLD_BLUE = "\033[1;34m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format=f"{BOLD_BLUE}%(asctime)s [%(filename)s:%(lineno)d:%(funcName)s]{RESET} %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("rsstvlm.log"), logging.StreamHandler()],
)

# Suppress HTTP request logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

rag_logger = logging.getLogger("rag_logger")
agent_logger = logging.getLogger("agent_logger")
mcp_logger = logging.getLogger("mcp_logger")
