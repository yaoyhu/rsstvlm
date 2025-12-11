import os

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

# wille be removed
LLM = os.getenv("LLM")
EMBEDDING_LLM = os.getenv("EMBEDDING_LLM")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_BASE")

### neo4j
NEO4j_USR = os.getenv("NEO4j_USR")
NEO4j_PASSWD = os.getenv("NEO4j_PASSWD")
NEO4j_PATH = os.getenv("NEO4j_PATH")

### qwen
QWEN3_VL = os.getenv("qwen3-vl-32b-instruct")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_BASE = os.getenv("QWEN_API_BASE")

### deepseek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")

### tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

### for convenience
LLM_MODEL = "qwen3-vl-30b"
EMBEDDING_MOEDL = "qwen3-embedding"
QWEN3_EMBEDDING_8B_API_BASE = "http://localhost:8001/v1/"
QWEN3_VL_30B_API_BASE = "http://localhost:8003/v1/"

deepseek_agent = OpenAILike(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_API_BASE,
    is_chat_model=True,
    is_function_calling_model=True,
)

deepseek = OpenAILike(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_API_BASE,
    is_chat_model=True,
    is_function_calling_model=True,
)

qwen3_vl_function = OpenAILike(
    model="qwen3-vl-32b-instruct",
    api_key=QWEN_API_KEY,
    api_base=QWEN_API_BASE,
    max_tokens=1024,
    is_chat_model=True,
    is_function_calling_model=True,
)  # for function calling

qwen3_vl = OpenAILike(
    model="qwen3-vl-32b-instruct",
    api_key=QWEN_API_KEY,
    api_base=QWEN_API_BASE,
    max_tokens=1024,
    is_chat_model=True,
    is_function_calling_model=False,
)  # for final visual answer

qwen3_plus = OpenAILike(
    model="qwen-plus",
    api_key=QWEN_API_KEY,
    api_base=QWEN_API_BASE,
    max_tokens=1024,
    is_chat_model=True,
)  # for graph rag

# vllm serve
qwen3_embedding_8b = OpenAIEmbedding(
    model="text-embedding-3-small",  # this is a workaround
    model_name=EMBEDDING_MOEDL,
    api_key="not-needed",
    api_base=QWEN3_EMBEDDING_8B_API_BASE,
)

qwen3_vl_30b = OpenAILike(
    model=LLM_MODEL,
    api_key="not-needed",
    api_base=QWEN3_VL_30B_API_BASE,
    max_tokens=1024,
    is_chat_model=True,
)  # for final visual answer with vllm

qwen3_vl_30b_function = OpenAILike(
    model=LLM_MODEL,
    api_key="not-needed",
    api_base=QWEN3_VL_30B_API_BASE,
    max_tokens=1024,
    is_chat_model=True,
    is_function_calling_model=True,
)  # for function calling
