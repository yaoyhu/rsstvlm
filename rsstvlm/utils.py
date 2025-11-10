import os

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

# wille be removed
LLM = os.getenv("LLM")
EMBEDDING_LLM = os.getenv("EMBEDDING_LLM")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_BASE")

llm = OpenAI(
    model=LLM,
    api_key=API_KEY,
    api_base=BASE_URL,
)

embedding = OpenAIEmbedding(
    model=EMBEDDING_LLM,
    api_key=API_KEY,
    api_base=BASE_URL,
)

### neo4j
NEO4j_USR = os.getenv("NEO4j_USR")
NEO4j_PASSWD = os.getenv("NEO4j_PASSWD")
NEO4j_PATH = os.getenv("NEO4j_PATH")

### qwen
QWEN3_VL = os.getenv("qwen3-vl-32b-instruct")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_BASE = os.getenv("QWEN_API_BASE")

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

# TODO: vllm serve qwen
qwen3_embedding_8b = OpenAIEmbedding(
    model="text-embedding-3-small",  # this is a workaround
    model_name="qwen3-embedding",
    api_key="not-needed",
    api_base="http://localhost:8001/v1/",
)
