import os

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

LLM = os.getenv("LLM")
EMBEDDING_LLM = os.getenv("EMBEDDING_LLM")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_BASE")
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# siliconflow
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE")

### vllm
QWEN3_EMBEDDING_8B = os.getenv("QWEN3_EMBEDDING_8B")
VLLM_API_BASE = os.getenv("VLLM_API_BASE")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")

### neo4j
NEO4j_USR = os.getenv("NEO4j_USR")
NEO4j_PASSWD = os.getenv("NEO4j_PASSWD")
NEO4j_PATH = os.getenv("NEO4j_PATH")

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

qwen3_vl_function = OpenAILike(
    model="Qwen/Qwen3-VL-32B-Instruct",
    api_key=SILICONFLOW_API_KEY,
    api_base=SILICONFLOW_API_BASE,
    max_tokens=1024,
    is_chat_model=True,
    is_function_calling_model=True,
)

qwen3_vl = OpenAILike(
    model="Qwen/Qwen3-VL-32B-Instruct",
    api_key=SILICONFLOW_API_KEY,
    api_base=SILICONFLOW_API_BASE,
    max_tokens=1024,
    is_chat_model=True,
    is_function_calling_model=True,
)

# TODO: vllm serve qwen

embedding = OpenAIEmbedding(
    model="text-embedding-3-small",  # this is a workaround
    model_name=QWEN3_EMBEDDING_8B,
    api_key=VLLM_API_KEY,
    api_base=VLLM_API_BASE,
)


if __name__ == "__main__":
    # Test LLM, written by Gemini 2.5 Pro
    # print("Testing LLM connection...")
    # try:
    #     response = llm.complete("Say hi.")
    #     print(f"LLM response: {response}")
    #     print("✅ LLM connection successful.")
    # except Exception as e:
    #     print(f"❌ LLM connection failed: {e}")

    # print("-" * 20)

    # Test Embedding
    print("Testing Embedding connection...")
    try:
        embedding_vector = embedding.get_text_embedding(
            "This is a test sentence."
        )
        print(f"Embedding vector dimension: {len(embedding_vector)}")
        print(f"First 5 dims of embedding vector: {embedding_vector[:5]}")
        print("✅ Embedding connection successful.")
    except Exception as e:
        print(f"❌ Embedding connection failed: {e}")
