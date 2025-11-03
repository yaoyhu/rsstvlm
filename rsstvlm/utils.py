import os

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

LLM = os.getenv("LLM")
EMBEDDING_LLM = os.getenv("EMBEDDING_LLM")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_BASE")
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
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

if __name__ == "__main__":
    # Test LLM, written by Gemini 2.5 Pro
    print("Testing LLM connection...")
    try:
        response = llm.complete("Say hi.")
        print(f"LLM response: {response}")
        print("✅ LLM connection successful.")
    except Exception as e:
        print(f"❌ LLM connection failed: {e}")

    print("-" * 20)

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
