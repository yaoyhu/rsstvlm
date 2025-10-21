from sentence_transformers import SentenceTransformer
from transformers import AutoModelForImageTextToText


def get_llm_model(thinking: bool):
    """
    Return the Qwen3-VL-30B model configured for optional thinking mode.

    :thinking (bool): Select the thinking variant when True, otherwise use the instruct variant.
    """
    model_id = (
        "Qwen/Qwen3-VL-30B-A3B-Thinking"
        if thinking
        else "Qwen/Qwen3-VL-30B-A3B-Instruct"
    )
    return AutoModelForImageTextToText.from_pretrained(
        model_id, dtype="auto", device_map="auto", trust_remote_code=True
    )


def get_embedding_model():
    """
    TODO: use Qwen3-Embedding-8B for now.
    """
    return SentenceTransformer("Qwen/Qwen3-Embedding-8B")
