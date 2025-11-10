from abc import ABC, abstractmethod
from typing import Any

from rsstvlm.utils import qwen3_embedding_8b, qwen3_vl, qwen3_vl_function


class BaseAgent(ABC):
    """
    Base Agent class for custom agents, like RAG agents.
    """

    def __init__(self):
        self.llm_function = qwen3_vl_function
        self.llm = qwen3_vl
        self.embedding_model = qwen3_embedding_8b
        self.tools: list[Any] = []

    @abstractmethod
    async def _setup_tools(self) -> list[Any]:
        """Asynchronously initialise any tools required by the agent."""
        raise NotImplementedError
