from abc import ABC, abstractmethod

from rsstvlm.utils import embedding, llm


class BaseAgent(ABC):
    """
    Base Agent class for custom agents, like RAG agents.
    """

    def __init__(self):
        self.llm = llm
        self.embedding_model = embedding
        self.tools = self._setup_tools()

    @abstractmethod
    def _setup_tools(self) -> list:
        """Must be implmented by Child class."""
        pass
