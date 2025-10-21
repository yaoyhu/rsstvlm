from abc import ABC, abstractmethod

from rsstvlm.models.llms import get_embedding_model, get_llm_model


class BaseAgent(ABC):
    """
    Base Agent class for custom agents, like RAG agents.
    """

    def __init__(self):
        self.llm = get_llm_model(thinking=False)
        self.embedding_model = get_embedding_model()
        self.tools = self._setup_tools()

    @abstractmethod
    def _setup_tools(self) -> list:
        """Must be implmented by Child class."""
        pass
