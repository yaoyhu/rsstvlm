from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store.retrievers import (
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from rsstvlm.utils import deepseek


class CustomRetriever(BaseRetriever):
    """Hybrid retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: BaseRetriever,
        mode: str = "OR",
    ) -> None:
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


if __name__ == "__main__":
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
    from rsstvlm.services.graphrag.pipeline import GraphRAGPipeline

    kg = GraphRAGPipeline()

    # 先确保 index 存在
    kg.build_index(
        file_path="/satellite/d3/yaoyhu/rsstvlm/grobid/",
        exist=True,  # 从已有图加载
    )

    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=kg.storage_context,
        llm=deepseek,
        verbose=True,
    )

    # 从 PropertyGraphIndex 创建 vector retriever
    vector_retriever = kg.index.as_retriever(
        include_text=True,
        similarity_top_k=10,
    )

    # 创建混合 retriever
    hybrid_retriever = CustomRetriever(
        vector_retriever=vector_retriever,
        kg_retriever=graph_rag_retriever,
        mode="OR",
    )

    # 创建查询引擎
    response_synthesizer = get_response_synthesizer(llm=deepseek)
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
    )

    # 测试查询
    response = query_engine.query("What causes NO2 pollution?")
    print(response)
