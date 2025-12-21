from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
)
from llama_index.core.indices.vector_store.retrievers import (
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from rsstvlm.utils import deepseek, qwen3_embedding_8b


class CustomRetriever(BaseRetriever):
    """Hybrid retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: BaseRetriever,
        mode: str | None = "OR",
        verbose: bool | None = False,  # show retrieved results
    ) -> None:
        super().__init__()
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        self._verbose = verbose
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        if self._verbose:
            self._print_results(vector_nodes, kg_nodes)

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

    def _print_results(
        self,
        vector_nodes: list[NodeWithScore],
        kg_nodes: list[NodeWithScore],
    ) -> None:
        print("\n" + "=" * 80)
        print("🔍 VECTOR RETRIEVAL RESULTS")
        print("=" * 80)
        for i, node in enumerate(vector_nodes, 1):
            score_str = (
                f"{node.score:.4f}" if node.score is not None else "N/A"
            )
            print(f"\n--- Result {i} (Score: {score_str}) ---")
            print(f"Metadata: {node.node.metadata}")
            # Get the original text
            text_content = node.node.get_content()
            text_preview = text_content[:1000]
            print(f"Content: {text_preview}...")


if __name__ == "__main__":
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RetrieverQueryEngine
    from rsstvlm.services.graphrag.pipeline import GraphRAGPipeline
    from rsstvlm.services.graphrag.t2c import Text2CypherRetriever

    kg = GraphRAGPipeline()
    kg.build_index(exist=True)

    vector_retriever = VectorContextRetriever(
        graph_store=kg.graph_store,
        vector_store=kg.vec_store,
        embed_model=qwen3_embedding_8b,
        include_text=True,
        path_depth=2,  # 图谱扩展深度 (1=直接关系, 2=2跳关系)
        limit=30,  # 最多返回多少个三元组
        similarity_score=0.7,  # 最低相似度阈值
        similarity_top_k=5,
        verbose=True,
    )

    kg_retriever = Text2CypherRetriever(
        graph_store=kg.graph_store,
        llm=deepseek,
        verbose=True,
    )

    query = "What is the relationship between O3 and pollution?"

    hybrid_retriever = CustomRetriever(
        vector_retriever=vector_retriever,
        kg_retriever=kg_retriever,
        mode="OR",
        verbose=True,
    )

    response_synthesizer = get_response_synthesizer(llm=deepseek)
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
    )

    query_bundle = QueryBundle(
        query_str=query,
        embedding=qwen3_embedding_8b.get_query_embedding(query),
    )
    response = query_engine.query(query_bundle)
    print(response)
