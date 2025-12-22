from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    get_response_synthesizer,
)
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,
    VectorContextRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from rsstvlm.prompts.extraction import (
    EXTRACTION,
    entities,
    relations,
    validation_schema,
)
from rsstvlm.services.graphrag.parse import load_documents_from_json
from rsstvlm.services.graphrag.retrieve import CustomRetriever
from rsstvlm.services.graphrag.t2c import Text2CypherRetriever
from rsstvlm.utils import (
    NEO4j_PASSWD,
    NEO4j_USR,
    deepseek,
    qwen3_embedding_8b,
)


class GraphRAGPipeline:
    """
    Args:
        k: similarity_top_k for rag retriever, defaults to 5
    """

    def __init__(
        self,
        k: int | None = 5,
    ):
        self.index = None
        self.query_engine = None
        self.vec_store = Neo4jVectorStore(
            username=NEO4j_USR,
            password=NEO4j_PASSWD,
            url="bolt://localhost:7687",
            embedding_dimension=4096,
        )
        self.graph_store = Neo4jPropertyGraphStore(
            username=NEO4j_USR,
            password=NEO4j_PASSWD,
            url="bolt://localhost:7687",
            refresh_schema=False,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vec_store,
            graph_store=self.graph_store,
        )
        self.vec_retriever = VectorContextRetriever(
            graph_store=self.graph_store,
            vector_store=self.vec_store,
            embed_model=qwen3_embedding_8b,
            include_text=True,
            similarity_top_k=k,
            path_depth=2,  # 图谱扩展深度 (1=直接关系, 2=2跳关系)
            limit=30,  # 最多返回多少个三元组
            similarity_score=0.7,
            verbose=True,
        )
        self.kg_retriever = Text2CypherRetriever(
            graph_store=self.graph_store,
            llm=deepseek,
            verbose=True,
        )

    def build_index(
        self,
        file_path: str | None = None,
        exist: bool = False,
        num_files_limit: int = 50,
    ) -> str:
        """Build the knowledge graph."""
        if exist:
            self.index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                vector_store=self.vec_store,
                llm=deepseek,
                embed_model=qwen3_embedding_8b,
            )
        else:
            documents = load_documents_from_json(
                num_files_limit=num_files_limit,
                input_dir=file_path,
            )

            kg_extractor = SchemaLLMPathExtractor(
                llm=deepseek,
                extract_prompt=EXTRACTION,
                num_workers=8,
                possible_entities=entities,
                possible_relations=relations,
                kg_validation_schema=validation_schema,
                strict=False,  # Allow flexible extraction beyond schema
            )

            self.index = PropertyGraphIndex.from_documents(
                documents=documents,
                kg_extractors=[kg_extractor],
                embed_model=qwen3_embedding_8b,
                show_progress=True,
                vector_store=self.vec_store,
                property_graph_store=self.graph_store,
            )

        return "Index built successfully."

    def hybrid_query(self, query: str):
        """
        Query the existing Neo4j knowledge graph and return a synthesised answer.

        This tool performs a hybrid search by combining vector-based retrieval
        and knowledge graph retrieval using an OR mode, then synthesizes a response
        using the DeepSeek LLM.

        This tool enables the Agent to retrieve and reason over structured knowledge
        stored in the Neo4j database. It translates a natural-language question into
        Cypher-based graph queries, extracts relevant entities, relationships, and
        their contextual information, and produces a concise, human-readable summary
        of the findings.

        Args:
            query (str): The natural language query string to search for.

        Returns:
            Response: The synthesized response from the query engine containing
                relevant information retrieved from both vector store and
                knowledge graph sources.
        """
        query_bundle = QueryBundle(
            query_str=query,
            embedding=qwen3_embedding_8b.get_query_embedding(query),
        )
        hybrid_retriever = CustomRetriever(
            vector_retriever=self.vec_retriever,
            kg_retriever=self.kg_retriever,
            mode="OR",
            verbose=True,
        )
        response_synthesizer = get_response_synthesizer(llm=deepseek)
        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
        )

        response = query_engine.query(query_bundle)
        return response


def main():
    pipeline = GraphRAGPipeline()
    pipeline.build_index(
        file_path="/satellite/d3/yaoyhu/rsstvlm/grobid/",
        num_files_limit=50,
        # exist=True,
    )
    # query = "What is the relationship between NO2 and pollution?"
    # print(pipeline.hybrid_query(query))


if __name__ == "__main__":
    main()
