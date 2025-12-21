from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from rsstvlm.logger import rag_logger
from rsstvlm.prompts.extraction import (
    EXTRACTION,
    entities,
    relations,
    validation_schema,
)
from rsstvlm.services.graphrag.parse import load_documents_from_json
from rsstvlm.services.graphrag.query import GraphRAGQueryEngine
from rsstvlm.utils import (
    NEO4j_PASSWD,
    NEO4j_USR,
    deepseek,
    qwen3_embedding_8b,
)


class GraphRAGPipeline:
    def __init__(self):
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

        # self._create_query_engine()
        return "Index built successfully."

    def query(self, query_str: str) -> str:
        """Query the existing Neo4j knowledge graph and return a synthesised answer.

        This tool enables the Agent to retrieve and reason over structured knowledge
        stored in the Neo4j database. It translates a natural-language question into
        Cypher-based graph queries, extracts relevant entities, relationships, and
        their contextual information, and produces a concise, human-readable summary
        of the findings.

        Typical use cases:
            - Answering factual or analytical questions about stored entities.
            - Summarizing the relationships or patterns found in a subgraph.
            - Extracting aggregated insights from community or cluster summaries.

        Args:
            query_str (str):
                A natural-language query describing what to retrieve or analyze.
                The query should reference known entities, relationships, or topics
                that exist within the graph.

        Returns:
            str:
                A synthesized natural-language answer summarizing the most relevant
                information retrieved from the Neo4j graph. The result is optimized
                for readability and direct presentation to the user.
        """
        if not self._ensure_query_engine():
            return (
                "GraphRAG query engine is not ready. Ensure the Neo4j database "
                "is running and already populated before querying."
            )
        response = self.query_engine.custom_query(query_str)
        rag_logger.info("GraphRAG response: %s", response)
        return response

    def _create_query_engine(self) -> None:
        """Prepare the query engine after an index has been initialised."""
        if not self.index:
            return
        self.index.property_graph_store.build_communities()
        self.query_engine = GraphRAGQueryEngine(
            graph_store=self.index.property_graph_store,
            llm=deepseek,
            index=self.index,
            similarity_top_k=10,
        )

    def _ensure_query_engine(self) -> bool:
        """Initialise the query engine from the existing graph store if needed."""
        if self.query_engine:
            return True
        try:
            if not self.index:
                self.index = PropertyGraphIndex.from_existing(
                    property_graph_store=self.graph_store,
                    vector_store=self.vec_store,
                    llm=deepseek,
                    embed_model=qwen3_embedding_8b,
                )
            self._create_query_engine()
        except Exception:
            rag_logger.exception("Failed to initialise GraphRAG query engine")
            self.index = None
            self.query_engine = None
            return False

        return self.query_engine is not None


def main():
    pipeline = GraphRAGPipeline()
    print(
        pipeline.build_index(
            file_path="/satellite/d3/yaoyhu/rsstvlm/grobid/",
            num_files_limit=50,
        )
    )
    # pipeline.query("Querying the database, what does excessive NO2 cause?")


if __name__ == "__main__":
    main()
