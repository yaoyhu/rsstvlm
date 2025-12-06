import json
from typing import Any

from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from rsstvlm.logger import rag_logger
from rsstvlm.prompts.extraction import EXTRACTION
from rsstvlm.services.graphrag.extrator import GraphRAGExtractor
from rsstvlm.services.graphrag.query import GraphRAGQueryEngine
from rsstvlm.services.graphrag.store import GraphRAGStore
from rsstvlm.utils import (
    NEO4j_PASSWD,
    NEO4j_USR,
    deepseek,
    qwen3_embedding_8b,
)


class GraphRAGPipeline:
    """
    GraphRAG pipeline for building and querying a knowledge graph.
    """

    NAME = "GraphRAG"
    DESCRIPTION = "Query an existing Neo4j-backed knowledge graph."

    def __init__(self):
        self.index = None
        self.query_engine = None
        self.graph_store = GraphRAGStore(
            username=NEO4j_USR,
            password=NEO4j_PASSWD,
            url="bolt://localhost:7687",
        )

    def build_index(self, file_path: str, exist: bool = False) -> str:
        """Build the knowledge graph."""
        documents = SimpleDirectoryReader(
            input_dir=file_path, num_files_limit=50
        ).load_data(show_progress=True)
        nodes = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20,
        ).get_nodes_from_documents(documents)

        kg_extrator = GraphRAGExtractor(
            llm=deepseek,
            extract_prompt=EXTRACTION,
            max_paths_per_chunk=2,
            parse_fn=self._parse_fn,
        )

        if exist:
            self.index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                llm=deepseek,
                embed_model=qwen3_embedding_8b,
            )
        else:
            self.index = PropertyGraphIndex(
                nodes=nodes,
                kg_extractors=[kg_extrator],
                property_graph_store=self.graph_store,
                show_progress=True,
                llm=deepseek,
                embed_model=qwen3_embedding_8b,
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

    def _parse_fn(self, response_str: str) -> Any:
        entities = []
        relationships = []

        try:
            data = json.loads(response_str.strip())
        except json.JSONDecodeError:
            start = response_str.find("{")
            end = response_str.rfind("}")
            if start == -1 or end == -1 or start >= end:
                rag_logger.warning("No valid JSON found in response")
                return entities, relationships
            json_str = response_str[start : end + 1]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                rag_logger.exception(f"Error parsing JSON: {e}")
                rag_logger.debug(f"Attempted to parse: {json_str[:500]}...")
                return entities, relationships

        try:
            entities = [
                (
                    entity["entity_name"],
                    entity["entity_type"],
                    entity.get("entity_description", ""),
                )
                for entity in data.get("entities", [])
            ]
            relationships = [
                (
                    rel["source_entity"],
                    rel["target_entity"],
                    rel.get("relation")
                    or rel.get("relationship", "RELATED_TO"),
                    rel.get("relationship_description", ""),
                )
                for rel in data.get("relationships", [])
            ]
            rag_logger.info(
                f"Parsed {len(entities)} entities, {len(relationships)} relationships"
            )
        except (KeyError, TypeError) as e:
            rag_logger.exception(f"Error extracting data from JSON: {e}")

        return entities, relationships


def main():
    pipeline = GraphRAGPipeline()
    pipeline.build_index(file_path="/satellite/d3/yaoyhu/rsstvlm/raw_papers/")
    pipeline.query("Querying the database, what does excessive NO2 cause?")


if __name__ == "__main__":
    main()
