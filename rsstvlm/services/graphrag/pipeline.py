import json
import re
from typing import Any

from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from rsstvlm.logger import rag_logger
from rsstvlm.services.graphrag.extrator import GraphRAGExtractor
from rsstvlm.services.graphrag.query import GraphRAGQueryEngine
from rsstvlm.services.graphrag.store import GraphRAGStore
from rsstvlm.utils import (
    NEO4j_PASSWD,
    NEO4j_USR,
    deepseek,
    qwen3_embedding_8b,
)

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: { "entities": [], "relationships": [] }.

-An Output Example-
{
  "entities": [
    {
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics."
    },
    {
      "entity_name": "Theory of Relativity",
      "entity_type": "Scientific Theory",
      "entity_description": "A scientific theory developed by Albert Einstein, describing the laws of physics in relation to observers in different frames of reference."
    },
    {
      "entity_name": "Nobel Prize in Physics",
      "entity_type": "Award",
      "entity_description": "A prestigious international award in the field of physics, awarded annually by the Royal Swedish Academy of Sciences."
    }
  ],
  "relationships": [
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    },
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Nobel Prize in Physics",
      "relation": "won",
      "relationship_description": "Albert Einstein won the Nobel Prize in Physics in 1921."
    }
  ]
}

-Real Data-
######################
text: {text}
######################
output:"""


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
        # TODO: multiple pdfs
        """Build the knowledge graph."""
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        nodes = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20,
        ).get_nodes_from_documents(documents)

        kg_extrator = GraphRAGExtractor(
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            llm=deepseek,
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

        self._create_query_engine()
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
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, response_str, re.DOTALL)
        entities = []
        relationships = []
        if not match:
            return entities, relationships
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            entities = [
                (
                    entity["entity_name"],
                    entity["entity_type"],
                    entity["entity_description"],
                )
                for entity in data.get("entities", [])
            ]
            relationships = [
                (
                    relation["source_entity"],
                    relation["target_entity"],
                    relation["relation"],
                    relation["relationship_description"],
                )
                for relation in data.get("relationships", [])
            ]
            return entities, relationships
        except json.JSONDecodeError:
            rag_logger.exception(
                "Error parsing JSON returned by GraphRAG extractor"
            )
            return entities, relationships


def main():
    pipeline = GraphRAGPipeline()
    pipeline.build_index(
        "./tests/Accurate_and_Full-Coverage_Retrieval_of_Total_Column_Water_Vapor_From_Chinese_UV-VIS_Satellites_Using_an_Interpretable_Machine_Learning_Approach.pdf",
    )


if __name__ == "__main__":
    main()
