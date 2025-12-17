import json
import os
from typing import Any

from llama_index.core import (
    Document,
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
        file_path: str,
        exist: bool = False,
        num_files_limit: int = 1,
    ) -> str:
        """Build the knowledge graph."""
        documents = self.load_documents_from_json(
            input_dir=file_path,
            num_files_limit=num_files_limit,
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

        if exist:
            self.index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                llm=deepseek,
                embed_model=qwen3_embedding_8b,
            )
        else:
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

    def load_documents_from_json(
        self,
        input_dir: str = "/satellite/d3/yaoyhu/rsstvlm/grobid/",
        num_files_limit: int | None = None,
        processed_file: str | None = None,
    ) -> list[Document]:
        # 设置已处理文件记录的路径
        if processed_file is None:
            processed_file = os.path.join(
                "/satellite/d3/yaoyhu/rsstvlm/", "processed_files.txt"
            )

        # 读取已处理的文件列表
        processed_set: set[str] = set()
        if os.path.exists(processed_file):
            try:
                with open(processed_file, encoding="utf-8") as f:
                    processed_set = {
                        line.strip() for line in f if line.strip()
                    }
                rag_logger.info(
                    f"Loaded {len(processed_set)} previously processed files."
                )
            except Exception as e:
                rag_logger.warning(f"Failed to load processed file list: {e}")

        json_files = [
            f for f in os.listdir(input_dir) if f.lower().endswith(".json")
        ]
        rag_logger.info(f"Total: {len(json_files)} jsons found.")

        # 过滤掉已处理的文件
        json_files = [f for f in json_files if f not in processed_set]
        rag_logger.info(
            f"Remaining: {len(json_files)} jsons to process (after filtering processed)."
        )

        if num_files_limit:
            json_files = json_files[:num_files_limit]
            rag_logger.info(f"Selected: {len(json_files)} jsons to process.")

        documents = []
        newly_processed: list[str] = []

        for json_file in json_files:
            json_path = os.path.join(input_dir, json_file)
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                rag_logger.warning(f"Failed to load {json_file}: {e}")
                continue

            # metadata
            biblio = data.get("biblio", {})
            metadata = {
                "source": json_file,
                "title": biblio.get("title", "Unknown Title"),
                "doi": biblio.get("doi", "Unknown DOI"),
            }

            # 提取所有文本，按 section 组织
            sections = self._extract_sections_from_json(data)

            # 合并所有 section 为一个完整文档
            text_parts = []
            for section_name, paragraphs in sections.items():
                text_parts.append(f"## {section_name}\n")
                for i, para in enumerate(paragraphs, start=1):
                    text_parts.append(f"[{i}] {para}\n")
                text_parts.append("")  # 空行分隔 section

            full_text = "\n".join(text_parts)

            if full_text.strip():
                doc = Document(
                    text=full_text,
                    metadata=metadata,
                )
                documents.append(doc)
                newly_processed.append(json_file)
                rag_logger.info(
                    f"Loaded {json_file}: {len(sections)} sections"
                )

        # 将新处理的文件追加到记录文件中
        if newly_processed:
            try:
                with open(processed_file, "a", encoding="utf-8") as f:
                    for filename in newly_processed:
                        f.write(filename + "\n")
                rag_logger.info(
                    f"Recorded {len(newly_processed)} newly processed files."
                )
            except Exception as e:
                rag_logger.warning(
                    f"Failed to update processed file list: {e}"
                )

        rag_logger.info(f"Total documents created: {len(documents)}")
        return documents

    def _extract_sections_from_json(self, data: dict) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}

        # 1. abstract
        biblio = data.get("biblio", {})
        abstract_data = biblio.get("abstract", [])

        # Handle various formats for abstract
        abstract_texts = []
        if isinstance(abstract_data, list):
            for item in abstract_data:
                if isinstance(item, str):
                    abstract_texts.append(item.strip())
                elif isinstance(item, dict):
                    # Extract text from dict (e.g., {"text": "...", "id": "..."})
                    text = item.get("text", "")
                    if text:
                        abstract_texts.append(text.strip())
        elif isinstance(abstract_data, dict):
            text = abstract_data.get("text", "")
            if text:
                abstract_texts.append(text.strip())
        elif isinstance(abstract_data, str):
            abstract_texts.append(abstract_data.strip())

        if abstract_texts:
            sections["abstract"] = abstract_texts

        # 2. body_text
        body_text = data.get("body_text", [])

        for paragraph in body_text:
            if not isinstance(paragraph, dict):
                continue

            text = paragraph.get("text", "").strip()
            if not text:
                continue

            section_name = paragraph.get("head_section", "unknown")
            if not section_name:
                section_name = "unknown"

            section_name = section_name.strip()

            if section_name not in sections:
                sections[section_name] = []

            sections[section_name].append(text)

        return sections


def main():
    pipeline = GraphRAGPipeline()
    print(
        pipeline.build_index(
            file_path="/satellite/d3/yaoyhu/rsstvlm/grobid/",
            num_files_limit=1000,
        )
    )
    # pipeline.query("Querying the database, what does excessive NO2 cause?")


if __name__ == "__main__":
    main()
