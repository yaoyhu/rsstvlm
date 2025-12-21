import json
import os

from llama_index.core import Document
from rsstvlm.logger import rag_logger


def load_documents_from_json(
    num_files_limit: int | None = 50,
    input_dir: str = "/satellite/d3/yaoyhu/rsstvlm/grobid/",
    processed_file: str = "/satellite/d3/yaoyhu/rsstvlm/processed_files.txt",
) -> list[Document]:
    # 读取已处理的文件列表
    processed_set: set[str] = set()
    if os.path.exists(processed_file):
        try:
            with open(processed_file, encoding="utf-8") as f:
                processed_set = {line.strip() for line in f if line.strip()}
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

        # 提取所有文本, 按 section 组织
        sections = _extract_sections_from_json(data)

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
            rag_logger.info(f"Loaded {json_file}: {len(sections)} sections")

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
            rag_logger.warning(f"Failed to update processed file list: {e}")

    rag_logger.info(f"Total documents created: {len(documents)}")
    return documents


def _extract_sections_from_json(data: dict) -> dict[str, list[str]]:
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
