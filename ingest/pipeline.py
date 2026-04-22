from pathlib import Path

from docling.document_converter import DocumentConverter

from models.chunk import Chunk, ChunkType

LABEL_TO_TYPE = {
    "text": ChunkType.BODY,
    "footnote": ChunkType.FOOTNOTE,
    "caption": ChunkType.CAPTION,
    "table": ChunkType.TABLE,
    "formula": ChunkType.FORMULA,
    "list_item": ChunkType.LIST,
}


def ingest_pdf(path: str) -> list[Chunk]:
    converter = DocumentConverter()
    doc = converter.convert(path).document
    chunks = []
    current_section = None

    for item, level in doc.iterate_items():
        if item.label == "section_header":
            current_section = item.text
            continue
        if item.label not in LABEL_TO_TYPE:
            continue
        if item.label == LABEL_TO_TYPE["table"]:
            text = item.export_to_markdown(doc)
        else:
            text = item.text
        if not text or not text.strip():
            continue
        if not item.prov:
            continue

        page_start, page_end = min(p.page_no for p in item.prov), max(
            p.page_no for p in item.prov
        )

        chunks.append(
            Chunk(
                text=text,
                source_file=Path(path).name,
                page_start=page_start,
                page_end=page_end,
                section_title=current_section,
                chunk_type=LABEL_TO_TYPE[item.label],
                chunk_index=len(chunks),
            )
        )

    return chunks
