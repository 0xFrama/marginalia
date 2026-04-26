from pydantic import BaseModel
from enum import Enum
from hashlib import sha256


class ChunkType(str, Enum):
    TITLE = "title"
    BODY = "body"
    CAPTION = "caption"
    TABLE = "table"
    LIST = "list"
    FOOTNOTE = "footnote"
    FORMULA = "formula"


class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    section_title: str | None = None
    chunk_type: ChunkType
    chunk_index: int

    def to_record(self) -> dict:
        return {
            "schema_version": "chunk.v1",
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type.value,
            "text": self.text,
            "content_hash": sha256(self.text.strip().encode("utf-8")).hexdigest(),
        }
