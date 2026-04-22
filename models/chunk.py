from pydantic import BaseModel
from enum import Enum


class ChunkType(str, Enum):
    BODY = "body"
    CAPTION = "caption"
    TABLE = "table"
    LIST = "list"
    FOOTNOTE = "footnote"
    FORMULA = "formula"


class Chunk(BaseModel):
    text: str
    source_file: str
    page_start: int
    page_end: int
    section_title: str | None
    chunk_type: ChunkType
    chunk_index: int
