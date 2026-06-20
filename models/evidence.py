from typing import Literal
from pydantic import BaseModel


class EvidenceBlock(BaseModel):
    citation_id: int
    text: str
    kind: Literal["guideline", "patient"] = "guideline"
    source_label: str | None = None

    source_file: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    score: float | None = None
    rerank_score: float | None = None

    views: list[str] | None = None
    as_of: str | None = None
    sql: str | None = None


Evidence = EvidenceBlock
