from pydantic import BaseModel


class EvidenceBlock(BaseModel):
    citation_id: int
    text: str
    source_file: str
    page_start: int
    page_end: int
    section_title: str | None
    score: float
    rerank_score: float | None = None
