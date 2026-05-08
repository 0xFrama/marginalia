from pydantic import BaseModel

from models import EvidenceBlock


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    candidate_k: int = 10
    min_score: float | None = None
    use_reranker: bool = False


class IndexRequest(BaseModel):
    pdf_path: str


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_evidence: list[EvidenceBlock]
    evidence_context: str
