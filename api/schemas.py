from pydantic import BaseModel, Field

from models import EvidenceBlock
from models.chat import ChatMessage


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    candidate_k: int = 10
    min_score: float | None = None
    use_reranker: bool = False
    chat_history: list[ChatMessage] = Field(default_factory=list)


class IndexRequest(BaseModel):
    pdf_path: str


class AskResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    retrieved_evidence: list[EvidenceBlock]
    evidence_context: str
    cited_sources: list[EvidenceBlock]
