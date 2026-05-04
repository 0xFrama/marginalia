from models.evidence import EvidenceBlock
from pydantic import BaseModel


class AnswerResult(BaseModel):
    question: str
    answer: str
    sources: list[EvidenceBlock]
    evidence: str
