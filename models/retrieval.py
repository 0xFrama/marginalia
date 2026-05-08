from pydantic import BaseModel

from models.chunk import Chunk


class RetrievalHit(BaseModel):
    query_text: str
    chunk: Chunk
    score: float
    rerank_score: float | None = None
    rank: int
