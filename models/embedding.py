from pydantic import BaseModel

from models.chunk import Chunk


class EmbeddingRecord(BaseModel):
    chunk: Chunk
    embedding: list[float]
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None
