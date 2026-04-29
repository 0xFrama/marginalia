from pydantic import BaseModel

from models.chunk import Chunk


class EmbeddingRecord(BaseModel):
    chunk: Chunk
    embedding: list[float]
