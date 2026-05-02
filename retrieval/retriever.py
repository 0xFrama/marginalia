from index.qdrant_store import QdrantStore
from models.chunk import ChunkType
from models.retrieval import RetrievalHit


class Retriever:
    def __init__(self, store: QdrantStore) -> None:
        self.store = store

    def retrieve(
        self, question: str, chunk_types: list[ChunkType] | None = None, top_k: int = 5
    ) -> list[RetrievalHit]:

        chunk_types = chunk_types or [ChunkType.BODY]

        return self.store.search(
            query_text=question, chunk_types=chunk_types, top_k=top_k
        )
