from index.qdrant_store import QdrantStore
from models.chunk import ChunkType
from models.retrieval import RetrievalHit
from retrieval.reranker import CrossEncoderReranker


class Retriever:
    def __init__(self, store: QdrantStore) -> None:
        self.store = store

    def retrieve(
        self,
        question: str,
        chunk_types: list[ChunkType] | None = None,
        candidate_k: int = 10,
        min_score: float | None = None,
        reranker: CrossEncoderReranker | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:

        chunk_types = chunk_types or [ChunkType.BODY]
        search_k = candidate_k if reranker else top_k

        hits = self.store.search(
            query_text=question, chunk_types=chunk_types, top_k=search_k
        )

        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= min_score]

        if reranker:
            return reranker.rerank(question, hits, top_k=top_k)

        return hits
