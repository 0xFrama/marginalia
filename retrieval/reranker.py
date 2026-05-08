from sentence_transformers.cross_encoder import CrossEncoder

from models.retrieval import RetrievalHit

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, encoder: CrossEncoder | None = None) -> None:
        self.encoder = encoder or CrossEncoder(MODEL_NAME)

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]:
        if not hits:
            return []

        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self.encoder.predict(pairs)

        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = float(score)

        ranked_hits = sorted(
            hits,
            key=lambda hit: hit.rerank_score if hit.rerank_score is not None else float("-inf"),
            reverse=True,
        )[:top_k]

        for rank, hit in enumerate(ranked_hits, start=1):
            hit.rank = rank

        return ranked_hits
