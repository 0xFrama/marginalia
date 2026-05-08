from models import Chunk, ChunkType, RetrievalHit
from retrieval.reranker import CrossEncoderReranker


class FakeEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return self.scores


def make_hit(chunk_id: str, text: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        query_text="What is attention?",
        chunk=Chunk(
            doc_id="attention.pdf",
            chunk_id=chunk_id,
            text=text,
            source_file="attention.pdf",
            page_start=1,
            page_end=1,
            section_title="Attention",
            chunk_type=ChunkType.BODY,
            chunk_index=rank,
        ),
        score=0.5,
        rank=rank,
    )


def test_cross_encoder_reranker_scores_and_reorders_hits():
    encoder = FakeEncoder(scores=[0.1, 0.9])
    reranker = CrossEncoderReranker(encoder=encoder)
    first_hit = make_hit("attention.pdf:000001", "Weak evidence.", rank=1)
    second_hit = make_hit("attention.pdf:000002", "Strong evidence.", rank=2)

    hits = reranker.rerank(
        "What is attention?",
        [first_hit, second_hit],
        top_k=1,
    )

    assert encoder.calls == [
        [
            ("What is attention?", "Weak evidence."),
            ("What is attention?", "Strong evidence."),
        ]
    ]
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "attention.pdf:000002"
    assert hits[0].rerank_score == 0.9
    assert hits[0].rank == 1


def test_cross_encoder_reranker_handles_empty_hits():
    reranker = CrossEncoderReranker(encoder=FakeEncoder(scores=[]))

    assert reranker.rerank("What is attention?", [], top_k=3) == []
