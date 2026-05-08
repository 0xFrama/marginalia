from retrieval import Retriever
from models import Chunk, ChunkType, RetrievalHit


class FakeStore:
    def __init__(self) -> None:
        self.calls = []
        self.hits = [
            RetrievalHit(
                query_text="How should tomatoes be irrigated?",
                chunk=Chunk(
                    doc_id="agri-guide.pdf",
                    chunk_id="agri-guide.pdf:000001",
                    text="Tomatoes need regular irrigation during hot periods.",
                    source_file="agri-guide.pdf",
                    page_start=1,
                    page_end=1,
                    section_title="Irrigation",
                    chunk_type=ChunkType.BODY,
                    chunk_index=1,
                ),
                score=0.9,
                rank=1,
            )
        ]
        self.low_score_hit = RetrievalHit(
            query_text="How should tomatoes be irrigated?",
            chunk=Chunk(
                doc_id="agri-guide.pdf",
                chunk_id="agri-guide.pdf:000002",
                text="This weakly related chunk should be filtered out.",
                source_file="agri-guide.pdf",
                page_start=2,
                page_end=2,
                section_title="Background",
                chunk_type=ChunkType.BODY,
                chunk_index=2,
            ),
            score=0.4,
            rank=2,
        )

    def search(
        self,
        query_text: str,
        chunk_types: list[ChunkType] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        self.calls.append(
            {
                "query_text": query_text,
                "chunk_types": chunk_types,
                "top_k": top_k,
            }
        )
        return self.hits


class FakeReranker:
    def __init__(self) -> None:
        self.calls = []

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]:
        self.calls.append({"query": query, "hits": hits, "top_k": top_k})
        reranked = list(reversed(hits))[:top_k]
        for rank, hit in enumerate(reranked, start=1):
            hit.rank = rank
            hit.rerank_score = 1.0 / rank
        return reranked


def test_retriever_defaults_to_body_chunks():
    store = FakeStore()
    retriever = Retriever(store=store)

    hits = retriever.retrieve("How should tomatoes be irrigated?")

    assert hits == store.hits
    assert store.calls == [
        {
            "query_text": "How should tomatoes be irrigated?",
            "chunk_types": [ChunkType.BODY],
            "top_k": 5,
        }
    ]


def test_retriever_passes_custom_filters_and_top_k():
    store = FakeStore()
    retriever = Retriever(store=store)

    retriever.retrieve(
        "Show irrigation tables",
        chunk_types=[ChunkType.BODY, ChunkType.TABLE],
        top_k=3,
    )

    assert store.calls == [
        {
            "query_text": "Show irrigation tables",
            "chunk_types": [ChunkType.BODY, ChunkType.TABLE],
            "top_k": 3,
        }
    ]


def test_retriever_filters_hits_below_min_score():
    store = FakeStore()
    store.hits = [store.hits[0], store.low_score_hit]
    retriever = Retriever(store=store)

    hits = retriever.retrieve("How should tomatoes be irrigated?", min_score=0.5)

    assert len(hits) == 1
    assert hits[0].score == 0.9


def test_retriever_uses_candidate_k_when_reranking():
    store = FakeStore()
    store.hits = [store.hits[0], store.low_score_hit]
    reranker = FakeReranker()
    retriever = Retriever(store=store)

    hits = retriever.retrieve(
        "How should tomatoes be irrigated?",
        candidate_k=10,
        top_k=1,
        reranker=reranker,
    )

    assert store.calls == [
        {
            "query_text": "How should tomatoes be irrigated?",
            "chunk_types": [ChunkType.BODY],
            "top_k": 10,
        }
    ]
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["top_k"] == 1
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "agri-guide.pdf:000002"
