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
