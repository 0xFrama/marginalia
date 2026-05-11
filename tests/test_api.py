from fastapi.testclient import TestClient

from api.app import app, get_answerer, get_store
from models import AnswerResult, EvidenceBlock, IndexingResult


class FakeAnswerer:
    def __init__(self) -> None:
        self.calls = []

    def answer(
        self,
        question: str,
        candidate_k: int = 10,
        top_k: int = 3,
        min_score: float | None = None,
        reranker=None,
    ) -> AnswerResult:
        self.calls.append(
            {
                "question": question,
                "candidate_k": candidate_k,
                "top_k": top_k,
                "min_score": min_score,
                "reranker": reranker,
            }
        )
        return AnswerResult(
            question=question,
            answer="Attention maps queries and key-value pairs to outputs [1].",
            sources=[
                EvidenceBlock(
                    citation_id=1,
                    text="An attention function maps a query and key-value pairs to an output.",
                    source_file="attention.pdf",
                    page_start=3,
                    page_end=3,
                    section_title="3.2 Attention",
                    score=0.682,
                )
            ],
            cited_sources=[
                EvidenceBlock(
                    citation_id=1,
                    text="An attention function maps a query and key-value pairs to an output.",
                    source_file="attention.pdf",
                    page_start=3,
                    page_end=3,
                    section_title="3.2 Attention",
                    score=0.682,
                )
            ],
            evidence="[1] attention.pdf, page: 3\nAn attention function maps a query...",
        )


def test_ask_endpoint_returns_answer_and_retrieved_evidence():
    fake_answerer = FakeAnswerer()
    app.dependency_overrides[get_answerer] = lambda: fake_answerer
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "What is attention?",
            "top_k": 5,
            "candidate_k": 9,
            "min_score": 0.55,
            "use_reranker": False,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "What is attention?"
    assert payload["answer"] == "Attention maps queries and key-value pairs to outputs [1]."
    assert payload["evidence_context"].startswith("[1] attention.pdf")
    assert len(payload["retrieved_evidence"]) == 1
    assert payload["retrieved_evidence"][0]["citation_id"] == 1
    assert payload["retrieved_evidence"][0]["source_file"] == "attention.pdf"
    assert len(payload["cited_sources"]) == 1
    assert payload["cited_sources"][0]["citation_id"] == 1
    assert fake_answerer.calls == [
        {
            "question": "What is attention?",
            "candidate_k": 9,
            "top_k": 5,
            "min_score": 0.55,
            "reranker": None,
        }
    ]


def test_ask_endpoint_passes_reranker_when_requested(monkeypatch):
    fake_answerer = FakeAnswerer()
    fake_reranker = object()
    app.dependency_overrides[get_answerer] = lambda: fake_answerer
    monkeypatch.setattr("api.app.get_reranker", lambda: fake_reranker)
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "What is attention?",
            "top_k": 3,
            "candidate_k": 10,
            "use_reranker": True,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_answerer.calls[0]["reranker"] is fake_reranker


def test_index_endpoint_returns_indexing_result(monkeypatch):
    calls = []

    def fake_index_pdf(pdf_path: str, store) -> IndexingResult:
        calls.append({"pdf_path": pdf_path, "store": store})
        return IndexingResult(
            source_file="attention.pdf",
            chunk_count=139,
            indexed_count=139,
            collection_name="chunks",
        )

    fake_store = object()
    app.dependency_overrides[get_store] = lambda: fake_store
    monkeypatch.setattr("api.app.index_pdf", fake_index_pdf)
    client = TestClient(app)

    response = client.post("/index", json={"pdf_path": "samples/attention.pdf"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "source_file": "attention.pdf",
        "chunk_count": 139,
        "indexed_count": 139,
        "collection_name": "chunks",
    }
    assert calls == [{"pdf_path": "samples/attention.pdf", "store": fake_store}]
