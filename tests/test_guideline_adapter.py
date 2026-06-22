from orchestrator.guideline_adapter import get_guideline_evidence


class FakeRetriever:
    """Records the kwargs it was called with; returns no hits."""

    def __init__(self):
        self.called_with = None

    def retrieve(self, question, **kwargs):
        self.called_with = {"question": question, **kwargs}
        return []


def test_adapter_passes_retrieval_knobs_through():
    retriever = FakeRetriever()
    sentinel_reranker = object()

    get_guideline_evidence(
        "HbA1c target?",
        retriever,
        reranker=sentinel_reranker,
        candidate_k=10,
        top_k=5,
    )

    call = retriever.called_with
    assert call["question"] == "HbA1c target?"
    assert call["reranker"] is sentinel_reranker
    # "retrieve wide, rerank narrow": more candidates than final results
    assert call["candidate_k"] == 10
    assert call["top_k"] == 5
    assert call["candidate_k"] > call["top_k"]


def test_adapter_returns_evidence_list():
    # no hits -> empty evidence, never crashes
    assert get_guideline_evidence("q", FakeRetriever()) == []
