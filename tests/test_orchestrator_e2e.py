from models.chunk import Chunk, ChunkType
from models.retrieval import RetrievalHit
from orchestrator.pipeline import answer_question


class FakeLLM:
    """Returns canned replies in order: router, patient SQL, final answer."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FakeRetriever:
    """Returns one fixed guideline hit, ignoring the query."""

    def retrieve(self, question, **kwargs):
        chunk = Chunk(
            doc_id="ada-2025",
            chunk_id="ada-2025-0001",
            text="Target HbA1c for most adults with diabetes is below 7%.",
            source_file="ada_guidelines_2025.pdf",
            page_start=12,
            page_end=12,
            section_title="Glycemic Targets",
            chunk_type=ChunkType.BODY,
            chunk_index=1,
        )
        return [RetrievalHit(query_text=question, chunk=chunk, score=0.9, rank=1)]


def test_both_planes_fuse_into_one_cited_answer(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    llm = FakeLLM(
        [
            "both",  # router picks both planes
            "SELECT patient_id, code, value FROM v_observations",  # patient SQL
            "The patient's HbA1c is elevated [1]; guideline target is below 7%.",
        ]
    )

    result = answer_question(
        "Is this patient's HbA1c above the recommended target?",
        pid,
        engine,
        llm,
        retriever=FakeRetriever(),
    )

    # the router's decision is surfaced
    assert result.planes == ["patient", "guideline"]

    # both planes contributed evidence, fused into one list...
    kinds = {s.kind for s in result.sources}
    assert kinds == {"patient", "guideline"}

    # ...patient first, guideline last, renumbered 1..N
    assert result.sources[0].kind == "patient"
    assert result.sources[0].citation_id == 1
    assert result.sources[-1].kind == "guideline"
    assert result.sources[-1].citation_id == len(result.sources)

    # the answer is carried, and [1] resolves to the patient evidence
    assert "[1]" in result.answer
    assert result.cited_sources[0].citation_id == 1
    assert result.cited_sources[0].kind == "patient"
