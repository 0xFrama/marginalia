from models.chunk import Chunk, ChunkType
from models.retrieval import RetrievalHit
from orchestrator.graph import GraphOrchestrator


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        r = self.responses[self.calls]
        self.calls += 1
        return r


class FakeRetriever:
    def retrieve(self, question, **kwargs):
        chunk = Chunk(
            doc_id="g",
            chunk_id="g1",
            text="Target HbA1c is below 7%.",
            source_file="guidelines.pdf",
            page_start=1,
            page_end=1,
            section_title="Targets",
            chunk_type=ChunkType.BODY,
            chunk_index=1,
        )
        return [RetrievalHit(query_text=question, chunk=chunk, score=0.9, rank=1)]


def test_start_pauses_then_resume_completes(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id
    llm = FakeLLM(
        ["patient", "SELECT patient_id, value FROM v_observations", "HbA1c 6.8 [1]"]
    )
    orch = GraphOrchestrator(engine, llm, FakeRetriever())

    step = orch.start("latest HbA1c?", pid)
    assert step["status"] == "awaiting_approval"
    assert "SELECT" in step["sql"]
    assert step["thread_id"]

    final = orch.resume(step["thread_id"], "approve")
    assert final["status"] == "done"
    assert any(s.kind == "patient" for s in final["result"].sources)


def test_hitl_approve_yields_patient_evidence(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id
    llm = FakeLLM(
        ["patient", "SELECT patient_id, value FROM v_observations", "HbA1c 6.8 [1]"]
    )
    orch = GraphOrchestrator(engine, llm, FakeRetriever())
    result = orch.answer("latest HbA1c?", pid, approver=lambda sql: "approve")
    assert any(s.kind == "patient" for s in result.sources)


def test_hitl_reject_runs_no_sql(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id
    llm = FakeLLM(
        ["patient", "SELECT patient_id, value FROM v_observations", "n/a"]
    )
    orch = GraphOrchestrator(engine, llm, FakeRetriever())
    result = orch.answer("latest HbA1c?", pid, approver=lambda sql: "reject")
    assert all(s.kind != "patient" for s in result.sources)


class SpyLLM:
    """Records every prompt it sees; returns canned replies in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen_user_prompts = []

    def generate(self, system_prompt, user_prompt):
        self.seen_user_prompts.append(user_prompt)
        reply = self.responses[self.calls]
        self.calls += 1
        return reply


class SpyRetriever:
    """Records the query it was asked to search for."""

    def __init__(self):
        self.seen_query = None

    def retrieve(self, question, **kwargs):
        self.seen_query = question
        chunk = Chunk(
            doc_id="g",
            chunk_id="g1",
            text="Target HbA1c is below 7%.",
            source_file="guidelines.pdf",
            page_start=1,
            page_end=1,
            section_title="Targets",
            chunk_type=ChunkType.BODY,
            chunk_index=1,
        )
        return [RetrievalHit(query_text=question, chunk=chunk, score=0.9, rank=1)]


def test_real_identifiers_reach_neither_llm_nor_retriever(db_session):
    session, anchors = db_session
    engine = session.bind
    patient = anchors["diabetic"]  # Arthur Dent
    pid = patient.patient_id

    # the question carries the patient's real name
    question = f"Is {patient.name}'s HbA1c above the recommended target?"

    llm = SpyLLM(
        [
            "both",  # router
            "SELECT patient_id, code, value FROM v_observations",  # patient SQL
            "The HbA1c is elevated [1]; target is below 7% [2].",  # final answer
        ]
    )
    retriever = SpyRetriever()

    orch = GraphOrchestrator(engine, llm, retriever)
    result = orch.answer(question, pid)

    # 1) the real name never reached the LLM, on ANY of its calls
    for prompt in llm.seen_user_prompts:
        assert patient.name not in prompt
        assert patient.mrn not in prompt
    # and it was actually masked, not just absent
    assert any("[PATIENT_NAME]" in p for p in llm.seen_user_prompts)

    # 2) the real name never reached the retriever (the embedding egress)
    assert patient.name not in retriever.seen_query
    assert "[PATIENT_NAME]" in retriever.seen_query

    # 3) the system still produced a usable answer
    assert result.planes == ["patient", "guideline"]
    assert "[1]" in result.answer


def test_both_planes_through_graph(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    llm = FakeLLM(
        [
            "both",
            "SELECT patient_id, code, value FROM v_observations",
            "HbA1c is 6.8 [1]; target below 7% [2].",
        ]
    )
    orch = GraphOrchestrator(engine, llm, FakeRetriever())
    result = orch.answer("Is this patient's HbA1c above target?", pid)

    assert result.planes == ["patient", "guideline"]
    kinds = {s.kind for s in result.sources}
    assert kinds == {"patient", "guideline"}
    assert "[1]" in result.answer
