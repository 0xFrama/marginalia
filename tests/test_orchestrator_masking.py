from models.chunk import Chunk, ChunkType
from models.retrieval import RetrievalHit
from orchestrator.pipeline import answer_question


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
            doc_id="ada-2025",
            chunk_id="ada-2025-0001",
            text="Target HbA1c for most adults is below 7%.",
            source_file="ada_guidelines_2025.pdf",
            page_start=12,
            page_end=12,
            section_title="Glycemic Targets",
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

    result = answer_question(question, pid, engine, llm, retriever)

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
