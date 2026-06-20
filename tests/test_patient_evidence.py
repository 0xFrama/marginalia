from patient.evidence import rows_to_evidence
from patient.pipeline import answer_patient_query, PatientQueryResult


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_successful_result_maps_to_patient_evidence(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    llm = FakeLLM(["SELECT patient_id, code, value FROM v_observations"])
    result = answer_patient_query("latest HbA1c?", pid, engine, llm)

    evidence = rows_to_evidence(result)

    assert len(evidence) > 0
    first = evidence[0]
    assert first.kind == "patient"
    assert first.citation_id == 1            # numbering starts at 1
    assert first.sql is not None             # provenance: the query is carried
    assert first.text                        # the row rendered as text


def test_refusal_yields_no_evidence():
    refusal = PatientQueryResult(ok=False, error="couldn't answer")
    assert rows_to_evidence(refusal) == []
