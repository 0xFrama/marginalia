from patient.evidence import rows_to_evidence
from patient.pipeline import answer_patient_query


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_patient_plane_e2e(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    llm = FakeLLM(["SELECT patient_id, code FROM v_observations"])

    result = answer_patient_query("HbA1c readings?", pid, engine, llm)
    evidence = rows_to_evidence(result)

    assert result.ok is True
    assert len(evidence) > 0
    assert evidence[0].kind == "patient"
    assert evidence[0].sql is not None
