from orchestrator.patient_adapter import get_patient_evidence


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_adapter_returns_patient_evidence(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    llm = FakeLLM(["SELECT patient_id, code, value FROM v_observations"])
    evidence = get_patient_evidence("HbA1c readings?", pid, engine, llm)

    assert len(evidence) > 0
    assert evidence[0].kind == "patient"


def test_adapter_returns_empty_list_on_refusal(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    # invalid SQL on every attempt -> pipeline refuses -> adapter yields []
    llm = FakeLLM(["DROP TABLE patients", "DROP TABLE patients"])
    evidence = get_patient_evidence("delete everything", pid, engine, llm)

    assert evidence == []
