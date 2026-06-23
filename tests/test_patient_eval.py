from eval.patient_eval import is_correct, run_eval
from eval.patient_golden import EvalItem
from patient.pipeline import PatientQueryResult


class FakeRow:
    """Mimics a SQLAlchemy row: rows_to_evidence reads ._mapping."""

    def __init__(self, mapping):
        self._mapping = mapping


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return response


# --- is_correct (the scorer) -------------------------------------------------

def test_refusal_is_never_correct():
    result = PatientQueryResult(ok=False, error="could not answer")
    assert is_correct(result, "6.8") is False


def test_expected_value_present_is_correct():
    result = PatientQueryResult(ok=True, rows=[FakeRow({"code": "HbA1c", "value": 6.8})])
    assert is_correct(result, "6.8") is True


def test_expected_value_absent_is_incorrect():
    result = PatientQueryResult(ok=True, rows=[FakeRow({"code": "HbA1c", "value": 8.2})])
    assert is_correct(result, "6.8") is False


# --- run_eval (the runner) ---------------------------------------------------

def test_run_eval_records_pass_and_fail(db_session):
    session, anchors = db_session
    engine = session.bind

    gold_set = [
        EvalItem("latest HbA1c?", "diabetic", "6.8"),      # Arthur has it -> pass
        EvalItem("latest HbA1c?", "hypertensive", "6.8"),  # Elena has none -> fail
    ]
    # same valid SQL for both items; scoping limits it to each patient
    sql = "SELECT patient_id, code, value FROM v_observations WHERE code = 'HbA1c'"
    llm = FakeLLM([sql, sql])

    outcomes = run_eval(engine, anchors, llm, gold_set)

    assert outcomes[0][1] is True
    assert outcomes[1][1] is False
