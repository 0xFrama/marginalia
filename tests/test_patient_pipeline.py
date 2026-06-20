from patient.pipeline import answer_patient_query

GOOD_SQL = "SELECT patient_id, code, value FROM v_observations"
BAD_SQL = "DELETE FROM v_observations"  # rejected by the validator


class FakeLLM:
    """Returns canned SQL strings, one per call, so we can script the AI's
    behaviour (good first try, bad-then-good, always-bad) without a real API."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_happy_path_returns_rows_and_sql(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    llm = FakeLLM([GOOD_SQL])
    result = answer_patient_query("latest HbA1c?", pid, engine, llm)

    assert result.ok is True
    assert len(result.rows) > 0
    assert result.sql is not None
    assert llm.calls == 1  # no retry needed


def test_invalid_then_valid_succeeds_after_one_retry(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    # First answer is rejected by the validator; second is good.
    llm = FakeLLM([BAD_SQL, GOOD_SQL])
    result = answer_patient_query("latest HbA1c?", pid, engine, llm)

    assert result.ok is True
    assert len(result.rows) > 0
    assert llm.calls == 2  # used exactly one retry


def test_always_invalid_returns_refusal_not_exception(db_session):
    session, anchors = db_session
    engine = session.bind
    pid = anchors["diabetic"].patient_id

    # Every attempt is unsafe → the pipeline must refuse, not raise.
    llm = FakeLLM([BAD_SQL, BAD_SQL])
    result = answer_patient_query("latest HbA1c?", pid, engine, llm)

    assert result.ok is False
    assert result.rows == []        # never fabricates rows
    assert result.error is not None  # records why it refused
