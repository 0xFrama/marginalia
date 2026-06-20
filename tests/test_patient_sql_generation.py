from patient.sql_generation import generate_sql


class FakeLLM:
    """A stand-in for OpenAIService: same generate(system, user) shape, but it
    returns a canned SQL string and records what it was asked. Lets us test the
    wiring without a real (slow, costly, non-deterministic) API call."""

    def __init__(self, canned_sql):
        self.canned_sql = canned_sql
        self.last_system = None
        self.last_user = None

    def generate(self, system_prompt, user_prompt):
        self.last_system = system_prompt
        self.last_user = user_prompt
        return self.canned_sql


def test_generate_sql_returns_llm_output():
    fake = FakeLLM("SELECT code, value FROM v_observations")
    out = generate_sql("What is the latest HbA1c?", "DICTIONARY", 3, fake)
    assert out == "SELECT code, value FROM v_observations"


def test_generate_sql_passes_dictionary_and_patient_into_prompt():
    fake = FakeLLM("SELECT 1")
    generate_sql("any question", "MY_DICTIONARY_TEXT", 7, fake)
    # The schema dictionary must reach the model (this drives schema linking)...
    assert "MY_DICTIONARY_TEXT" in fake.last_system
    # ...and the patient id must be in the prompt context.
    assert "7" in fake.last_user
