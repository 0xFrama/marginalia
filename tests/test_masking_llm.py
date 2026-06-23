from masking.llm import MaskingLLM


class SpyLLM:
    """Records what it received, and returns a canned answer."""

    def __init__(self, answer):
        self.answer = answer
        self.seen_system = None
        self.seen_user = None

    def generate(self, system_prompt, user_prompt):
        self.seen_system = system_prompt
        self.seen_user = user_prompt
        return self.answer


IDS = {"PATIENT_NAME": "Arthur Dent", "MRN": "MRN-DET-001"}


def test_real_llm_never_sees_the_real_name():
    # the answer echoes the placeholder, so we can check unmasking too
    spy = SpyLLM("[PATIENT_NAME]'s HbA1c is 6.9.")
    wrapped = MaskingLLM(spy, IDS)

    result = wrapped.generate("system", "Is Arthur Dent's HbA1c high?")

    # 1) what left the system was masked: the real name never reached the LLM
    assert "Arthur Dent" not in spy.seen_user
    assert "[PATIENT_NAME]" in spy.seen_user

    # 2) what came back was unmasked: the caller sees the real name
    assert result == "Arthur Dent's HbA1c is 6.9."


def test_looks_like_a_normal_llm():
    # same generate(system, user) -> str shape as the real LLM
    wrapped = MaskingLLM(SpyLLM("ok"), IDS)
    assert wrapped.generate("s", "u") == "ok"


def test_empty_identifiers_passes_through_unchanged():
    spy = SpyLLM("nothing to restore")
    wrapped = MaskingLLM(spy, {})

    result = wrapped.generate("system", "What is the HbA1c target?")

    assert spy.seen_user == "What is the HbA1c target?"  # untouched
    assert result == "nothing to restore"
