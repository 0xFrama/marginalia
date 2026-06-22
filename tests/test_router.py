from orchestrator.router import route


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, system_prompt, user_prompt):
        return self.reply


def test_patient_only():
    assert route("latest HbA1c for this patient?", FakeLLM("patient")) == ["patient"]


def test_guideline_only():
    assert route("recommended HbA1c target?", FakeLLM("guideline")) == ["guideline"]


def test_both():
    assert route("is this patient above target?", FakeLLM("both")) == [
        "patient",
        "guideline",
    ]


def test_messy_reply_is_cleaned():
    # LLM adds capitals / whitespace / punctuation -> still parsed
    assert route("q", FakeLLM("  Patient\n")) == ["patient"]


def test_unknown_reply_falls_back_to_both():
    # the parser GUARANTEES the safe default, even if the prompt is ignored
    assert route("q", FakeLLM("I think maybe the patient one?")) == [
        "patient",
        "guideline",
    ]
