from qa import SYSTEM_PROMPT
from qa.prompt import build_qa_prompt


def test_system_prompt():
    assert "only using the provided evidence" in SYSTEM_PROMPT
    assert "citation" in SYSTEM_PROMPT or "[1]" in SYSTEM_PROMPT
    assert "insufficient" in SYSTEM_PROMPT


def test_prompt_contains_the_evidence():
    question = "Is too much sun harmful to vegetables?"
    result = build_qa_prompt(
        question, "[1] guide.pdf, page: 2\nTomatoes need irrigation."
    )
    assert "[1] guide.pdf, page: 2" in result
    assert "Tomatoes need irrigation" in result
    assert question in result


def test_empty_evidence():
    question = "Is too much sun harmful to vegetables?"
    result = build_qa_prompt(question, "")
    assert "No evidence was retrieved" in result
