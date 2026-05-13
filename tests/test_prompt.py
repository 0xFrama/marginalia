from models.chat import ChatMessage
from qa import SYSTEM_PROMPT
from qa.prompt import build_qa_messages, build_qa_prompt


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


def test_prompt_contains_chat_history():
    question = "Can you explain it more simply?"
    history = [
        ChatMessage(role="user", content="What is attention?"),
        ChatMessage(role="assistant", content="Attention maps queries to outputs [1]."),
    ]

    result = build_qa_prompt(question, "[1] attention.pdf, page: 3", history)

    assert "Chat history:" in result
    assert "User: What is attention?" in result
    assert "Assistant: Attention maps queries to outputs [1]." in result
    assert question in result


def test_build_qa_messages_includes_history_and_final_question():
    question = "Can you explain it more simply?"
    history = [
        ChatMessage(role="user", content="What is attention?"),
        ChatMessage(role="assistant", content="Attention maps queries to outputs [1]."),
    ]

    messages = build_qa_messages(
        question,
        "[1] attention.pdf, page: 3\nAttention maps queries to outputs.",
        history,
    )

    assert len(messages) == 4
    assert "grounded assistant" in messages[0].content
    assert messages[1].content == "What is attention?"
    assert messages[2].content == "Attention maps queries to outputs [1]."
    assert question in messages[3].content
    assert "attention.pdf" in messages[3].content
