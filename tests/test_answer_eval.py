import json
from unittest.mock import MagicMock

import pytest

from eval.answer_eval import EvalScore, score_answer


def make_fake_client(response_json: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        response_json
    )
    return client


def test_score_answer_returns_eval_score():
    client = make_fake_client(
        {
            "faithfulness": 4,
            "faithfulness_reason": "Most claims are grounded in the context.",
            "relevance": 5,
            "relevance_reason": "Directly answers the question.",
        }
    )

    score = score_answer(
        question="What sensors are used in precision agriculture?",
        evidence="Soil moisture sensors and GPS are commonly used.",
        answer="Precision agriculture uses soil moisture sensors and GPS [1].",
        client=client,
    )

    assert isinstance(score, EvalScore)
    assert score.faithfulness == 4
    assert score.relevance == 5
    assert score.faithfulness_reason == "Most claims are grounded in the context."
    assert score.relevance_reason == "Directly answers the question."
    assert score.question == "What sensors are used in precision agriculture?"


def test_score_answer_sends_question_evidence_answer_to_judge():
    client = make_fake_client(
        {
            "faithfulness": 3,
            "faithfulness_reason": "Some claims are not in the context.",
            "relevance": 4,
            "relevance_reason": "Mostly addresses the question.",
        }
    )

    question = "What sensors are used in precision agriculture?"
    evidence = "GPS and soil sensors are used."
    answer = "Farmers use GPS and soil sensors [1]."

    score_answer(question=question, evidence=evidence, answer=answer, client=client)

    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    user_content = messages[1]["content"]
    assert question in user_content
    assert evidence in user_content
    assert answer in user_content


def test_score_answer_calls_judge_with_temperature_zero():
    client = make_fake_client(
        {
            "faithfulness": 5,
            "faithfulness_reason": "All claims are grounded.",
            "relevance": 5,
            "relevance_reason": "Fully answers the question.",
        }
    )

    score_answer(
        question="What is GPS?",
        evidence="GPS is a satellite navigation system.",
        answer="GPS is a satellite navigation system [1].",
        client=client,
    )

    call_kwargs = client.chat.completions.create.call_args
    assert call_kwargs.kwargs["temperature"] == 0.0


def test_score_answer_raises_on_malformed_response():
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "not valid json"

    with pytest.raises(ValueError, match="Judge returned unexpected response"):
        score_answer(
            question="What is GPS?",
            evidence="GPS is a satellite navigation system.",
            answer="GPS is a satellite navigation system [1].",
            client=client,
        )
