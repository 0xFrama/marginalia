import json
from dataclasses import dataclass

from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for retrieval-augmented generation systems.
Given a question, retrieved context, and a generated answer, evaluate the answer on two criteria.
Return a JSON object with exactly these fields:
- faithfulness: integer 1-5 (5 = every claim in the answer is supported by the context, 1 = major unsupported claims)
- faithfulness_reason: one sentence explaining the faithfulness score
- relevance: integer 1-5 (5 = answer directly and completely addresses the question, 1 = answer does not address the question)
- relevance_reason: one sentence explaining the relevance score
Return only valid JSON, no surrounding text."""

JUDGE_USER_TEMPLATE = """Question: {question}

Retrieved context:
{evidence}

Generated answer:
{answer}

Evaluate the answer."""


@dataclass
class EvalScore:
    question: str
    faithfulness: int
    faithfulness_reason: str
    relevance: int
    relevance_reason: str


def score_answer(
    question: str,
    evidence: str,
    answer: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> EvalScore:
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        evidence=evidence,
        answer=answer,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    try:
        data = json.loads(raw)
        return EvalScore(
            question=question,
            faithfulness=int(data["faithfulness"]),
            faithfulness_reason=data["faithfulness_reason"],
            relevance=int(data["relevance"]),
            relevance_reason=data["relevance_reason"],
        )
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Judge returned unexpected response: {e!r}\nRaw: {raw}") from e
