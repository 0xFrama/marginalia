import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from eval.answer_eval import EvalScore, score_answer
from index.qdrant_store import QdrantStore
from qa.answerer import Answerer
from qa.llm import OpenAIService
from retrieval.retriever import Retriever

load_dotenv()

CASES_PATH = Path(__file__).parent / "answer_cases.jsonl"
PASS_THRESHOLD = 4


def load_cases(path: Path) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                case = json.loads(line)
                assert "question" in case, f"Missing 'question' field in case: {case}"
                cases.append(case)
    return cases


def print_score(score: EvalScore) -> None:
    status_f = "PASS" if score.faithfulness >= PASS_THRESHOLD else "FAIL"
    status_r = "PASS" if score.relevance >= PASS_THRESHOLD else "FAIL"
    print(f"Q: {score.question}")
    print(f"  faithfulness: {score.faithfulness}/5 [{status_f}] — {score.faithfulness_reason}")
    print(f"  relevance:    {score.relevance}/5 [{status_r}] — {score.relevance_reason}")


def main() -> None:
    cases = load_cases(CASES_PATH)
    if not cases:
        raise RuntimeError("No evaluation cases found in " + str(CASES_PATH))

    api_key = os.environ["OPENAI_API_KEY"]

    store = QdrantStore()
    retriever = Retriever(store=store)
    llm = OpenAIService()
    answerer = Answerer(retriever=retriever, llm_client=llm)
    judge = OpenAI(api_key=api_key)

    scores: list[EvalScore] = []
    failed = 0
    n = len(cases)
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{n}] {case['question']}")
        try:
            result = answerer.answer(case["question"])
            score = score_answer(
                question=case["question"],
                evidence=result.evidence,
                answer=result.answer,
                client=judge,
            )
            scores.append(score)
            print_score(score)
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
        print()

    if not scores:
        print("No scores collected — all cases failed.")
        return

    avg_f = sum(s.faithfulness for s in scores) / len(scores)
    avg_r = sum(s.relevance for s in scores) / len(scores)
    passed_f = sum(1 for s in scores if s.faithfulness >= PASS_THRESHOLD)
    passed_r = sum(1 for s in scores if s.relevance >= PASS_THRESHOLD)
    print(f"faithfulness: {avg_f:.2f}/5 ({passed_f}/{len(scores)} passed)")
    print(f"relevance:    {avg_r:.2f}/5 ({passed_r}/{len(scores)} passed)")
    if failed:
        print(f"failed: {failed}/{n}")


if __name__ == "__main__":
    main()
