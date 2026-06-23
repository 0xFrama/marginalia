"""Run the patient-plane execution-accuracy eval against a real LLM.

Usage (needs OPENAI_API_KEY in the environment):
    uv run python -m eval.run_patient_eval
"""

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from eval.patient_eval import run_eval
from patient.db import create_all
from patient.seed import seed_database
from patient.views import create_views
from qa.llm import OpenAIService

# Load OPENAI_API_KEY (and anything else) from the .env file into the
# environment, before we construct the LLM that reads it. Same pattern as
# api/app.py and the other eval scripts.
load_dotenv()


def main():
    # Build a fresh, deterministic database (same steps as the test fixture).
    engine = create_engine("sqlite:///:memory:")
    create_all(engine)
    session = Session(engine)
    anchors = seed_database(session)
    create_views(engine)

    # Real LLM (reads OPENAI_API_KEY from the environment).
    llm = OpenAIService()

    outcomes = run_eval(engine, anchors, llm)

    passed = sum(1 for _, ok in outcomes if ok)
    total = len(outcomes)

    print("\n=== Patient-plane execution accuracy ===")
    for item, ok in outcomes:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] ({item.patient_key}) {item.question}  -> expected {item.expected!r}")
    print(f"\nScore: {passed}/{total} = {passed / total:.0%}\n")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    main()
