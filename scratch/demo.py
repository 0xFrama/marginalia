from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from index import QdrantStore
from masking.identifiers import get_patient_identifiers
from masking.mask import mask
from orchestrator.graph import GraphOrchestrator
from patient.db import create_all
from patient.seed import seed_database
from patient.views import create_views
from qa.llm import OpenAIService
from retrieval import Retriever


def line():
    print("=" * 64)


def main():
    # --- build the deterministic patient database ---
    engine = create_engine("sqlite:///:memory:")
    create_all(engine)
    session = Session(engine)
    anchors = seed_database(session)
    create_views(engine)

    patient = anchors["diabetic"]  # Arthur Dent
    pid = patient.patient_id

    # --- the doctor's question (contains the patient's real name) ---
    question = f"Is {patient.name}'s HbA1c above the recommended target?"

    line()
    print(" MARGINALIA - two-plane clinical assistant (demo)")
    line()
    print(f"\nDoctor's question:\n  {question}")
    print(f"\nPatient in scope: {patient.name} (id={pid})")

    # --- show the PHI-masking boundary ---
    identifiers = get_patient_identifiers(engine, pid)
    masked_q, _ = mask(question, identifiers)
    print("\n[PHI masking] real identifiers (never leave the system):")
    for label, value in identifiers.items():
        print(f"    {label} = {value}")
    print(f"\n[PHI masking] question as sent to outside services:\n  {masked_q}")

    # --- run he graph, pausing for human approval --- #
    llm = OpenAIService()
    retriever = Retriever(QdrantStore())
    orch = GraphOrchestrator(engine, llm, retriever)

    def approver(sql):
        print(f"\n[human-in-the-loop] generated patient SQL awaiting approval:\n {sql}")
        reply = input("Approve running this SQL? [y/n] ").strip().lower()
        return "approve" if reply.startswith("y") else "reject"

    result = orch.answer(question, pid, approver=approver)

    print(f"\n[router] planes selected: {result.planes}")

    print("\n[answer] (name restored for the doctor):")
    print("  " + result.answer.replace("\n", "\n  "))

    print("\n[evidence gathered from BOTH planes, fused and renumbered]:")
    for src in result.sources:
        if src.kind == "patient":
            print(f"  [{src.citation_id}] PATIENT RECORD  ->  {src.text}")
            print(f"       provenance SQL: {src.sql}")
        else:
            snippet = " ".join(src.text.split())[:90]
            print(
                f"  [{src.citation_id}] GUIDELINE  ->  {src.source_file} "
                f"(p{src.page_start})"
            )
            print(f'       "{snippet}..."')

    cited = [s.citation_id for s in result.cited_sources]
    print(f"\n[citations the answer used]: {cited}")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    main()
