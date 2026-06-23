from eval.patient_golden import GOLDEN_SET
from patient.evidence import rows_to_evidence
from patient.pipeline import answer_patient_query


def is_correct(result, expected) -> bool:
    if not result.ok:
        return False

    evidence = rows_to_evidence(result)
    text = " ".join(e.text for e in evidence)
    return expected in text


def run_eval(engine, anchors, llm, gold_set=GOLDEN_SET):
    outcomes = []
    for item in gold_set:
        patient_id = anchors[item.patient_key].patient_id
        result = answer_patient_query(item.question, patient_id, engine, llm)
        passed = is_correct(result, item.expected)
        outcomes.append((item, passed))
    return outcomes
