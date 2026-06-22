from patient.pipeline import answer_patient_query
from patient.evidence import rows_to_evidence


def get_patient_evidence(question, patient_id, engine, llm):
    result = answer_patient_query(question, patient_id, engine, llm)
    return rows_to_evidence(result)
