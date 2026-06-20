from dataclasses import dataclass, field

from patient.db import run_patient_query
from patient.dictionary import build_dictionary
from patient.sql_generation import generate_sql
from patient.validation import validate


@dataclass
class PatientQueryResult:
    ok: bool
    rows: list = field(default_factory=list)
    sql: str | None = None
    error: str | None = None


def answer_patient_query(question, patient_id, engine, llm, max_attempts=2):
    dictionary = build_dictionary()
    last_error = None

    for attempt in range(max_attempts):
        prompt_question = question
        if last_error:
            prompt_question = (
                f"{question}\n\nPrevious attempt failed: {last_error}\nPlease fix it."
            )

        try:
            raw_sql = generate_sql(prompt_question, dictionary, patient_id, llm)
            safe_sql = validate(raw_sql)
            rows = run_patient_query(engine, safe_sql, patient_id)
            return PatientQueryResult(ok=True, rows=rows, sql=safe_sql, error=None)
        except Exception as e:
            last_error = str(e)

    return PatientQueryResult(ok=False, error=last_error)
