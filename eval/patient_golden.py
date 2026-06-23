from dataclasses import dataclass


@dataclass
class EvalItem:
    question: str  # doctor's question
    patient_key: str  # diabetic, hypertensive, comorbid
    expected: str  # expected string to appear in the answer if correct


GOLDEN_SET = [
    EvalItem("What is the patient's latest HbA1c?", "diabetic", "6.8"),
    EvalItem("How many HbA1c readings does the patient have?", "diabetic", "3"),
    EvalItem(
        "What is the patient's latest systolic blood pressure?", "hypertensive", "130"
    ),
    EvalItem("What is the patient's HbA1c?", "comorbid", "7.1"),
    EvalItem("What was the patient's earliest HbA1c?", "diabetic", "8.2"),
    EvalItem(
        "What is the diagnosis code for the patient's diabetes?", "comorbid", "E11.9"
    ),
]
