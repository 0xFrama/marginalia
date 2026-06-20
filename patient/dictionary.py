from dataclasses import dataclass, field


@dataclass
class Column:
    name: str
    type: str
    description: str
    examples: list[str] = field(default_factory=list)


@dataclass
class ViewDoc:
    name: str
    description: str
    columns: list[Column]


VIEW_DOCS = [
    ViewDoc(
        name="v_patient_summary",
        description="One row per patient: de-identified demographics.",
        columns=[
            Column(
                "patient_id", "int", "Patient identifier (use this to scope queries)."
            ),
            Column("sex", "text", "Patient sex.", ["Male", "Female"]),
            Column("age", "int", "Patient age in years, computed from date of birth."),
        ],
    ),
    ViewDoc(
        name="v_conditions",
        description="One row per diagnosed condition (problem list).",
        columns=[
            Column("patient_id", "int", "Patient identifier."),
            Column("code", "text", "ICD-10 diagnosis code.", ["E11.9", "I10"]),
            Column(
                "description",
                "text",
                "Human-readable diagnosis.",
                [
                    "Type 2 diabetes mellitus without complications",
                    "Essential (primary) hypertension",
                ],
            ),
            Column("onset_date", "date", "Date the condition was first recorded."),
        ],
    ),
    ViewDoc(
        name="v_medications",
        description="One row per medication the patient is or was on.",
        columns=[
            Column("patient_id", "int", "Patient identifier."),
            Column(
                "drug_name",
                "text",
                "Name of the medication.",
                ["Metformin", "Chlorthalidone", "ACEi"],
            ),
            Column("dose", "text", "Dosage.", ["500mg", "20mg"]),
            Column(
                "status",
                "text",
                "Whether the medication is current.",
                ["Active", "Stopped"],
            ),
            Column("started_on", "date", "Date the medication was started."),
        ],
    ),
    ViewDoc(
        name="v_observations",
        description="One row per lab result or vital sign measurement, timestamped.",
        columns=[
            Column("patient_id", "int", "Patient identifier."),
            Column("code", "text", "What was measured.", ["HbA1c", "SystolicBP"]),
            Column("value", "float", "The measured number."),
            Column("unit", "text", "Unit of the value.", ["%", "mmHg"]),
            Column(
                "taken_at",
                "datetime",
                "When the measurement was taken (use for 'most recent').",
            ),
        ],
    ),
]


def build_dictionary() -> str:
    lines = []
    for view in VIEW_DOCS:
        lines.append(f"View: {view.name}")
        lines.append(f"Description: {view.description}")
        lines.append("Columns:")

        for col in view.columns:
            col_line = f" - {col.name} ({col.type}): {col.description}"

            if col.examples:
                col_line += f" Examples: {', '.join(col.examples)}"

            lines.append(col_line)

        lines.append("")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    print(build_dictionary())
