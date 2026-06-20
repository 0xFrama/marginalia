from models.evidence import Evidence


def rows_to_evidence(result):
    if not result.ok:
        return []

    evidence = []
    for i, row in enumerate(result.rows, start=1):
        text = ", ".join(f"{col}={val}" for col, val in row._mapping.items())
        evidence.append(
            Evidence(
                citation_id=i,
                text=text,
                kind="patient",
                source_label="Patient record",
                sql=result.sql,
            )
        )
    return evidence
