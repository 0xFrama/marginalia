from sqlalchemy import text

ALL_VIEW_DDLS = [
    "CREATE VIEW IF NOT EXISTS v_conditions AS SELECT patient_id, code, description, onset_date FROM conditions",
    "CREATE VIEW IF NOT EXISTS v_patient_summary AS SELECT patient_id, sex, CAST((julianday('now') - julianday(dob)) / 365.25 AS INTEGER) AS age FROM patients",
    "CREATE VIEW IF NOT EXISTS v_medications AS SELECT patient_id, drug_name, dose, status, started_on FROM medications",
    "CREATE VIEW IF NOT EXISTS v_observations AS SELECT patient_id, code, value, unit, taken_at FROM observations",
]


def create_views(engine):
    for ddl in ALL_VIEW_DDLS:
        with engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()
