import pytest

from sqlalchemy import text

from patient.db import readonly_connection, run_patient_query


def test_write_is_rejected(db_session):
    session, _ = db_session
    engine = session.bind

    with pytest.raises(Exception):
        with readonly_connection(engine) as conn:
            conn.execute(
                text(
                    "INSERT INTO patients (name, dob, mrn, sex) "
                    "VALUES ('x', '2000-01-01', 'MRN-TEST-999', 'Male')"
                )
            )


def test_query_is_scoped_to_patient(db_session):
    session, anchors = db_session
    engine = session.bind

    patient_a = anchors["diabetic"].patient_id
    patient_b = anchors["hypertensive"].patient_id

    unscoped_sql = "SELECT patient_id, code, value FROM v_observations"
    rows = run_patient_query(engine, unscoped_sql, patient_a)

    returned_ids = {row[0] for row in rows}
    assert returned_ids == {patient_a}  # only patient A came back
    assert patient_b not in returned_ids  # patient B never leaked
