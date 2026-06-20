from patient.schema import Observations, Patients


def test_seed_creates_patients(db_session):
    session, _ = db_session
    # Guaranteed three anchors regardless of the random padding
    assert session.query(Patients).count() >= 3


def test_seeded_patient_has_recent_observations(db_session):
    session, _ = db_session
    mrn_det_001_patient = session.query(Patients).filter_by(mrn="MRN-DET-001").one()
    results = (
        session.query(Observations)
        .filter_by(patient_id=mrn_det_001_patient.patient_id, code="HbA1c")
        .all()
    )
    assert len(results) >= 1
