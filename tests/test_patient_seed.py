import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from patient.db import create_all
from patient.schema import Observations, Patients
from patient.seed import seed_database


@pytest.fixture(scope="function")
def db_session():
    engine_ = create_engine("sqlite:///:memory:")
    create_all(engine_)

    session_ = Session(engine_)

    seed_database(session_)

    yield session_

    session_.close()
    engine_.dispose()


def test_seed_creates_patients(db_session):
    # Guaranteed three anchors regardless of the random padding
    assert db_session.query(Patients).count() >= 3


def test_seeded_patient_has_recent_observations(db_session):
    mrn_det_001_patient = db_session.query(Patients).filter_by(mrn="MRN-DET-001").one()
    results = (
        db_session.query(Observations)
        .filter_by(patient_id=mrn_det_001_patient.patient_id, code="HbA1c")
        .all()
    )
    assert len(results) >= 1
