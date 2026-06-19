import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from patient.db import create_all
from patient.schema import Observations, Patients
from patient.seed import seed_database


@pytest.fixture(
    scope="function"
)  # 'function' scope means this runs fresh for EVERY single test
def db_session():
    # 1. Create a brand-new, completely isolated in-memory databse engine
    engine_ = create_engine("sqlite:///:memory:")

    # 2. Build the tables from scratch
    create_all(engine_)

    # 3. Open a Session, seed the deterministic anchor pattients and commit them
    session_ = Session(engine_)
    anchors = seed_database(session_)

    # 4. yield both the session and the anchor data to the test
    yield session_, anchors

    # 5. teardown: wen the test ends, the engine is destroyed, freeing up the RAM
    session_.close()
    engine_.dispose()


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
