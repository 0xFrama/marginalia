from datetime import date, datetime, timedelta
import random
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from patient.db import create_all

from patient.schema import (
    Patients,
    Conditions,
    Medications,
    Observations,
    Encounters,
    Allergies,
    Billing,
    Appointments,
    RawHl7Staging,
    AuditLog,
)

fake = Faker()

DIABETES = {"code": "E11.9", "desc": "Type 2 diabetes mellitus without complications"}
HYPERTENSION = {"code": "I10", "desc": "Essential (primary) hypertension"}


def seed_database(session: Session) -> dict:
    anchors = {}

    medications_pool = {
        "db": ["Metformin", "SGLT2 inhibitors"],
        "htn": ["Chlorthalidone", "Furosemide"],
        "both": ["ACEi", "ARBs"],
    }

    # =========================================================================
    # 1. ANCHOR PATIENTS (Deterministic for testing)
    # =========================================================================

    # Patient A: Pure Diabetic (Testing: chronological HbA1c history)
    p_diabetic = Patients(
        name="Arthur Dent", dob=date(1965, 3, 12), mrn="MRN-DET-001", sex="Male"
    )
    session.add(p_diabetic)
    session.flush()

    c_db = Conditions(
        patient_id=p_diabetic.patient_id,
        code=DIABETES["code"],
        description=DIABETES["desc"],
        onset_date=date(2024, 1, 1),
    )
    m_db = Medications(
        patient_id=p_diabetic.patient_id,
        name=random.choice(medications_pool["db"]),
        dose="500mg",
        status="Active",
        started_on=date(2024, 1, 5),
    )

    obs_db1 = Observations(
        patient_id=p_diabetic.patient_id,
        code="HbA1c",
        value=8.2,
        unit="%",
        taken_at=datetime(2025, 1, 15, 9, 0),
    )
    obs_db2 = Observations(
        patient_id=p_diabetic.patient_id,
        code="HbA1c",
        value=7.5,
        unit="%",
        taken_at=datetime(2025, 6, 20, 10, 30),
    )
    obs_db3 = Observations(
        patient_id=p_diabetic.patient_id,
        code="HbA1c",
        value=6.8,
        unit="%",
        taken_at=datetime(2026, 2, 10, 8, 15),
    )  # Latest

    session.add_all([c_db, m_db, obs_db1, obs_db2, obs_db3])
    anchors["diabetic"] = p_diabetic

    # Patient B: Pure Hypertensive
    p_hyper = Patients(
        name="Elena Gilbert", dob=date(1971, 8, 24), mrn="MRN-HYP-002", sex="Female"
    )
    session.add(p_hyper)
    session.flush()

    c_htn = Conditions(
        patient_id=p_hyper.patient_id,
        code=HYPERTENSION["code"],
        description=HYPERTENSION["desc"],
        onset_date=date(2023, 5, 12),
    )
    m_htn = Medications(
        patient_id=p_hyper.patient_id,
        name=random.choice(medications_pool["htn"]),
        dose="20mg",
        status="Active",
        started_on=date(2023, 5, 14),
    )
    obs_htn1 = Observations(
        patient_id=p_hyper.patient_id,
        code="SystolicBP",
        value=145.0,
        unit="mmHg",
        taken_at=datetime(2025, 11, 1, 14, 0),
    )
    obs_htn2 = Observations(
        patient_id=p_hyper.patient_id,
        code="SystolicBP",
        value=130.0,
        unit="mmHg",
        taken_at=datetime(2026, 4, 18, 11, 0),
    )  # Latest

    session.add_all([c_htn, m_htn, obs_htn1, obs_htn2])
    anchors["hypertensive"] = p_hyper

    # Patient C: The Comorbid Combo Anchor
    p_comorbid = Patients(
        name="Bruce Banner", dob=date(1969, 12, 18), mrn="MRN-COM-003", sex="Male"
    )
    session.add(p_comorbid)
    session.flush()

    c_both1 = Conditions(
        patient_id=p_comorbid.patient_id,
        code=DIABETES["code"],
        description=DIABETES["desc"],
        onset_date=date(2022, 11, 1),
    )
    c_both2 = Conditions(
        patient_id=p_comorbid.patient_id,
        code=HYPERTENSION["code"],
        description=HYPERTENSION["desc"],
        onset_date=date(2022, 11, 1),
    )
    m_both = Medications(
        patient_id=p_comorbid.patient_id,
        name=random.choice(medications_pool["both"]),
        dose="10mg",
        status="Active",
        started_on=date(2022, 11, 5),
    )
    obs_com1 = Observations(
        patient_id=p_comorbid.patient_id,
        code="HbA1c",
        value=7.1,
        unit="%",
        taken_at=datetime(2025, 8, 12, 9, 0),
    )
    obs_com2 = Observations(
        patient_id=p_comorbid.patient_id,
        code="SystolicBP",
        value=138.0,
        unit="mmHg",
        taken_at=datetime(2025, 8, 12, 9, 0),
    )

    session.add_all([c_both1, c_both2, m_both, obs_com1, obs_com2])
    anchors["comorbid"] = p_comorbid

    # =========================================================================
    # 2. FAKER PADDING (Remaining 12 patients for bulk volume)
    # =========================================================================
    padded_patients = []
    for _ in range(12):
        gender = fake.random_element(elements=("Male", "Female"))
        random_patient = Patients(
            name=fake.name(),
            dob=fake.date_of_birth(minimum_age=55, maximum_age=95),
            mrn=fake.bothify(text="MRN-######-??").upper(),
            sex=gender,
        )
        padded_patients.append(random_patient)

    session.add_all(padded_patients)
    session.flush()

    for patient in padded_patients:
        scenario = random.choices(["htn", "db", "both"], weights=[0.45, 0.45, 0.10])[0]

        if scenario in ["htn", "both"]:
            session.add(
                Conditions(
                    patient_id=patient.patient_id,
                    code=HYPERTENSION["code"],
                    description=HYPERTENSION["desc"],
                    onset_date=fake.date_between(start_date="-3y", end_date="today"),
                )
            )
            session.add(
                Observations(
                    patient_id=patient.patient_id,
                    code="SystolicBP",
                    value=float(random.randint(120, 160)),
                    unit="mmHg",
                    taken_at=datetime.now() - timedelta(days=random.randint(1, 100)),
                )
            )
        if scenario in ["db", "both"]:
            session.add(
                Conditions(
                    patient_id=patient.patient_id,
                    code=DIABETES["code"],
                    description=DIABETES["desc"],
                    onset_date=fake.date_between(start_date="-3y", end_date="today"),
                )
            )
            session.add(
                Observations(
                    patient_id=patient.patient_id,
                    code="HbA1c",
                    value=round(random.uniform(5.5, 9.0), 1),
                    unit="%",
                    taken_at=datetime.now() - timedelta(days=random.randint(1, 100)),
                )
            )

        session.add(
            Medications(
                patient_id=patient.patient_id,
                name=random.choice(medications_pool[scenario]),
                dose="Standard",
                status="Active",
                started_on=fake.date_between(start_date="-2y", end_date="today"),
            )
        )

    session.add(
        Encounters(
            patient_id=p_diabetic.patient_id,
            reason="Routine Follow-up",
            encounter_date=date(2025, 6, 20),
        )
    )
    session.add(
        Allergies(
            patient_id=p_diabetic.patient_id,
            substance="Penicillin",
            reaction="Rash",
        )
    )
    # =========================================================================
    # 3. TIER-2 NOISE TABLES (Minimal rows just to pass join filters)
    # =========================================================================
    session.add(
        Billing(
            pat_ref="MRN-DET-001", amt_cents=15000, dx_cd="E11.9", ts=datetime.now()
        )
    )
    session.add(
        Appointments(
            pat_ref="MRN-DET-001",
            prov_id=99,
            slot_ts=datetime.now() + timedelta(days=7),
            status_cd="Scheduled",
        )
    )
    session.add(
        RawHl7Staging(
            rcvd_ts=datetime.now(),
            raw_msg="MSH|^~\\&|EPIC||||20260619||ORU^R01|12345|P|2.5|",
        )
    )
    session.add(
        AuditLog(
            usr="dr_smith", act="VIEW_PATIENT", obj_ref="MRN-DET-001", ts=datetime.now()
        )
    )

    session.commit()
    return anchors


# Execution context
if __name__ == "__main__":
    engine = create_engine("sqlite:///:memory:")
    create_all(engine)

    with Session(engine) as session:
        anchors = seed_database(session)

        mrn_det_001_patient = session.query(Patients).filter_by(mrn="MRN-DET-001").one()
        print(mrn_det_001_patient)
        print(mrn_det_001_patient.patient_id)
        stmt = session.query(Observations).filter_by(
            patient_id=mrn_det_001_patient.patient_id, code="HbA1c"
        )
        print(stmt)
        print(
            f"Seeding completed! Verified Anchor Patient ID: {anchors['diabetic'].patient_id} ({anchors['diabetic'].name})"
        )
