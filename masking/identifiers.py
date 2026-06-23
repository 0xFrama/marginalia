from sqlalchemy.orm import Session

from patient.schema import Patients


def get_patient_identifiers(engine, patient_id) -> dict:
    with Session(engine) as session:
        patient = session.get(Patients, patient_id)
        if patient is None:
            return {}
        return {
            "PATIENT_NAME": patient.name,
            "MRN": patient.mrn,
            "DOB": patient.dob.isoformat(),
        }
