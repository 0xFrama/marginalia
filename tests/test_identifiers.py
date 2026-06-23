from masking.identifiers import get_patient_identifiers


def test_returns_real_identifiers_for_known_patient(db_session):
    session, anchors = db_session
    engine = session.bind
    patient = anchors["diabetic"]  # Arthur Dent

    ids = get_patient_identifiers(engine, patient.patient_id)

    assert ids["PATIENT_NAME"] == patient.name
    assert ids["MRN"] == patient.mrn
    assert ids["DOB"] == patient.dob.isoformat()  # date returned as text
    assert isinstance(ids["DOB"], str)


def test_unknown_patient_returns_empty_dict(db_session):
    session, anchors = db_session
    engine = session.bind

    # an ID that does not exist -> no crash, empty result
    assert get_patient_identifiers(engine, 999999) == {}
