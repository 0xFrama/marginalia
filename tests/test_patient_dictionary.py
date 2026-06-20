from patient.dictionary import VIEW_DOCS, build_dictionary

PHI_COLUMNS = {"name", "mrn", "dob"}
EXPECTED_VIEWS = {
    "v_patient_summary",
    "v_conditions",
    "v_medications",
    "v_observations",
}


def test_dictionary_covers_all_views():
    text = build_dictionary()
    for view_name in EXPECTED_VIEWS:
        assert view_name in text


def test_dictionary_has_no_phi_columns():
    for view in VIEW_DOCS:
        for col in view.columns:
            assert col.name not in PHI_COLUMNS
