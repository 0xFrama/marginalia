from masking.mask import mask, unmask

IDS = {
    "PATIENT_NAME": "Arthur Dent",
    "MRN": "MRN-DET-001",
    "DOB": "1979-03-11",
}


def test_mask_replaces_present_identifier():
    masked, mapping = mask("Is Arthur Dent's HbA1c high?", IDS)

    assert masked == "Is [PATIENT_NAME]'s HbA1c high?"
    # only the value that actually appeared is recorded
    assert mapping == {"[PATIENT_NAME]": "Arthur Dent"}


def test_mask_ignores_absent_identifiers():
    # none of the identifiers appear -> text unchanged, empty map
    masked, mapping = mask("What is the HbA1c target?", IDS)
    assert masked == "What is the HbA1c target?"
    assert mapping == {}


def test_mask_handles_multiple_identifiers():
    text = "Arthur Dent, MRN-DET-001, born 1979-03-11"
    masked, mapping = mask(text, IDS)
    assert "Arthur Dent" not in masked
    assert "MRN-DET-001" not in masked
    assert "1979-03-11" not in masked
    assert len(mapping) == 3


def test_round_trip_restores_original():
    original = "Is Arthur Dent's HbA1c high?"
    masked, mapping = mask(original, IDS)
    assert unmask(masked, mapping) == original


def test_empty_identifiers_is_a_noop():
    masked, mapping = mask("nothing to hide here", {})
    assert masked == "nothing to hide here"
    assert mapping == {}
    assert unmask(masked, mapping) == "nothing to hide here"
