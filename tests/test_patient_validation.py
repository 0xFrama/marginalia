import pytest

from patient.validation import validate, UnsafeSQLError


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DELETE FROM v_observations",
        "UPDATE v_medications SET dose = '0'",
        "SELECT 1; DROP TABLE patients",
        "SELECT * FROM billing",
        "SELECT * FROM patients",
    ],
)
def test_unsafe_sql_is_rejected(bad_sql):
    with pytest.raises(UnsafeSQLError):
        validate(bad_sql)


def test_valid_select_over_view_passes():
    out = validate("SELECT code, value FROM v_observations")
    assert "v_observations" in out


def test_limit_is_injected_when_missing():
    out = validate("SELECT code FROM v_observations")
    assert "LIMIT 100" in out.upper()


def test_existing_limit_is_preserved():
    out = validate("SELECT code FROM v_observations LIMIT 5")
    assert "LIMIT 5" in out.upper()
    assert "LIMIT 100" not in out.upper()
