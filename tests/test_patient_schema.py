from sqlalchemy import create_engine, inspect

from patient.db import create_all
from patient.schema import Base

engine = create_engine("sqlite:///:memory:")


def test_schema_creates_all_tables():
    """When you use :memory:, the entire database lives within the computer's RAM. Fast but volatile."""
    create_all(engine)
    table_set = set(inspect(engine).get_table_names())

    assert table_set == {
        "patients",
        "conditions",
        "medications",
        "observations",
        "encounters",
        "allergies",
        "billing",
        "appointments",
        "raw_hl7_staging",
        "audit_log",
    }


def test_patient_has_phi_columns():
    create_all(engine)
    patient_cols = inspect(engine).get_columns("patients")
    actual = [col["name"] for col in patient_cols]
    assert {"name", "mrn", "dob"}.issubset(actual)
