import sqlglot

from sqlalchemy import inspect

from patient.views import ALL_VIEW_DDLS


def test_views_exclude_identifiers(db_session):
    session, _ = db_session
    engine = session.bind

    view_cols = set()
    for view_name in inspect(engine).get_view_names():
        for d in inspect(engine).get_columns(view_name):
            view_cols.add(d["name"])

    assert set(inspect(engine).get_view_names()) == {
        "v_patient_summary",
        "v_conditions",
        "v_medications",
        "v_observations",
    }
    assert {"name", "mrn", "dob"}.isdisjoint(view_cols)


def test_views_only_reference_tier1():
    references_tables = set()
    for ddl in ALL_VIEW_DDLS:
        tree = sqlglot.parse_one(ddl)
        # |= adds to the existing set instead of replacing it.
        references_tables |= {t.name for t in tree.find_all(sqlglot.exp.Table)}

    assert {"billing", "appointments", "raw_hl7_staging", "audit_log"}.isdisjoint(
        references_tables
    )
