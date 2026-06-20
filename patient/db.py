from sqlalchemy import text
from patient.schema import Base
from contextlib import contextmanager


def create_all(engine):
    Base.metadata.create_all(engine)


@contextmanager
def readonly_connection(engine):
    with engine.connect() as conn:
        conn.execute(text("PRAGMA query_only = ON;"))
        yield conn


def run_patient_query(engine, sql, patient_id):
    wrapped_sql = f"SELECT * FROM ( {sql} ) WHERE patient_id = :pid"
    with readonly_connection(engine) as conn:
        result = conn.execute(text(wrapped_sql), {"pid": patient_id})
        return result.fetchall()
