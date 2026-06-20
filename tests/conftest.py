import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from patient.db import create_all
from patient.seed import seed_database
from patient.views import create_views


@pytest.fixture(
    scope="function"
)  # 'function' scope means this runs fresh for EVERY single test
def db_session():
    # 1. Create a brand-new, completely isolated in-memory database engine
    engine_ = create_engine("sqlite:///:memory:")

    # 2. Build the tables from scratch
    create_all(engine_)

    # 3. Open a Session, seed the deterministic anchor patients and commit them
    session_ = Session(engine_)
    anchors = seed_database(session_)

    # 4. Create the curated de-identified views on top of the seeded tables
    create_views(engine_)

    # 5. yield both the session and the anchor data to the test
    yield session_, anchors

    # 6. teardown: when the test ends, the engine is destroyed, freeing up the RAM
    session_.close()
    engine_.dispose()
