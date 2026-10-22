"""
Shared fixtures for export/import integration tests.

Migrated from backend/tests/conftest.py — provides the DB engine, session,
and cleanup logic that the export/import integration tests depend on.
"""

import os
import glob
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_db_url():
    """Create test database and return its URL.

    IMPORTANT: Uses TEST_DB_* env vars (matching docker-compose db_test service)
    to guarantee tests never touch the dev/prod database.  The defaults point to
    the dedicated test PostgreSQL on port 5433.
    """
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    host = os.getenv("TEST_DB_HOST", "localhost")
    port = os.getenv("TEST_DB_PORT", "5433")
    user = os.getenv("TEST_DB_USER", "test_user")
    password = os.getenv("TEST_DB_PASSWORD", "test_pass")
    test_db_name = "mattin_test_temp"

    # Safety guard: refuse to connect to dev/prod database port
    if port == "5432":
        raise RuntimeError(
            "SAFETY: TEST_DB_PORT is 5432 (dev/prod). "
            "Expected test DB on port 5433. Refusing to run."
        )

    conn = psycopg2.connect(
        dbname="postgres", user=user, password=password, host=host, port=port
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    try:
        cursor.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
    except Exception:
        pass

    cursor.execute(f"CREATE DATABASE {test_db_name}")
    cursor.close()
    conn.close()

    yield f"postgresql://{user}:{password}@{host}:{port}/{test_db_name}"

    # Cleanup
    conn = psycopg2.connect(
        dbname="postgres", user=user, password=password, host=host, port=port
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT pg_terminate_backend(pg_stat_activity.pid)
        FROM pg_stat_activity
        WHERE pg_stat_activity.datname = '{test_db_name}'
        AND pid <> pg_backend_pid()
    """)
    try:
        cursor.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
    except Exception:
        pass
    cursor.close()
    conn.close()


@pytest.fixture(scope="session")
def engine(test_db_url):
    """Create database engine for tests."""
    from sqlalchemy import create_engine
    from db.database import Base

    eng = create_engine(test_db_url)
    # Fork: PostgreSQL ENUM types declared with create_type=False (Alembic-owned;
    # see spoint001 and friends) are never emitted by create_all. This throwaway
    # database is built via create_all only, so create the types here first.
    import sqlalchemy as _sa
    import psycopg.errors as _psycopg_errors
    with eng.begin() as conn:
        for table in Base.metadata.tables.values():
            for column in table.columns:
                typ = column.type
                if isinstance(typ, _sa.Enum) and not typ.create_type:
                    labels = ", ".join(f"'{label}'" for label in typ.enums)
                    try:
                        conn.exec_driver_sql(f"CREATE TYPE {typ.name} AS ENUM ({labels})")
                    except _psycopg_errors.DuplicateObject:
                        pass  # already exists from a previous run's failed cleanup
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture
def db_session(engine):
    """Create a new database session for each test."""
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test export files after each test."""
    yield
    for pattern in ["test_export_*.json", "*_modified.json"]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except Exception:
                pass
