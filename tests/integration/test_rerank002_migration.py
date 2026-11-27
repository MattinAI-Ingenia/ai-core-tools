"""Round-trip test of the real `rerank002` migration (App default rerank FK RESTRICT).

Same ephemeral-database + real `alembic` CLI pattern as
`tests/integration/a2a_server/test_a2a001_migration_round_trip.py`: the shared
`db` fixture builds its schema from the ORM models (which already contain
everything rerank002 adds), so it cannot exercise the migration itself.

Requires: the `db_test` docker-compose service (RERANK002_TEST_DB_* env vars).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg2
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]

DB_HOST = os.environ.get("RERANK002_TEST_DB_HOST", "127.0.0.1")
DB_PORT = os.environ.get("RERANK002_TEST_DB_PORT", "5433")
DB_SUPERUSER = os.environ.get("RERANK002_TEST_DB_USER", "test_user")
DB_SUPERUSER_PASSWORD = os.environ.get("RERANK002_TEST_DB_PASSWORD", "test_pass")  # pragma: allowlist secret
MAINTENANCE_DB = os.environ.get("RERANK002_TEST_DB_MAINT", "test_db")
EPHEMERAL_DB_NAME = "mattin_test_rerank002_round_trip"
PARENT_REVISION = "rerank001"


def _admin_connection():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_SUPERUSER, password=DB_SUPERUSER_PASSWORD, dbname=MAINTENANCE_DB
    )
    conn.autocommit = True
    return conn


def _drop_ephemeral_db() -> None:
    conn = _admin_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{EPHEMERAL_DB_NAME}" WITH (FORCE)')
    finally:
        conn.close()


def _create_ephemeral_db() -> None:
    _drop_ephemeral_db()
    conn = _admin_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{EPHEMERAL_DB_NAME}"')
    finally:
        conn.close()


def _run_alembic(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        DATABASE_USER=DB_SUPERUSER,
        DATABASE_PASSWORD=DB_SUPERUSER_PASSWORD,
        DATABASE_HOST=DB_HOST,
        DATABASE_PORT=DB_PORT,
        DATABASE_NAME=EPHEMERAL_DB_NAME,
    )
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
    )


def _alembic_ok(*args: str) -> None:
    result = _run_alembic(*args)
    assert result.returncode == 0, (
        f"alembic {' '.join(args)} failed (rc={result.returncode}).\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.fixture
def ephemeral_db():
    _create_ephemeral_db()
    engine = None
    try:
        url = f"postgresql://{DB_SUPERUSER}:{DB_SUPERUSER_PASSWORD}@{DB_HOST}:{DB_PORT}/{EPHEMERAL_DB_NAME}"
        engine = create_engine(url, pool_pre_ping=True)
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        _drop_ephemeral_db()



def _delete_rule(conn):
    return conn.execute(text(
        "SELECT delete_rule FROM information_schema.referential_constraints "
        "WHERE constraint_name = 'App_default_rerank_service_id_fkey'")).scalar_one()


def test_rerank002_fk_restrict_round_trip(ephemeral_db):
    _alembic_ok("upgrade", "rerank002")
    with ephemeral_db.connect() as conn:
        assert _delete_rule(conn) == "RESTRICT"
    with ephemeral_db.begin() as conn:
        app_id = conn.execute(
            text("INSERT INTO \"App\" (name, slug) VALUES (:n, 'rerank002-app') RETURNING app_id"), {"n": "rerank002 app"}
        ).scalar_one()
        svc = conn.execute(text(
            "INSERT INTO rerank_service (name, provider, model_name) VALUES ('s', 'CohereCompatible', 'm') "
            "RETURNING service_id")).scalar_one()
        conn.execute(text('UPDATE "App" SET default_rerank_service_id = :s WHERE app_id = :a'), {"s": svc, "a": app_id})
    with ephemeral_db.connect() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(text("DELETE FROM rerank_service WHERE service_id = :s"), {"s": svc})
        conn.rollback()

    _alembic_ok("downgrade", PARENT_REVISION)
    with ephemeral_db.connect() as conn:
        assert _delete_rule(conn) == "SET NULL"

    _alembic_ok("upgrade", "rerank002")
    with ephemeral_db.connect() as conn:
        assert _delete_rule(conn) == "RESTRICT"
