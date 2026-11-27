"""Round-trip + backfill test of the real `rerank001` migration (rerank services, Agent rerank columns).

Same ephemeral-database + real `alembic` CLI pattern as
`tests/integration/a2a_server/test_a2a001_migration_round_trip.py`: the shared
`db` fixture builds its schema from the ORM models (which already contain
everything rerank001 adds), so it cannot exercise the migration itself.

Requires: the `db_test` docker-compose service on port 5433.
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

DB_HOST = os.environ.get("RERANK001_TEST_DB_HOST", "localhost")
DB_PORT = os.environ.get("RERANK001_TEST_DB_PORT", "5433")
DB_SUPERUSER = os.environ.get("RERANK001_TEST_DB_USER", "test_user")
DB_SUPERUSER_PASSWORD = os.environ.get("RERANK001_TEST_DB_PASSWORD", "test_pass")  # pragma: allowlist secret
MAINTENANCE_DB = os.environ.get("RERANK001_TEST_DB_MAINT", "test_db")
EPHEMERAL_DB_NAME = "mattin_test_rerank001_round_trip"
PARENT_REVISION = "lexfts001"


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


def _insert_agent(conn, app_id, name, strategy_sql, top_n):
    return conn.execute(
        text(
            'INSERT INTO "Agent" (name, app_id, type, rag_strategy, rag_rerank_top_n) '
            "VALUES (:n, :a, 'agent', " + strategy_sql + ", :t) RETURNING agent_id"
        ),
        {"n": name, "a": app_id, "t": top_n},
    ).scalar_one()


def test_rerank001_backfill_constraints_round_trip(ephemeral_db):
    _alembic_ok("upgrade", PARENT_REVISION)
    with ephemeral_db.begin() as conn:
        app_id = conn.execute(
            text('INSERT INTO "App" (name, slug) VALUES (:n, :s) RETURNING app_id'),
            {"n": "rerank001 app", "s": "rerank001-app"},
        ).scalar_one()
        ids = {
            "rerank": _insert_agent(conn, app_id, "a1", "'[\"rerank\"]'::json", 7),
            "cross": _insert_agent(conn, app_id, "a2", "'[\"cross_encoder_rerank\"]'::json", None),
            "empty": _insert_agent(conn, app_id, "a3", "'[]'::json", 5),
            "null": _insert_agent(conn, app_id, "a4", "NULL", None),
            "string": _insert_agent(conn, app_id, "a5", "'\"rerank\"'::json", 3),
            "jnull": _insert_agent(conn, app_id, "a6", "'null'::json", None),
            "mixed": _insert_agent(conn, app_id, "a7", "'[\"rerank\",\"x\"]'::json", None),
        }

    _alembic_ok("upgrade", "rerank001")

    def rows():
        with ephemeral_db.connect() as conn:
            return {
                r[0]: r[1:]
                for r in conn.execute(
                    text('SELECT agent_id, rag_rerank_enabled, rag_rerank_service_id, rag_rerank_top_n FROM "Agent"')
                )
            }

    r = rows()
    assert r[ids["rerank"]] == (True, None, 7)
    assert r[ids["cross"]] == (True, None, None)
    assert r[ids["string"]] == (True, None, 3)
    assert r[ids["empty"]] == (False, None, 5)
    assert r[ids["null"]] == (False, None, None)
    assert r[ids["jnull"]] == (False, None, None)
    assert r[ids["mixed"]] == (True, None, None)
    with ephemeral_db.connect() as conn:
        cols = {c["name"] for c in inspect(conn).get_columns("Agent")}
        assert not ({"rag_strategy", "rag_rerank_similarity_threshold"} & cols)

    with ephemeral_db.connect() as conn:
        conn.execute(text(
            "INSERT INTO rerank_service (name, provider, model_name, is_platform_default) "
            "VALUES ('p1', 'CohereCompatible', 'm', true)"))
        conn.commit()
        with pytest.raises(IntegrityError):  # second platform default
            conn.execute(text(
                "INSERT INTO rerank_service (name, provider, model_name, is_platform_default) "
                "VALUES ('p2', 'CohereCompatible', 'm', true)"))
        conn.rollback()
        with pytest.raises(IntegrityError):  # app-scoped service flagged as platform default
            conn.execute(text(
                "INSERT INTO rerank_service (name, provider, model_name, app_id, is_platform_default) "
                "VALUES ('p3', 'CohereCompatible', 'm', :a, true)"), {"a": app_id})
        conn.rollback()
        conn.execute(text("DELETE FROM rerank_service"))
        conn.commit()

    # FK behaviour: Agent reference blocks deletion (NO ACTION); App default is SET NULL.
    with ephemeral_db.begin() as conn:
        svc = conn.execute(text(
            "INSERT INTO rerank_service (name, provider, model_name) VALUES ('s', 'CohereCompatible', 'm') "
            "RETURNING service_id")).scalar_one()
        conn.execute(text('UPDATE "App" SET default_rerank_service_id = :s WHERE app_id = :a'), {"s": svc, "a": app_id})
        conn.execute(text('UPDATE "Agent" SET rag_rerank_service_id = :s WHERE agent_id = :g'),
                     {"s": svc, "g": ids["rerank"]})
    with ephemeral_db.connect() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(text("DELETE FROM rerank_service WHERE service_id = :s"), {"s": svc})
        conn.rollback()
    with ephemeral_db.begin() as conn:
        conn.execute(text('UPDATE "Agent" SET rag_rerank_service_id = NULL WHERE agent_id = :g'), {"g": ids["rerank"]})
        conn.execute(text("DELETE FROM rerank_service WHERE service_id = :s"), {"s": svc})
        assert conn.execute(text('SELECT default_rerank_service_id FROM "App" WHERE app_id = :a'),
                            {"a": app_id}).scalar_one() is None
    # Live references at downgrade time must not block it.
    with ephemeral_db.begin() as conn:
        svc = conn.execute(text(
            "INSERT INTO rerank_service (name, provider, model_name) VALUES ('s2', 'CohereCompatible', 'm') "
            "RETURNING service_id")).scalar_one()
        conn.execute(text('UPDATE "App" SET default_rerank_service_id = :s WHERE app_id = :a'), {"s": svc, "a": app_id})
        conn.execute(text('UPDATE "Agent" SET rag_rerank_service_id = :s WHERE agent_id = :g'),
                     {"s": svc, "g": ids["rerank"]})

    _alembic_ok("downgrade", PARENT_REVISION)
    with ephemeral_db.connect() as conn:
        insp = inspect(conn)
        assert "rerank_service" not in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("Agent")}
        assert {"rag_strategy", "rag_rerank_similarity_threshold"} <= cols
        assert not ({"rag_rerank_enabled", "rag_rerank_service_id"} & cols)
        assert "default_rerank_service_id" not in {c["name"] for c in insp.get_columns("App")}
        got = {
            r[0]: r[1:]
            for r in conn.execute(text('SELECT agent_id, rag_strategy::text, rag_rerank_similarity_threshold FROM "Agent"'))
        }
    for k in ("rerank", "cross", "string", "mixed"):
        assert json.loads(got[ids[k]][0]) == ["rerank"]
        assert got[ids[k]][1] is None
    for k in ("empty", "null", "jnull"):
        assert got[ids[k]] == (None, None)

    _alembic_ok("upgrade", "rerank001")
    with ephemeral_db.connect() as conn:
        insp = inspect(conn)
        assert "rerank_service" in insp.get_table_names()
        assert "rag_strategy" not in {c["name"] for c in insp.get_columns("Agent")}
