"""Integration test for the D-10 usage-accounting fix in
`services.global_memory_service.run_write_path` (spec `user-memory-mem0`,
step_008 security audit, MEDIUM finding).

Runs against the REAL test database (not a mocked session): the whole point
of the bug being fixed is that `UsageRecordRepository.increment()` only
`flush()`es, so a session that is closed without an explicit `commit()`
silently rolls the row back. A test built entirely on `MagicMock` sessions
cannot prove that a real Postgres transaction actually landed — only a real
engine/session pair can.

Uses `db.database.SessionLocal` directly (never the `tests/conftest.py` `db`
fixture): that fixture's per-test session is a SAVEPOINT inside one held-open
connection, invisible to any OTHER connection (including the separate
`SessionLocal()` sessions `run_write_path` itself opens) until the outer
transaction commits — which never happens in that fixture. Proving the row
survives across a genuinely different, later-opened session requires
genuinely committed data, so this test creates and cleans up its own rows
directly.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from db.database import SessionLocal
from models.usage_record import UsageRecord
from models.user import User
from services import global_memory_service
from tools.memory import settings as memory_settings
from tools.memory.identity import MemoryIdentity
from tools.memory.resolution import ResolvedMemoryServices

pytestmark = pytest.mark.integration


def _make_user() -> int:
    session = SessionLocal()
    try:
        user = User(email=f"usage_write_path_test_{int(time.time() * 1000)}@test.com", name="Usage Test User")
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.user_id
    finally:
        session.close()


def _cleanup(user_id: int) -> None:
    session = SessionLocal()
    try:
        session.query(UsageRecord).filter(UsageRecord.user_id == user_id).delete()
        session.query(User).filter(User.user_id == user_id).delete()
        session.commit()
    finally:
        session.close()


def _identity(user_id: int) -> MemoryIdentity:
    root = memory_settings.namespace_root()
    return MemoryIdentity(
        app_id=1, user_id=user_id, agent_id=1, namespace=(root, "1", str(user_id))
    )


def _resolved_system_extraction() -> ResolvedMemoryServices:
    return ResolvedMemoryServices(
        extraction_service_id=10,
        extraction_source="system",
        extraction_is_system=True,
        embedding_service_id=20,
        embedding_source="app_config",
        embedding_is_system=False,
    )


@pytest.mark.asyncio
async def test_usage_row_survives_session_close_even_when_extraction_finds_zero_facts(
    monkeypatch, test_engine
):
    """MEDIUM (step_008 audit, D-10):
      1. `UsageRecordRepository.increment()` only `flush()`es — the request
         path gets a later commit for free, but `run_write_path`'s
         background task does not, so the row used to be rolled back and
         never recorded. Fixed by committing the usage session before
         closing it.
      2. Usage used to be recorded only after a successful `client.add`,
         so a `SKIP_NO_FACTS` turn (extraction ran, found nothing) cost
         real provider money for zero quota consumed. Fixed by recording
         usage immediately after `extract_facts` returns, regardless of
         its result.

    This test drives BOTH fixes at once: `extract_facts` is mocked to
    return `[]`, and the assertion reads the row back from a completely
    FRESH `SessionLocal()` — never the session `run_write_path` used
    internally — proving the commit really landed in Postgres.
    """
    user_id = _make_user()
    try:
        monkeypatch.setattr("services.tier_enforcement_service.is_self_managed", lambda: False)
        monkeypatch.setattr("services.usage_tracking_service.is_self_managed", lambda: False)

        identity = _identity(user_id)
        resolved = _resolved_system_extraction()

        with patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=[])):
            await global_memory_service.run_write_path(
                identity=identity,
                resolved=resolved,
                llm=None,
                embeddings=None,
                embedding_service_id=20,
                original_user_message="hi",
                response_text="hello",
                turn_used_tool_call=False,
            )

        fresh_session = SessionLocal()
        try:
            record = (
                fresh_session.query(UsageRecord)
                .filter(UsageRecord.user_id == user_id)
                .first()
            )
            assert record is not None, (
                "usage row did not survive run_write_path's session closing — "
                "the commit-before-close fix regressed"
            )
            assert record.call_count == 1
        finally:
            fresh_session.close()
    finally:
        _cleanup(user_id)
