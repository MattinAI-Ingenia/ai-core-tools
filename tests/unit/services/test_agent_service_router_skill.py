"""Unit tests for LightRAG Query Router skill lifecycle helpers in AgentService."""
import pytest
from unittest.mock import MagicMock, call

from services.agent_service import AgentService, LIGHTRAG_ROUTER_SKILL_NAME, ROUTER_SKILL_CONTENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db(existing_skill=None, agent_skill_count=0):
    """Return a mock Session pre-configured for common query patterns."""
    db = MagicMock()
    # db.query(Skill).filter(...).first() -> existing_skill
    db.query.return_value.filter.return_value.first.return_value = existing_skill
    # db.query(AgentSkill).filter(...).count() -> agent_skill_count
    db.query.return_value.filter.return_value.count.return_value = agent_skill_count
    return db


# ---------------------------------------------------------------------------
# ensure_lightrag_router_skill
# ---------------------------------------------------------------------------

def test_ensure_creates_skill_when_missing():
    svc = AgentService()
    db = make_db(existing_skill=None)

    skill = svc.ensure_lightrag_router_skill(db, app_id=1)

    db.add.assert_called_once()
    added = db.add.call_args[0][0]
    assert added.name == LIGHTRAG_ROUTER_SKILL_NAME
    assert added.content == ROUTER_SKILL_CONTENT
    assert added.app_id == 1
    db.commit.assert_called()
    db.refresh.assert_called()


def test_ensure_returns_existing_without_creating():
    svc = AgentService()
    existing = MagicMock()
    db = make_db(existing_skill=existing)

    result = svc.ensure_lightrag_router_skill(db, app_id=1)

    assert result is existing
    db.add.assert_not_called()


# ---------------------------------------------------------------------------
# attach_skill_to_agent
# ---------------------------------------------------------------------------

def test_attach_adds_association_when_missing():
    svc = AgentService()
    db = make_db(existing_skill=None)  # first() returns None → no existing AgentSkill

    svc.attach_skill_to_agent(db, agent_id=10, skill_id=42)

    db.add.assert_called_once()
    added = db.add.call_args[0][0]
    assert added.agent_id == 10
    assert added.skill_id == 42
    db.commit.assert_called()


def test_attach_skips_when_already_attached():
    svc = AgentService()
    existing_assoc = MagicMock()
    db = make_db(existing_skill=existing_assoc)

    svc.attach_skill_to_agent(db, agent_id=10, skill_id=42)

    db.add.assert_not_called()


# ---------------------------------------------------------------------------
# cleanup_lightrag_router_skill
# ---------------------------------------------------------------------------

def test_cleanup_detaches_and_deletes_orphaned_skill():
    svc = AgentService()
    skill = MagicMock()
    skill.skill_id = 42

    db = MagicMock()
    # First filter().first() returns the skill
    db.query.return_value.filter.return_value.first.return_value = skill
    # After delete, count() returns 0 (orphaned)
    db.query.return_value.filter.return_value.count.return_value = 0

    svc.cleanup_lightrag_router_skill(db, app_id=1, agent_id=10)

    db.query.return_value.filter.return_value.delete.assert_called()
    db.delete.assert_called_once_with(skill)


def test_cleanup_keeps_skill_when_other_agents_use_it():
    svc = AgentService()
    skill = MagicMock()
    skill.skill_id = 42

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = skill
    # Another agent still uses it
    db.query.return_value.filter.return_value.count.return_value = 1

    svc.cleanup_lightrag_router_skill(db, app_id=1, agent_id=10)

    db.delete.assert_not_called()


def test_cleanup_is_noop_when_skill_missing():
    svc = AgentService()
    db = make_db(existing_skill=None)

    svc.cleanup_lightrag_router_skill(db, app_id=1, agent_id=10)

    db.delete.assert_not_called()
    db.commit.assert_not_called()
