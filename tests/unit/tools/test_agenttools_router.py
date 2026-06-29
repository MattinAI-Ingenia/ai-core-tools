"""Unit tests for LightRAG skill-routed helpers in agentTools."""
import pytest
from unittest.mock import MagicMock

from tools.agentTools import _is_router_skill_active, LIGHTRAG_ROUTER_SKILL_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_assoc(name):
    assoc = MagicMock()
    assoc.skill = MagicMock()
    assoc.skill.name = name
    return assoc


# ---------------------------------------------------------------------------
# _is_router_skill_active
# ---------------------------------------------------------------------------

def test_returns_true_when_router_skill_present():
    assocs = [make_assoc(LIGHTRAG_ROUTER_SKILL_NAME)]
    assert _is_router_skill_active(assocs) is True


def test_returns_false_when_only_other_skills():
    assocs = [make_assoc("Some Other Skill"), make_assoc("Another Skill")]
    assert _is_router_skill_active(assocs) is False


def test_returns_false_for_empty_list():
    assert _is_router_skill_active([]) is False


def test_returns_false_for_none():
    assert _is_router_skill_active(None) is False


def test_returns_true_when_router_skill_among_others():
    assocs = [
        make_assoc("Unrelated"),
        make_assoc(LIGHTRAG_ROUTER_SKILL_NAME),
        make_assoc("Another"),
    ]
    assert _is_router_skill_active(assocs) is True


def test_ignores_assoc_without_skill():
    assoc_no_skill = MagicMock()
    assoc_no_skill.skill = None
    assocs = [assoc_no_skill, make_assoc(LIGHTRAG_ROUTER_SKILL_NAME)]
    assert _is_router_skill_active(assocs) is True


def test_ignores_assoc_without_skill_and_no_router():
    assoc_no_skill = MagicMock()
    assoc_no_skill.skill = None
    assert _is_router_skill_active([assoc_no_skill]) is False
