"""The LightRAG router skill is never an on-demand skill.

create_agent inlines it into the system prompt when active; every other prompt
builder (top-level agent and agent-as-tool sub-agents alike) goes through
_resolve_skills_for_prompt, which must drop it — it would only mislead a model
that has no dynamic router tool.
"""
import pytest

from models.agent import Agent
from services.agent_service import LIGHTRAG_ROUTER_SKILL_NAME
from tests.unit.tools.test_skill_prompt_unchanged import make_assoc, make_skill
from tools.agentTools import _resolve_skills_for_prompt, _router_skill_attached


def _agent(*skills):
    agent = Agent(agent_id=1, skill_router_enabled=False)
    agent.skill_associations = [make_assoc(s) for s in skills]
    return agent


@pytest.mark.asyncio
async def test_router_skill_is_dropped_and_the_others_kept():
    router = make_skill(1, LIGHTRAG_ROUTER_SKILL_NAME, description="mode rules")
    other = make_skill(2, "Email Drafting", description="Drafts emails")

    snapshots, section = await _resolve_skills_for_prompt(_agent(router, other))

    assert [s.name for s in snapshots] == ["Email Drafting"]
    assert "Email Drafting" in section
    assert LIGHTRAG_ROUTER_SKILL_NAME not in section


@pytest.mark.asyncio
async def test_only_the_router_skill_means_no_skills_at_all():
    router = make_skill(1, LIGHTRAG_ROUTER_SKILL_NAME)

    assert await _resolve_skills_for_prompt(_agent(router)) == ([], None)


def test_router_skill_counts_only_while_attached_and_enabled():
    """Upstream's per-skill kill switch (is_enabled) applies to the router skill too."""
    on = make_skill(1, LIGHTRAG_ROUTER_SKILL_NAME)
    off = make_skill(1, LIGHTRAG_ROUTER_SKILL_NAME, is_enabled=False)

    assert _router_skill_attached(_agent(on))
    assert not _router_skill_attached(_agent(off))
    assert not _router_skill_attached(_agent(make_skill(2, "Email Drafting")))
    assert not _router_skill_attached(_agent())
