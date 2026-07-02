"""Unit tests for the LightRAG query mode field on the agent schemas."""
import pytest
from pydantic import ValidationError

from schemas.agent_schemas import CreateUpdateAgentSchema, AgentDetailSchema


def _agent(**kwargs):
    return CreateUpdateAgentSchema(name="test-agent", **kwargs)


def test_skill_routed_is_accepted():
    agent = _agent(lightrag_query_mode="skill-routed")
    assert agent.lightrag_query_mode == "skill-routed"


def test_existing_modes_still_accepted():
    for mode in ("local", "global", "hybrid", "mix", "naive", "bypass"):
        agent = _agent(lightrag_query_mode=mode)
        assert agent.lightrag_query_mode == mode


def test_none_is_accepted():
    """Non-LightRAG agents leave the mode unset."""
    assert _agent().lightrag_query_mode is None


def test_invalid_mode_raises_validation_error():
    with pytest.raises(ValidationError):
        _agent(lightrag_query_mode="unknown")


def test_skill_routed_in_lightrag_query_modes_list():
    modes = AgentDetailSchema.model_fields["lightrag_query_modes"].default
    assert "skill-routed" in modes


def test_skill_routed_is_first_in_modes_list():
    """skill-routed should appear first so the frontend defaults to it."""
    modes = AgentDetailSchema.model_fields["lightrag_query_modes"].default
    assert modes[0] == "skill-routed"
