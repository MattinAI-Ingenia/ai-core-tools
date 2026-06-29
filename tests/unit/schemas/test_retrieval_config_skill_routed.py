"""Unit tests for skill-routed mode in RetrievalConfig."""
import pytest
from pydantic import ValidationError

from schemas.agent_schemas import RetrievalConfig, AgentDetailSchema


def test_skill_routed_is_accepted():
    config = RetrievalConfig(lightrag_query_mode="skill-routed")
    assert config.lightrag_query_mode == "skill-routed"


def test_existing_modes_still_accepted():
    for mode in ("local", "global", "hybrid", "mix", "naive", "bypass"):
        config = RetrievalConfig(lightrag_query_mode=mode)
        assert config.lightrag_query_mode == mode


def test_invalid_mode_raises_validation_error():
    with pytest.raises(ValidationError):
        RetrievalConfig(lightrag_query_mode="unknown")


def test_skill_routed_in_lightrag_query_modes_list():
    modes = AgentDetailSchema.model_fields["lightrag_query_modes"].default
    assert "skill-routed" in modes


def test_skill_routed_is_first_in_modes_list():
    """skill-routed should appear first so the frontend defaults to it."""
    modes = AgentDetailSchema.model_fields["lightrag_query_modes"].default
    assert modes[0] == "skill-routed"
