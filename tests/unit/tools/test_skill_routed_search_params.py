"""Regression: skill-routed agents must get their resolved RAG config.

_resolve_and_build_retriever_tool used to `return` the skill-routed tool
*before* calling resolve_search_params — the only place that maps
Agent.rag_k -> k and Agent.rag_chunk_top_k -> lightrag_chunk_top_k. The dynamic
tool therefore received the raw caller params (None on the agent-execution
path) and every per-agent RAG setting was silently dropped, falling back to
deployment-wide env defaults.

It was invisible from outside: the UI and the eval harness both read the stored
DB values, so both reported a configuration that never ran. An A/B of
rag_chunk_top_k 30 vs 60 executed at 30 both times and "proved" the change had
no effect.
"""

from types import SimpleNamespace
from unittest.mock import patch

from tools.agentTools import (
    LIGHTRAG_ROUTER_SKILL_NAME,
    _resolve_and_build_retriever_tool,
)


def _lightrag_agent(**overrides):
    silo = SimpleNamespace(
        silo_id=35,
        vector_db_type="LIGHTRAG",
        metadata_definition=None,
    )
    router_skill = SimpleNamespace(skill=SimpleNamespace(name=LIGHTRAG_ROUTER_SKILL_NAME))
    agent = SimpleNamespace(
        app_id=1,
        silo=silo,
        lightrag_query_mode="skill-routed",
        skill_associations=[router_skill],
        rag_k=10,
        rag_chunk_top_k=60,
        rag_k_mode="fixed",
        rag_search_type="similarity",
        rag_score_threshold=None,
        rag_fixed_filters=None,
        rag_max_retrieval_calls=4,
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _captured_search_params(agent, caller_search_params=None):
    seen = {}

    def _fake_tool(silo, search_params=None, offset=None):
        seen["silo"] = silo
        seen["search_params"] = search_params
        return "tool"

    with (
        patch("tools.agentTools._create_dynamic_lightrag_tool", _fake_tool),
        patch("tools.agentTools._create_coverage_tool", return_value="coverage_tool"),
    ):
        result = _resolve_and_build_retriever_tool(agent, caller_search_params)
    assert result == ["tool", "coverage_tool"]
    return seen["search_params"]


def test_agent_rag_settings_reach_the_skill_routed_tool():
    params = _captured_search_params(_lightrag_agent())

    assert params is not None, "skill-routed got no params at all — the original bug"
    assert params["k"] == 10, "Agent.rag_k must reach the retriever"
    assert params["lightrag_chunk_top_k"] == 60, (
        "Agent.rag_chunk_top_k must reach the retriever — this is the value an "
        "A/B experiment changes, and it silently did nothing before"
    )


def test_caller_params_still_win_over_the_agent():
    """Precedence is caller > agent > env default; resolving earlier must not
    invert it."""
    params = _captured_search_params(
        _lightrag_agent(), {"k": 99, "lightrag_chunk_top_k": 5}
    )

    assert params["k"] == 99
    assert params["lightrag_chunk_top_k"] == 5


def test_unset_chunk_top_k_is_omitted_so_lightrag_keeps_its_env_default():
    """None must not be forwarded as an explicit value: LightRAG only falls back
    to its own CHUNK_TOP_K when the key is absent."""
    params = _captured_search_params(_lightrag_agent(rag_chunk_top_k=None))

    assert "lightrag_chunk_top_k" not in params
