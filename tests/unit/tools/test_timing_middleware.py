"""The [timing] line must carry the turn's own token usage (and survive a failing call)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.agentTools as agent_tools
from tools.agentTools import _TimingMiddleware


def _run(mw, handler):
    return asyncio.run(mw.awrap_model_call(None, handler))


def _logged(spy):
    # (format, agent_id, elapsed_ms, input_tokens, output_tokens)
    return spy.info.call_args.args


def test_logs_input_and_output_tokens_of_the_turn(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(agent_tools, "logger", spy)
    msg = SimpleNamespace(usage_metadata={"input_tokens": 1234, "output_tokens": 56})

    async def handler(_):
        return SimpleNamespace(result=[msg])

    _run(_TimingMiddleware(3), handler)

    assert "[timing] llm_turn" in _logged(spy)[0]
    assert _logged(spy)[3:] == (1234, 56)


def test_a_failing_turn_is_still_logged_without_tokens(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(agent_tools, "logger", spy)

    async def handler(_):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _run(_TimingMiddleware(3), handler)

    assert _logged(spy)[3:] == (None, None)
