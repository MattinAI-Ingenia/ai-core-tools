"""Unit tests for _emit_monitoring_log metric filtering."""
import pytest
from unittest.mock import MagicMock


class TestEmitMonitoringLog:
    def _make_handler(self, usage: dict):
        handler = MagicMock()
        handler.usage_metadata = usage
        return handler

    def _run(self, usage: dict, metrics_cfg: dict | None):
        from services.agent_streaming_service import _emit_monitoring_log
        log_calls = []
        monitoring_config = {"metrics": metrics_cfg} if metrics_cfg is not None else None
        _emit_monitoring_log(
            agent_id=1,
            monitoring_handler=self._make_handler(usage),
            monitoring_config=monitoring_config,
            log_fn=log_calls.append,
        )
        return log_calls

    def test_all_metrics_enabled_emits_all(self):
        usage = {"gpt-4": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
        calls = self._run(usage, None)
        assert len(calls) == 1
        line = calls[0]
        assert "input_tokens=100" in line
        assert "output_tokens=50" in line
        assert "total_tokens=150" in line
        assert "models=" in line
        assert "llm_calls=1" in line

    def test_output_tokens_disabled(self):
        usage = {"gpt-4": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
        calls = self._run(usage, {"input_tokens": True, "output_tokens": False, "total_tokens": True, "models": True, "llm_calls": True})
        assert len(calls) == 1
        assert "output_tokens" not in calls[0]
        assert "input_tokens=100" in calls[0]

    def test_no_config_all_on(self):
        """Absent config means all metrics enabled."""
        usage = {"gpt-4": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        calls = self._run(usage, None)
        assert len(calls) == 1
        assert "input_tokens=10" in calls[0]

    def test_all_flags_false_emits_only_prefix(self):
        usage = {"gpt-4": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        calls = self._run(usage, {"input_tokens": False, "output_tokens": False, "total_tokens": False, "models": False, "llm_calls": False})
        # Parts list has only the prefix, so nothing is logged
        assert len(calls) == 0

    def test_subset_metrics(self):
        usage = {"gpt-4": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}}
        calls = self._run(usage, {"models": True, "llm_calls": True, "input_tokens": False, "output_tokens": False, "total_tokens": False})
        assert len(calls) == 1
        assert "models=" in calls[0]
        assert "llm_calls=" in calls[0]
        assert "input_tokens" not in calls[0]
