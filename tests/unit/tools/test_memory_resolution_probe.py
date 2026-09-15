"""Unit tests for the embedding-dimensionality probe timeout in
`tools.memory.resolution` (security-audit HIGH, step_009 follow-up).

Covers only the timeout-bounding behaviour added to `_probe_embedding_dims` /
`probe_embedding_dims_for_config` — the pre-existing cache/mismatch logic is
covered indirectly via `tests/unit/services/test_app_service_validate_memory_ids.py`.
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.memory.resolution as resolution_module
from tools.memory.resolution import _probe_embedding_dims, probe_embedding_dims_for_config


@pytest.fixture(autouse=True)
def _clear_dims_cache():
    resolution_module._DIMS_CACHE.clear()
    yield
    resolution_module._DIMS_CACHE.clear()


class TestProbeEmbeddingDimsTimeout:
    def test_no_timeout_calls_embed_query_directly_unchanged_behaviour(self):
        """The runtime path (`build_memory_clients`) never passes `timeout` — its
        call must remain a direct, unbounded call, exactly as before this fix."""
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 8)

        dims = _probe_embedding_dims(embeddings, embedding_service_id=1)

        assert dims == 8
        embeddings.embed_query.assert_called_once_with("dimension probe")

    def test_fast_call_within_timeout_succeeds(self):
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 4)

        dims = _probe_embedding_dims(embeddings, embedding_service_id=2, timeout=1.0)

        assert dims == 4

    def test_slow_call_past_timeout_returns_none_instead_of_blocking(self):
        embeddings = MagicMock()

        def _slow_embed_query(_text):
            time.sleep(1.0)
            return [0.0] * 4

        embeddings.embed_query = _slow_embed_query

        started = time.monotonic()
        dims = _probe_embedding_dims(embeddings, embedding_service_id=3, timeout=0.05)
        elapsed = time.monotonic() - started

        assert dims is None
        assert elapsed < 0.5, "the probe must return promptly at the timeout, not wait for the slow call"

    def test_probe_embedding_dims_for_config_applies_the_config_timeout(self, monkeypatch):
        """`probe_embedding_dims_for_config` must pass a bounded timeout, not `None`,
        to `_probe_embedding_dims` — otherwise a slow/unreachable endpoint blocks the
        `/internal` App-configuration boundary indefinitely."""
        captured = {}

        def _fake_probe(embeddings, embedding_service_id, timeout=None):
            captured["timeout"] = timeout
            return 16

        monkeypatch.setattr(resolution_module, "_probe_embedding_dims", _fake_probe)
        monkeypatch.setattr(
            resolution_module, "get_embeddings_model", lambda svc: MagicMock()
        )

        dims = probe_embedding_dims_for_config(SimpleNamespace(service_id=1))

        assert dims == 16
        assert captured["timeout"] is not None
        assert captured["timeout"] == resolution_module._CONFIG_PROBE_TIMEOUT_SECONDS
