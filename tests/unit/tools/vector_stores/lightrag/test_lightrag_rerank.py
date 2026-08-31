"""Pins the LightRAG rerank wiring.

Worth a test of its own because every part of this is a silent no-op when
mis-wired: the embedded LightRAG library never reads its own RERANK_* env vars
(only its standalone API server does), and QueryParam.enable_rerank already
defaults to true — so a missing rerank_model_func produces reranking that is
"enabled" and does nothing. That failure mode is invisible from the outside:
answers just stay mediocre.
"""

from unittest.mock import AsyncMock, patch

import pytest

import config
from tools.vector_stores.lightrag_store import _build_rerank_func


def _configure(monkeypatch, url, model="BAAI/bge-reranker-v2-m3", api_key=None):
    monkeypatch.setattr(config, "LIGHTRAG_RERANK_URL", url)
    monkeypatch.setattr(config, "LIGHTRAG_RERANK_MODEL", model)
    monkeypatch.setattr(config, "LIGHTRAG_RERANK_API_KEY", api_key)


def test_no_endpoint_configured_means_no_rerank_hook(monkeypatch):
    _configure(monkeypatch, None)
    assert _build_rerank_func() is None


def test_endpoint_configured_returns_a_hook(monkeypatch):
    _configure(monkeypatch, "http://reranker:7997/rerank")
    assert callable(_build_rerank_func())


@pytest.mark.asyncio
async def test_hook_forwards_our_config_and_lightrag_s_top_n(monkeypatch):
    """LightRAG calls the hook as ``f(query=, documents=, top_n=)`` and expects
    ``[{"index": int, "relevance_score": float}]`` back (see
    ``apply_rerank_if_enabled`` in lightrag/utils.py). top_n is LightRAG's own
    chunk_top_k, so it must be passed through untouched — swallowing it would
    return every candidate and defeat the cut the rerank exists to inform."""
    _configure(monkeypatch, "http://reranker:7997/rerank")

    # Patched BEFORE building the hook: _build_rerank_func imports
    # generic_rerank_api into its closure at build time, so a patch applied
    # afterwards would leave the closure pointing at the real function — which
    # then makes a real HTTP call (and retries it, tenacity wraps it).
    api = AsyncMock(return_value=[{"index": 1, "relevance_score": 0.9}])
    with patch("lightrag.rerank.generic_rerank_api", api):
        hook = _build_rerank_func()
        result = await hook(query="q", documents=["a", "b"], top_n=30)

    assert result == [{"index": 1, "relevance_score": 0.9}]
    kwargs = api.await_args.kwargs
    assert kwargs["query"] == "q"
    assert kwargs["documents"] == ["a", "b"]
    assert kwargs["top_n"] == 30
    assert kwargs["model"] == "BAAI/bge-reranker-v2-m3"
    assert kwargs["base_url"] == "http://reranker:7997/rerank"
