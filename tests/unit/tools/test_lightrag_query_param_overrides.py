"""Locks the retriever's QueryParam construction.

``chunk_top_k`` / ``max_total_tokens`` must reach LightRAG when the silo sets
them, and must be *absent* when unset so LightRAG's own defaults (its dataclass
fields read CHUNK_TOP_K / MAX_TOTAL_TOKENS from the environment) still apply.
"""
import pytest

pytest.importorskip("lightrag")

from lightrag.base import QueryParam  # noqa: E402

from backend.tools.vector_stores.lightrag_store import LightRAGRetriever  # noqa: E402


def _retriever(**kwargs):
    return LightRAGRetriever(store=object(), collection_name="silo_1", **kwargs)


def test_unset_overrides_keep_lightrag_defaults():
    param = _retriever(query_mode="hybrid", top_k=30)._query_param()
    defaults = QueryParam()

    assert param.chunk_top_k == defaults.chunk_top_k
    assert param.max_total_tokens == defaults.max_total_tokens
    assert param.top_k == 30
    assert param.only_need_context is True


def test_set_overrides_reach_query_param():
    param = _retriever(
        query_mode="naive", top_k=30, chunk_top_k=60, max_total_tokens=120000
    )._query_param()

    assert param.chunk_top_k == 60
    assert param.max_total_tokens == 120000
    assert param.mode == "naive"
