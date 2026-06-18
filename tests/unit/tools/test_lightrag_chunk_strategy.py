"""Locks the silo chunk-strategy → LightRAG native process-option contract.

Fails if a LightRAG upgrade renames the selector chars or if the store's
mapping drifts from the four native chunking methods.
"""
import pytest

pytest.importorskip("lightrag")

from lightrag.parser.routing import chunk_strategy_key  # noqa: E402

from backend.tools.vector_stores.lightrag_store import _CHUNK_STRATEGY_OPTION  # noqa: E402


# Each strategy name must map to the option char LightRAG routes to that
# strategy's native chunker sub-dict.
EXPECTED = {
    "fixed_token": "fixed_token",
    "recursive_character": "recursive_character",
    "semantic_vector": "semantic_vector",
    "paragraph_semantic": "paragraph_semantic",
}


def test_strategy_options_route_to_native_chunkers():
    for strategy, option in _CHUNK_STRATEGY_OPTION.items():
        assert chunk_strategy_key(option) == EXPECTED[strategy]


def test_unknown_and_legacy_strategy_falls_back_to_fixed():
    # None / legacy "token_window" / garbage all default to fixed-token "F".
    for value in (None, "token_window", "nonsense"):
        assert _CHUNK_STRATEGY_OPTION.get(value, "F") == "F"
