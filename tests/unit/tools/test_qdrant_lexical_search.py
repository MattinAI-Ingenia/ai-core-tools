"""QdrantStore.lexical_search: full-text-filtered candidates ranked with BM25."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.vector_stores.qdrant_store import QdrantStore


def _store():
    store = QdrantStore.__new__(QdrantStore)  # skip the network client set-up
    store.client = MagicMock()
    return store


def _point(pid, text, metadata=None):
    return SimpleNamespace(id=pid, payload={"page_content": text, "metadata": metadata or {}})


def test_ranks_term_matches_and_keeps_metadata():
    store = _store()
    QdrantStore._lexical_indexed_collections.discard("silo_9")
    points = [
        _point(1, "cadena de suministro y logistica", {"resource_id": 2}),
        _point(2, "engrasa la cadena de la bicicleta cada 150 km", {"resource_id": 1}),
    ]
    with patch.object(QdrantStore, "_iter_filtered_points", return_value=iter([points])) as scroll:
        docs = store.lexical_search("silo_9", "cadena bicicleta", 2)

    assert [d.metadata["resource_id"] for d in docs] == [1, 2]
    assert docs[0].metadata["_id"] == "2"
    store.client.create_payload_index.assert_called_once()
    text_filter = scroll.call_args.args[1]
    assert len(text_filter.should) == 2  # one MatchText per distinct term


def test_blank_query_returns_nothing_without_touching_qdrant():
    store = _store()
    assert store.lexical_search("silo_9", "  ¿? ", 5) == []
    store.client.create_payload_index.assert_not_called()
