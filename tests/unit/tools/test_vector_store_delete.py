"""Unit tests for vector store delete_documents with metadata filters.

Verifies that:
  - QdrantStore uses a native filter-based delete (FilterSelector) rather than
    scroll + delete-by-ids, so collections with >1000 chunks are fully cleared.
  - PGVectorStore iterates in batches until similarity_search returns empty,
    covering resources with more than 1000 chunks.
  - PGVectorStore logs a WARNING and stops at the 100-iteration safety cap.
"""

from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# QdrantStore — native filter-based delete
# ---------------------------------------------------------------------------


class TestQdrantStoreDeleteByFilter:
    """QdrantStore.delete_documents with a metadata-filter dict."""

    def _make_store(self):
        """Return a QdrantStore with a mocked QdrantClient and QdrantVectorStore."""
        with patch("qdrant_client.QdrantClient"), patch("langchain_qdrant.QdrantVectorStore"):
            from tools.vector_stores.qdrant_store import QdrantStore

            store = QdrantStore.__new__(QdrantStore)
            store.url = "http://localhost:6333"
            store.api_key = None
            store.prefer_grpc = False
            store._QdrantVectorStore = MagicMock()

            mock_client = MagicMock()
            store.client = mock_client

            return store, mock_client

    def test_delete_uses_filter_selector_not_scroll(self):
        """delete_documents must call client.delete with a FilterSelector, not scroll."""
        store, mock_client = self._make_store()

        pgvector_filter = {"resource_id": {"$eq": 42}}
        store.delete_documents("silo_1", ids=pgvector_filter)

        # scroll must NOT be called for filter-based deletion
        mock_client.scroll.assert_not_called()

        # client.delete must be called exactly once
        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args
        assert call_kwargs.kwargs.get("collection_name") == "silo_1" or call_kwargs.args[0] == "silo_1"

        # The points_selector argument must be a FilterSelector instance
        from qdrant_client.models import FilterSelector

        points_selector = (
            call_kwargs.kwargs.get("points_selector")
            or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        )
        assert isinstance(points_selector, FilterSelector), (
            f"Expected FilterSelector, got {type(points_selector)}"
        )

    def test_delete_by_id_list_does_not_use_filter_selector(self):
        """Direct list deletion must bypass the filter path (regression guard)."""
        store, mock_client = self._make_store()

        mock_vector_store = MagicMock()

        with patch.object(store, "_get_vector_store", return_value=mock_vector_store):
            store.delete_documents("silo_1", ids=["id-1", "id-2"])

        # client.delete (native) must NOT be called — deletion goes through the
        # LangChain wrapper's delete method instead.
        mock_client.delete.assert_not_called()
        mock_vector_store.delete.assert_called_once_with(ids=["id-1", "id-2"])

    def test_delete_skips_on_none_filter(self):
        """None filter must not trigger any client call."""
        store, mock_client = self._make_store()

        store.delete_documents("silo_1", ids=None)

        mock_client.delete.assert_not_called()
        mock_client.scroll.assert_not_called()


# ---------------------------------------------------------------------------
# PGVectorStore — iterative batch delete
# ---------------------------------------------------------------------------


class TestPGVectorStoreDeleteByFilter:
    """PGVectorStore.delete_documents with a metadata-filter dict."""

    def _make_store(self):
        """Return a PGVectorStore backed by a mock engine."""
        from tools.vector_stores.pgvector_store import PGVectorStore

        store = PGVectorStore.__new__(PGVectorStore)
        mock_db = MagicMock()
        mock_db._async_engine = None
        store.db = mock_db
        store.engine = mock_db.engine
        store.async_engine = None
        return store

    def test_delete_single_batch(self):
        """Resources with <=1000 chunks are cleared in one iteration."""
        store = self._make_store()

        doc_ids = [f"doc-{i}" for i in range(50)]
        fake_docs = [MagicMock(id=id_) for id_ in doc_ids]

        mock_vs = MagicMock()
        # First call returns docs; second call (after delete) returns empty
        mock_vs.similarity_search.side_effect = [fake_docs, []]

        with patch.object(store, "_get_vector_store", return_value=mock_vs):
            store.delete_documents("silo_1", ids={"resource_id": {"$eq": 7}})

        assert mock_vs.similarity_search.call_count == 2
        mock_vs.delete.assert_called_once_with(ids=doc_ids)

    def test_delete_iterates_until_empty(self):
        """Resources with >1000 chunks trigger multiple iterations until empty."""
        store = self._make_store()

        # Simulate 2500 chunks across 3 batches: 1000, 1000, 500, then 0
        batch_1 = [MagicMock(id=f"a{i}") for i in range(1000)]
        batch_2 = [MagicMock(id=f"b{i}") for i in range(1000)]
        batch_3 = [MagicMock(id=f"c{i}") for i in range(500)]

        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = [batch_1, batch_2, batch_3, []]

        with patch.object(store, "_get_vector_store", return_value=mock_vs):
            store.delete_documents("silo_1", ids={"resource_id": {"$eq": 99}})

        assert mock_vs.similarity_search.call_count == 4
        assert mock_vs.delete.call_count == 3
        mock_vs.delete.assert_any_call(ids=[d.id for d in batch_1])
        mock_vs.delete.assert_any_call(ids=[d.id for d in batch_2])
        mock_vs.delete.assert_any_call(ids=[d.id for d in batch_3])

    def test_delete_raises_at_safety_cap(self):
        """After 100 iterations without emptying, a RuntimeError is raised.

        This ensures that reindex_resource (which propagates exceptions) will fail-fast
        rather than silently leaving residual chunks that would cause duplicates.
        """
        store = self._make_store()

        # Always return 1000 docs — simulates a store that never empties
        infinite_batch = [MagicMock(id=f"x{i}") for i in range(1000)]
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = infinite_batch

        with patch.object(store, "_get_vector_store", return_value=mock_vs):
            with pytest.raises(RuntimeError, match="could not empty filter"):
                store.delete_documents("silo_1", ids={"resource_id": {"$eq": 1}})

        assert mock_vs.similarity_search.call_count == 100
        assert mock_vs.delete.call_count == 100

    def test_delete_by_id_list_skips_iteration(self):
        """Direct list deletion must not enter the iteration loop."""
        store = self._make_store()

        mock_vs = MagicMock()
        with patch.object(store, "_get_vector_store", return_value=mock_vs):
            store.delete_documents("silo_1", ids=["id-1", "id-2"])

        mock_vs.delete.assert_called_once_with(ids=["id-1", "id-2"])
        mock_vs.similarity_search.assert_not_called()
