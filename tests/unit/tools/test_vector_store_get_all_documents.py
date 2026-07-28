"""Unit tests for VectorStoreInterface.get_all_documents (PGVector + Qdrant).

get_all_documents fetches the full corpus of a collection with no similarity
ranking and no embeddings — it powers the BM25 search method's in-memory index.
All tests are pure-Python — no real DB/Qdrant server, no I/O.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestPGVectorStoreGetAllDocuments:
    """PGVectorStore.get_all_documents runs a direct SELECT scoped to the collection."""

    def _make_store(self):
        from tools.vector_stores.pgvector_store import PGVectorStore

        store = PGVectorStore.__new__(PGVectorStore)
        store.db = MagicMock()
        store.async_engine = None

        self.mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = self.mock_conn
        mock_engine.connect.return_value.__exit__.return_value = False
        store.engine = mock_engine
        return store

    def test_returns_documents_without_filter(self):
        """No filter — every row in the collection is returned as a Document."""
        store = self._make_store()
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("uuid-1", "content one", {"resource_id": 1}),
            ("uuid-2", "content two", {"resource_id": 2}),
        ]

        docs = store.get_all_documents("silo_1")

        self.mock_conn.execute.assert_called_once()
        sql_arg, params = self.mock_conn.execute.call_args.args
        assert "SELECT" in str(sql_arg)
        assert "langchain_pg_embedding" in str(sql_arg)
        assert params["name"] == "silo_1"
        assert "limit" not in params

        assert len(docs) == 2
        assert docs[0].page_content == "content one"
        assert docs[0].metadata["resource_id"] == 1
        assert docs[0].metadata["_id"] == "uuid-1"
        assert docs[1].page_content == "content two"

    def test_applies_metadata_filter(self):
        """A metadata filter is translated to a WHERE fragment via _build_filter_sql."""
        store = self._make_store()
        self.mock_conn.execute.return_value.fetchall.return_value = []

        store.get_all_documents("silo_1", filter_metadata={"resource_id": {"$eq": 7}})

        sql_arg, params = self.mock_conn.execute.call_args.args
        sql = str(sql_arg)
        assert "cmetadata ->>" in sql
        assert 7 in params.values() or "7" in params.values()

    def test_applies_limit(self):
        """A limit adds a LIMIT clause and binds the :limit parameter."""
        store = self._make_store()
        self.mock_conn.execute.return_value.fetchall.return_value = []

        store.get_all_documents("silo_1", limit=10)

        sql_arg, params = self.mock_conn.execute.call_args.args
        assert "LIMIT" in str(sql_arg)
        assert params["limit"] == 10

    def test_empty_collection_returns_empty_list(self):
        """No rows -> empty list, no error."""
        store = self._make_store()
        self.mock_conn.execute.return_value.fetchall.return_value = []

        docs = store.get_all_documents("silo_empty")

        assert docs == []

    def test_error_propagates(self):
        """A DB error is logged and re-raised, matching search_similar_documents semantics."""
        store = self._make_store()
        self.mock_conn.execute.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            store.get_all_documents("silo_1")


class TestQdrantStoreGetAllDocuments:
    """QdrantStore.get_all_documents pages through _iter_filtered_points."""

    def _make_store(self):
        with patch("qdrant_client.QdrantClient"), patch("langchain_qdrant.QdrantVectorStore"):
            from tools.vector_stores.qdrant_store import QdrantStore

            store = QdrantStore.__new__(QdrantStore)
            store.url = "http://localhost:6333"
            store.api_key = None
            store.prefer_grpc = False
            store._QdrantVectorStore = MagicMock()
            store.client = MagicMock()
            return store

    def _make_point(self, point_id, page_content, metadata):
        point = MagicMock()
        point.id = point_id
        point.payload = {"page_content": page_content, "metadata": metadata}
        return point

    def test_returns_documents_without_filter(self):
        """No filter -> scroll_filter=None, all points returned as Documents."""
        store = self._make_store()
        points = [
            self._make_point("id-1", "content one", {"resource_id": 1}),
            self._make_point("id-2", "content two", {"resource_id": 2}),
        ]
        store.client.scroll.return_value = (points, None)

        docs = store.get_all_documents("silo_1")

        store.client.scroll.assert_called_once()
        call_kwargs = store.client.scroll.call_args.kwargs
        assert call_kwargs["collection_name"] == "silo_1"
        assert call_kwargs["scroll_filter"] is None

        assert len(docs) == 2
        assert docs[0].page_content == "content one"
        assert docs[0].metadata["resource_id"] == 1
        assert docs[0].metadata["_id"] == "id-1"

    def test_applies_metadata_filter(self):
        """A metadata filter is translated to a native Qdrant Filter before scrolling."""
        store = self._make_store()
        store.client.scroll.return_value = ([], None)

        store.get_all_documents("silo_1", filter_metadata={"resource_id": {"$eq": 7}})

        call_kwargs = store.client.scroll.call_args.kwargs
        qdrant_filter = call_kwargs["scroll_filter"]
        assert qdrant_filter is not None
        assert any(
            getattr(c, "key", None) == "metadata.resource_id"
            for c in qdrant_filter.must
        )

    def test_applies_limit_across_batches(self):
        """Paging stops as soon as `limit` documents have been collected.

        `_iter_filtered_points` only continues past a batch when that batch is
        full (len == batch_size, default 200) and an offset was returned, so
        the first batch here is a full page to force a second scroll call.
        """
        store = self._make_store()
        batch_1 = [self._make_point(f"id-{i}", f"content {i}", {}) for i in range(200)]
        batch_2 = [self._make_point(f"id-{i}", f"content {i}", {}) for i in range(200, 203)]
        store.client.scroll.side_effect = [
            (batch_1, "offset-1"),
            (batch_2, None),
        ]

        docs = store.get_all_documents("silo_1", limit=202)

        assert len(docs) == 202
        # First batch (200) fully consumed, second batch stops after 2 more.
        assert store.client.scroll.call_count == 2

    def test_empty_collection_returns_empty_list(self):
        """Empty scroll result -> empty list, no error."""
        store = self._make_store()
        store.client.scroll.return_value = ([], None)

        docs = store.get_all_documents("silo_empty")

        assert docs == []
