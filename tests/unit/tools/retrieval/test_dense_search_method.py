"""Unit tests for DenseSearchMethod."""

from unittest.mock import MagicMock

import pytest

from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.search_methods.dense_search_method import DenseSearchMethod


class TestDenseSearchMethod:

    def test_build_calls_get_retriever_with_expected_args(self):
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.get_retriever.return_value = mock_retriever

        embedding_service = MagicMock()
        ctx = RetrievalContext(
            vector_store=mock_vector_store,
            collection_name="silo_1",
            embedding_service=embedding_service,
            search_kwargs={"k": 5},
            params={"search_type": "mmr"},
        )

        result = DenseSearchMethod().build(ctx)

        assert result is mock_retriever
        mock_vector_store.get_retriever.assert_called_once_with(
            "silo_1",
            embedding_service,
            {"k": 5},
            search_type="mmr",
            use_async=True,
        )

    def test_build_defaults_search_type_to_similarity(self):
        mock_vector_store = MagicMock()
        ctx = RetrievalContext(
            vector_store=mock_vector_store,
            collection_name="silo_1",
            search_kwargs={"k": 5},
        )

        DenseSearchMethod().build(ctx)

        _, kwargs = mock_vector_store.get_retriever.call_args
        assert kwargs["search_type"] == "similarity"

    def test_build_respects_use_async_false(self):
        """Sync callers (e.g. the Silo Playground, which calls .invoke() rather than
        .ainvoke()) must get a sync-mode retriever, or invoking it raises "This
        method must be called without async_mode"."""
        mock_vector_store = MagicMock()
        ctx = RetrievalContext(
            vector_store=mock_vector_store,
            collection_name="silo_1",
            search_kwargs={"k": 5},
            params={"use_async": False},
        )

        DenseSearchMethod().build(ctx)

        _, kwargs = mock_vector_store.get_retriever.call_args
        assert kwargs["use_async"] is False

    def test_build_defaults_use_async_to_true(self):
        mock_vector_store = MagicMock()
        ctx = RetrievalContext(vector_store=mock_vector_store, collection_name="silo_1")

        DenseSearchMethod().build(ctx)

        _, kwargs = mock_vector_store.get_retriever.call_args
        assert kwargs["use_async"] is True

    def test_build_raises_without_vector_store(self):
        ctx = RetrievalContext(vector_store=None, collection_name="silo_1")

        with pytest.raises(ValueError, match="requires a vector_store"):
            DenseSearchMethod().build(ctx)
