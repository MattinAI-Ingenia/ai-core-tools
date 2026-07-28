"""Unit tests for BM25SearchMethod."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.search_methods.bm25_search_method import BM25SearchMethod, _preprocess


class TestBM25SearchMethod:

    def test_build_returns_bm25_retriever_with_expected_k(self):
        from langchain_community.retrievers import BM25Retriever

        mock_vector_store = MagicMock()
        mock_vector_store.get_all_documents.return_value = [
            Document(page_content="the cat sat on the mat"),
            Document(page_content="dogs are great pets"),
        ]

        ctx = RetrievalContext(
            vector_store=mock_vector_store,
            collection_name="silo_1",
            search_kwargs={"k": 3, "filter": {"resource_id": {"$eq": 7}}},
        )

        retriever = BM25SearchMethod().build(ctx)

        assert isinstance(retriever, BM25Retriever)
        assert retriever.k == 3
        mock_vector_store.get_all_documents.assert_called_once_with(
            "silo_1",
            filter_metadata={"resource_id": {"$eq": 7}},
            limit=5000,
        )

    def test_build_respects_bm25_max_docs_param(self):
        mock_vector_store = MagicMock()
        mock_vector_store.get_all_documents.return_value = [
            Document(page_content="some content"),
        ]

        ctx = RetrievalContext(
            vector_store=mock_vector_store,
            collection_name="silo_1",
            params={"bm25_max_docs": 100},
        )

        BM25SearchMethod().build(ctx)

        mock_vector_store.get_all_documents.assert_called_once_with(
            "silo_1",
            filter_metadata=None,
            limit=100,
        )

    def test_build_raises_on_empty_corpus(self):
        mock_vector_store = MagicMock()
        mock_vector_store.get_all_documents.return_value = []

        ctx = RetrievalContext(vector_store=mock_vector_store, collection_name="silo_1")

        with pytest.raises(ValueError, match="no documents to index"):
            BM25SearchMethod().build(ctx)

    def test_build_raises_without_vector_store(self):
        ctx = RetrievalContext(vector_store=None, collection_name="silo_1")

        with pytest.raises(ValueError, match="requires a vector_store"):
            BM25SearchMethod().build(ctx)

    def test_build_raises_without_collection_name(self):
        ctx = RetrievalContext(vector_store=MagicMock(), collection_name=None)

        with pytest.raises(ValueError, match="requires a collection_name"):
            BM25SearchMethod().build(ctx)

    def test_build_passes_custom_preprocess_func(self):
        mock_vector_store = MagicMock()
        mock_vector_store.get_all_documents.return_value = [Document(page_content="some content")]

        ctx = RetrievalContext(vector_store=mock_vector_store, collection_name="silo_1")

        retriever = BM25SearchMethod().build(ctx)

        assert retriever.preprocess_func is _preprocess

    def test_singular_query_matches_document_with_plural_and_punctuation(self):
        """Regression: bare BM25 tokenizes "plans?" as one token, distinct from the
        query "plan" — no stem/lemma match. _preprocess must close that gap."""
        # A handful of unrelated filler docs — with only 2 total documents, BM25's
        # IDF term for a word appearing in exactly 1 of them is log(1)=0 (a corpus-size
        # artifact of the classic Robertson-Sparck Jones formula, not a real-world case).
        mock_vector_store = MagicMock()
        mock_vector_store.get_all_documents.return_value = [
            Document(page_content="What are you doing tomorrow? Have you got any plans?"),
            Document(page_content="Dogs are great pets."),
            Document(page_content="The weather today is sunny and warm."),
            Document(page_content="I like to read books on the weekend."),
            Document(page_content="She plays the piano every evening."),
        ]
        ctx = RetrievalContext(vector_store=mock_vector_store, collection_name="silo_1", search_kwargs={"k": 1})

        retriever = BM25SearchMethod().build(ctx)
        results = retriever.invoke("plan")

        assert len(results) == 1
        assert "plans?" in results[0].page_content


class TestBM25Preprocess:

    def test_lowercases_and_strips_punctuation(self):
        assert _preprocess("Have you got any plans?") == _preprocess("have you got any plans")

    def test_stems_plural_to_match_singular_query(self):
        assert "plan" in _preprocess("Have you got any plans?")
        assert _preprocess("plan") == ["plan"]

    def test_stems_verb_conjugation(self):
        assert _preprocess("running") == _preprocess("run")
