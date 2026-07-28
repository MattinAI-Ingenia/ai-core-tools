"""Unit tests for RerankStrategy."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.strategies.rerank_strategy import RerankStrategy


class _FakeRetriever(BaseRetriever):
    """Minimal BaseRetriever — ContextualCompressionRetriever validates
    ``base_retriever`` as a ``Runnable`` instance, so a plain MagicMock fails
    pydantic validation."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        return [Document(page_content="stub")]


class _FakeEmbeddings(Embeddings):
    """Minimal Embeddings implementation — EmbeddingsFilter validates the
    ``embeddings`` field as an ``Embeddings`` instance, so a plain MagicMock
    fails pydantic validation."""

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class TestRerankStrategy:

    def test_apply_wraps_base_retriever_in_contextual_compression_retriever(self):
        mock_embeddings = _FakeEmbeddings()
        base_retriever = _FakeRetriever()
        embedding_service = MagicMock()

        ctx = RetrievalContext(
            embedding_service=embedding_service,
            params={"top_n": 8, "similarity_threshold": 0.42},
        )

        with patch(
            "tools.retrieval.strategies.rerank_strategy.get_embeddings_model",
            return_value=mock_embeddings,
        ) as mock_get_embeddings:
            result = RerankStrategy().apply([base_retriever], ctx)

        mock_get_embeddings.assert_called_once_with(embedding_service)
        assert isinstance(result, ContextualCompressionRetriever)
        assert result.base_retriever is base_retriever
        assert isinstance(result.base_compressor, EmbeddingsFilter)
        assert result.base_compressor.k == 8
        assert result.base_compressor.similarity_threshold == 0.42
        assert result.base_compressor.embeddings is mock_embeddings

    def test_apply_uses_default_top_n_when_not_provided(self):
        embedding_service = MagicMock()
        ctx = RetrievalContext(embedding_service=embedding_service)

        with patch(
            "tools.retrieval.strategies.rerank_strategy.get_embeddings_model",
            return_value=_FakeEmbeddings(),
        ):
            result = RerankStrategy().apply([_FakeRetriever()], ctx)

        assert result.base_compressor.k == 5
        assert result.base_compressor.similarity_threshold is None

    def test_apply_raises_without_embedding_service(self):
        ctx = RetrievalContext(embedding_service=None)

        with pytest.raises(ValueError, match="requires the silo's embedding_service"):
            RerankStrategy().apply([_FakeRetriever()], ctx)
