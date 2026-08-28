"""Unit tests for CrossEncoderRerankStrategy."""

from unittest.mock import patch

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_context import RetrievalContext


class _FakeRetriever(BaseRetriever):
    """Minimal BaseRetriever — ContextualCompressionRetriever validates
    ``base_retriever`` as a ``Runnable`` instance, so a plain MagicMock fails
    pydantic validation."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        return [Document(page_content="stub")]


class _FakeCrossEncoder(BaseCrossEncoder):
    """Minimal BaseCrossEncoder — CrossEncoderReranker validates the ``model``
    field as a ``BaseCrossEncoder`` instance, so a plain MagicMock fails
    pydantic validation."""

    def score(self, text_pairs):
        return [0.0 for _ in text_pairs]


class TestCrossEncoderRerankStrategy:

    def _make_strategy(self):
        # Skip the real HuggingFaceCrossEncoder load (downloads/loads a model)
        # by patching it out during __init__.
        with patch(
            "tools.retrieval.strategies.cross_encoder_rerank_strategy.HuggingFaceCrossEncoder"
        ) as mock_cls:
            mock_cls.return_value = _FakeCrossEncoder()
            from tools.retrieval.strategies.cross_encoder_rerank_strategy import (
                CrossEncoderRerankStrategy,
            )

            strategy = CrossEncoderRerankStrategy()
        return strategy, mock_cls

    def test_init_loads_configured_model_on_configured_device(self):
        import config

        strategy, mock_cls = self._make_strategy()

        mock_cls.assert_called_once_with(
            model_name=config.CROSS_ENCODER_RERANK_MODEL,
            model_kwargs={"device": config.CROSS_ENCODER_RERANK_DEVICE},
        )
        assert strategy._cross_encoder is mock_cls.return_value

    def test_apply_wraps_base_retriever_in_contextual_compression_retriever(self):
        strategy, _ = self._make_strategy()
        base_retriever = _FakeRetriever()

        ctx = RetrievalContext(params={"top_n": 8})

        result = strategy.apply([base_retriever], ctx)

        assert isinstance(result, ContextualCompressionRetriever)
        assert result.base_retriever is base_retriever
        assert isinstance(result.base_compressor, CrossEncoderReranker)
        assert result.base_compressor.top_n == 8
        assert result.base_compressor.model is strategy._cross_encoder

    def test_apply_uses_default_top_n_when_not_provided(self):
        strategy, _ = self._make_strategy()

        ctx = RetrievalContext()

        result = strategy.apply([_FakeRetriever()], ctx)

        assert result.base_compressor.top_n == 5

    def test_apply_does_not_require_embedding_service(self):
        strategy, _ = self._make_strategy()

        ctx = RetrievalContext(embedding_service=None)

        result = strategy.apply([_FakeRetriever()], ctx)

        assert isinstance(result, ContextualCompressionRetriever)
