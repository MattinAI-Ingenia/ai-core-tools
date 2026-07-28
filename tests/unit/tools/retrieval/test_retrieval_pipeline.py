"""Unit tests for RetrievalPipeline."""

from unittest.mock import MagicMock, patch

import pytest

from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.retrieval_pipeline import RetrievalPipeline


class TestRetrievalPipeline:

    def test_raises_without_search_methods(self):
        ctx = RetrievalContext()
        with pytest.raises(ValueError, match="at least one search method"):
            RetrievalPipeline.build(ctx, search_method_names=[])

    def test_single_search_method_no_transformer_returns_as_is(self):
        ctx = RetrievalContext()
        mock_retriever = MagicMock(name="dense_retriever")
        mock_search_method = MagicMock()
        mock_search_method.build.return_value = mock_retriever

        with patch(
            "tools.retrieval.retrieval_pipeline.SearchMethodFactory.get_search_method",
            return_value=mock_search_method,
        ) as mock_get_search_method:
            result = RetrievalPipeline.build(ctx, search_method_names=["dense"])

        mock_get_search_method.assert_called_once_with("dense")
        mock_search_method.build.assert_called_once_with(ctx)
        assert result is mock_retriever

    def test_transformer_is_applied_to_search_method_retriever(self):
        ctx = RetrievalContext()
        base_retriever = MagicMock(name="base_retriever")
        wrapped_retriever = MagicMock(name="wrapped_retriever")

        mock_search_method = MagicMock()
        mock_search_method.build.return_value = base_retriever

        mock_transformer = MagicMock()
        mock_transformer.apply.return_value = wrapped_retriever

        with patch(
            "tools.retrieval.retrieval_pipeline.SearchMethodFactory.get_search_method",
            return_value=mock_search_method,
        ), patch(
            "tools.retrieval.retrieval_pipeline.StrategyFactory.get_strategy",
            return_value=mock_transformer,
        ) as mock_get_strategy:
            result = RetrievalPipeline.build(
                ctx, search_method_names=["dense"], transformer_names=["rerank"]
            )

        mock_get_strategy.assert_called_once_with("rerank")
        mock_transformer.apply.assert_called_once_with([base_retriever], ctx)
        assert result is wrapped_retriever

    def test_multiple_transformers_applied_in_order(self):
        ctx = RetrievalContext()
        base_retriever = MagicMock(name="base_retriever")
        after_first = MagicMock(name="after_first")
        after_second = MagicMock(name="after_second")

        mock_search_method = MagicMock()
        mock_search_method.build.return_value = base_retriever

        first_transformer = MagicMock()
        first_transformer.apply.return_value = after_first
        second_transformer = MagicMock()
        second_transformer.apply.return_value = after_second

        with patch(
            "tools.retrieval.retrieval_pipeline.SearchMethodFactory.get_search_method",
            return_value=mock_search_method,
        ), patch(
            "tools.retrieval.retrieval_pipeline.StrategyFactory.get_strategy",
            side_effect=[first_transformer, second_transformer],
        ):
            result = RetrievalPipeline.build(
                ctx,
                search_method_names=["dense"],
                transformer_names=["strategy_a", "strategy_b"],
            )

        first_transformer.apply.assert_called_once_with([base_retriever], ctx)
        second_transformer.apply.assert_called_once_with([after_first], ctx)
        assert result is after_second
