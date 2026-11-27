"""Unit tests for RetrievalPipeline."""

from unittest.mock import MagicMock, patch

import pytest

from tools.retrieval.rerank_transformer import RerankPlan, RerankRetriever
from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.retrieval_pipeline import RetrievalPipeline

_GET = "tools.retrieval.retrieval_pipeline.SearchMethodFactory.get_search_method"


def _method(retriever):
    m = MagicMock()
    m.build.return_value = retriever
    return m


class TestRetrievalPipeline:

    def test_raises_without_search_methods(self):
        with pytest.raises(ValueError, match="at least one search method"):
            RetrievalPipeline.build(RetrievalContext(), search_method_names=[])

    def test_single_search_method_without_rerank_returns_as_is(self):
        ctx = RetrievalContext()
        retriever = MagicMock(name="dense_retriever")
        method = _method(retriever)
        with patch(_GET, return_value=method) as get:
            result = RetrievalPipeline.build(ctx, ["dense"])
        get.assert_called_once_with("dense")
        method.build.assert_called_once_with(ctx)
        assert result is retriever

    def test_single_search_method_never_calls_fusion(self):
        retriever = MagicMock(name="dense_retriever")
        with patch(_GET, return_value=_method(retriever)), patch("tools.retrieval.fusion.fuse_with_rrf") as fuse:
            result = RetrievalPipeline.build(RetrievalContext(), ["dense"])
        fuse.assert_not_called()
        assert result is retriever

    def test_rerank_plan_wraps_search_method_retriever(self):
        plan = RerankPlan(top_n=3, candidate_limit=10, skip_reason="not_configured")
        ctx = RetrievalContext(rerank=plan)
        base = MagicMock(name="base", spec=["invoke"])
        with patch(_GET, return_value=_method(base)), \
             patch("tools.retrieval.rerank_transformer.ServiceRerankTransformer.apply") as apply:
            result = RetrievalPipeline.build(ctx, ["dense"])
        apply.assert_called_once_with([base], ctx)
        assert result is apply.return_value

    def test_multiple_search_methods_are_fused_before_rerank(self):
        dense, bm25, fused = MagicMock(), MagicMock(), MagicMock(name="fused")
        ctx = RetrievalContext(rerank=RerankPlan(top_n=3, candidate_limit=None))
        with patch(_GET, side_effect=[_method(dense), _method(bm25)]), \
             patch("tools.retrieval.fusion.fuse_with_rrf", return_value=fused) as fuse, \
             patch("tools.retrieval.rerank_transformer.ServiceRerankTransformer.apply") as apply:
            result = RetrievalPipeline.build(ctx, ["dense", "bm25"])
        fuse.assert_called_once_with([dense, bm25], weights=None)
        apply.assert_called_once_with([fused], ctx)
        assert result is apply.return_value

    def test_rerank_retriever_type_is_returned_for_real_transformer(self):
        from langchain_core.retrievers import BaseRetriever

        class _R(BaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager):
                return []

        ctx = RetrievalContext(rerank=RerankPlan(top_n=2, candidate_limit=None))
        with patch(_GET, return_value=_method(_R())):
            assert isinstance(RetrievalPipeline.build(ctx, ["dense"]), RerankRetriever)
