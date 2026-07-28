"""Unit tests for StrategyFactory."""

import pytest

from tools.retrieval.strategies.strategy_factory import StrategyFactory


class TestStrategyFactory:

    def setup_method(self):
        # Instances are cached at the class level; reset between tests so
        # assertions about "same instance" are meaningful and isolated.
        StrategyFactory._instances = {}

    def test_supported_but_not_implemented_hybrid_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError, match="hybrid"):
            StrategyFactory.get_strategy("hybrid")

    def test_supported_but_not_implemented_multi_query_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError, match="multi_query"):
            StrategyFactory.get_strategy("multi_query")

    def test_unsupported_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported retrieval strategy"):
            StrategyFactory.get_strategy("does_not_exist")

    def test_empty_name_raises_value_error(self):
        with pytest.raises(ValueError, match="strategy name is required"):
            StrategyFactory.get_strategy("")

    def test_rerank_returns_rerank_strategy(self):
        from tools.retrieval.strategies.rerank_strategy import RerankStrategy

        strategy = StrategyFactory.get_strategy("rerank")
        assert isinstance(strategy, RerankStrategy)

    def test_name_is_case_insensitive(self):
        from tools.retrieval.strategies.rerank_strategy import RerankStrategy

        strategy = StrategyFactory.get_strategy("RERANK")
        assert isinstance(strategy, RerankStrategy)

    def test_instances_are_cached(self):
        first = StrategyFactory.get_strategy("rerank")
        second = StrategyFactory.get_strategy("rerank")
        assert first is second

    def test_get_available_strategy_options(self):
        options = StrategyFactory.get_available_strategy_options()
        codes = [o["code"] for o in options]
        assert codes == ["rerank"]
        assert all({"code", "label"} == set(o.keys()) for o in options)
        # Only implemented strategies are exposed — "hybrid"/"multi_query" must not leak.
        assert "hybrid" not in codes
        assert "multi_query" not in codes
