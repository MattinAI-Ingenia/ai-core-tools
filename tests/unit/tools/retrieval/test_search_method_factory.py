"""Unit tests for SearchMethodFactory."""

import pytest

from tools.retrieval.search_methods.search_method_factory import SearchMethodFactory


class TestSearchMethodFactory:

    def setup_method(self):
        # Instances are cached at the class level; reset between tests so
        # assertions about "same instance" are meaningful and isolated.
        SearchMethodFactory._instances = {}

    def test_supported_but_not_implemented_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError, match="sparse"):
            SearchMethodFactory.get_search_method("sparse")

    def test_unsupported_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported retrieval search method"):
            SearchMethodFactory.get_search_method("does_not_exist")

    def test_dense_is_default_when_no_name_given(self):
        from tools.retrieval.search_methods.dense_search_method import DenseSearchMethod

        method = SearchMethodFactory.get_search_method(None)
        assert isinstance(method, DenseSearchMethod)

    def test_bm25_returns_bm25_search_method(self):
        from tools.retrieval.search_methods.bm25_search_method import BM25SearchMethod

        method = SearchMethodFactory.get_search_method("bm25")
        assert isinstance(method, BM25SearchMethod)

    def test_name_is_case_insensitive(self):
        from tools.retrieval.search_methods.dense_search_method import DenseSearchMethod

        method = SearchMethodFactory.get_search_method("DENSE")
        assert isinstance(method, DenseSearchMethod)

    def test_instances_are_cached(self):
        first = SearchMethodFactory.get_search_method("dense")
        second = SearchMethodFactory.get_search_method("dense")
        assert first is second

    def test_get_available_search_method_options(self):
        options = SearchMethodFactory.get_available_search_method_options()
        codes = [o["code"] for o in options]
        assert codes == ["dense", "bm25"]
        assert all({"code", "label"} == set(o.keys()) for o in options)
        # Only implemented methods are exposed — "sparse" must not leak.
        assert "sparse" not in codes
