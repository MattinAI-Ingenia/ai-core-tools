"""Unit tests for tools.retrieval.fusion.fuse_with_rrf."""

from typing import List

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from tools.retrieval.fusion import fuse_with_rrf


class _FakeRetriever(BaseRetriever):
    """Minimal real Runnable so EnsembleRetriever's pydantic validation accepts it."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return []


def _fake_retrievers(n: int) -> List[BaseRetriever]:
    return [_FakeRetriever() for _ in range(n)]


class TestFuseWithRrf:

    def test_raises_without_retrievers(self):
        with pytest.raises(ValueError, match="at least one retriever"):
            fuse_with_rrf([])

    def test_defaults_to_equal_weights(self):
        retrievers = _fake_retrievers(3)

        result = fuse_with_rrf(retrievers)

        assert result.retrievers == retrievers
        assert result.weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_mismatched_weights_length_falls_back_to_equal(self):
        retrievers = _fake_retrievers(2)

        result = fuse_with_rrf(retrievers, weights=[1.0])

        assert result.weights == pytest.approx([0.5, 0.5])

    def test_explicit_matching_weights_are_used(self):
        retrievers = _fake_retrievers(2)

        result = fuse_with_rrf(retrievers, weights=[0.2, 0.8])

        assert result.weights == pytest.approx([0.2, 0.8])
