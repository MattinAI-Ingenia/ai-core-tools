from abc import ABC, abstractmethod
from typing import List

from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_context import RetrievalContext


class RetrievalComponent(ABC):
    """Marker base class shared by every retrieval search method and transformer."""


class SearchMethod(RetrievalComponent):
    """A search method that *produces* a retriever from the corpus."""

    @abstractmethod
    def build(self, ctx: RetrievalContext) -> BaseRetriever:
        """Build a retriever from the corpus using the request context.

        Args:
            ctx: The shared retrieval context (vector store, collection,
                embeddings, search kwargs and component params).

        Returns:
            A ``BaseRetriever`` querying the corpus with this search method.

        Raises:
            Exception: If the retriever cannot be built.
        """
        raise NotImplementedError


class RetrievalTransformer(RetrievalComponent):
    """A strategy that *wraps/combines* one or more retrievers into one."""

    @abstractmethod
    def apply(
        self,
        retrievers: List[BaseRetriever],
        ctx: RetrievalContext,
    ) -> BaseRetriever:
        """Combine/wrap the given retrievers into a single retriever.

        Args:
            retrievers: The retrievers produced by the pipeline's search methods
                (or by a previous transformer). Single-input transformers
                operate on ``retrievers[0]``; combining transformers use all of
                them.
            ctx: The shared retrieval context.

        Returns:
            A single ``BaseRetriever``.

        Raises:
            Exception: If the transformer cannot be applied.
        """
        raise NotImplementedError
