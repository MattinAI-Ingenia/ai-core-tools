from typing import List

from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.search_methods.search_method_factory import SearchMethodFactory
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """Build a final retriever from named search methods and an optional rerank stage."""

    @staticmethod
    def build(
        ctx: RetrievalContext,
        search_method_names: List[str],
    ) -> BaseRetriever:
        """Assemble the retrieval pipeline and return the final retriever.

        Args:
            ctx: Shared retrieval context passed to every component.
            search_method_names: Ordered search methods to build (at least one).

        Returns:
            The final composed ``BaseRetriever``.

        Raises:
            ValueError: If no search method is provided.
        """
        if not search_method_names:
            raise ValueError("RetrievalPipeline requires at least one search method.")

        logger.debug(
            "Building retrieval pipeline: search_methods=%s, rerank=%s",
            search_method_names,
            ctx.rerank is not None,
        )

        retrievers: List[BaseRetriever] = [
            SearchMethodFactory.get_search_method(name).build(ctx) for name in search_method_names
        ]

        if len(retrievers) > 1:
            from tools.retrieval.fusion import fuse_with_rrf

            retrievers = [fuse_with_rrf(retrievers, weights=ctx.params.get("hybrid_weights"))]

        retriever = retrievers[0]
        if ctx.rerank is not None:
            from tools.retrieval.rerank_transformer import ServiceRerankTransformer

            retriever = ServiceRerankTransformer().apply([retriever], ctx)

        return retriever
