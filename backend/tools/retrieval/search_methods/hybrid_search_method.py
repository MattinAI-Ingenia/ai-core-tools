from typing import List

from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_component import SearchMethod
from tools.retrieval.retrieval_context import RetrievalContext
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_WEIGHTS: List[float] = [0.5, 0.5]


class HybridSearchMethod(SearchMethod):
    """Produce a single retriever that fuses dense and BM25 results via RRF."""

    def build(self, ctx: RetrievalContext) -> BaseRetriever:
        from tools.retrieval.fusion import fuse_with_rrf
        from tools.retrieval.search_methods.dense_search_method import DenseSearchMethod
        from tools.retrieval.search_methods.bm25_search_method import BM25SearchMethod

        dense_retriever = DenseSearchMethod().build(ctx)
        bm25_retriever = BM25SearchMethod().build(ctx)

        weights = ctx.params.get("hybrid_weights", DEFAULT_WEIGHTS)

        logger.debug(
            "HybridSearchMethod: fusing dense+bm25 for collection=%s (weights=%s)",
            ctx.collection_name,
            weights,
        )

        return fuse_with_rrf([dense_retriever, bm25_retriever], weights=weights)
