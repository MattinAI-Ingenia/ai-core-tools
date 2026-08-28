"""
Hybrid (dense + lexical) search method.

Combines the dense (embeddings similarity) and BM25 (lexical) search methods
into a single retriever via LangChain's ``EnsembleRetriever``, which fuses the
two ranked lists using Reciprocal Rank Fusion (RRF) — a document that ranks
well in either list scores well overall, without needing comparable raw score
scales between a cosine-similarity search and a term-frequency search.

This reuses ``DenseSearchMethod``/``BM25SearchMethod`` as-is (same
``search_kwargs``/``params`` contract), so hybrid search inherits their
existing behaviour (search_type for the dense side, bm25_max_docs for the
lexical side) with no duplicated logic.
"""

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
