"""
Shared Reciprocal Rank Fusion (RRF) helper.

Wraps LangChain's ``EnsembleRetriever`` so both ``HybridSearchMethod`` (fixed
dense+bm25 pair) and ``RetrievalPipeline`` (arbitrary N selected search
methods) fuse ranked lists the same way, without duplicating the
construction logic.
"""

from typing import List, Optional

from langchain_core.retrievers import BaseRetriever

from utils.logger import get_logger

logger = get_logger(__name__)


def fuse_with_rrf(retrievers: List[BaseRetriever], weights: Optional[List[float]] = None) -> BaseRetriever:
    """Fuse multiple retrievers into one via Reciprocal Rank Fusion.

    Args:
        retrievers: The retrievers to fuse (at least one).
        weights: Per-retriever weights, same length as ``retrievers``. Falls
            back to equal weighting when omitted or mismatched in length.

    Returns:
        A single ``BaseRetriever`` (``EnsembleRetriever``) fusing the inputs.

    Raises:
        ValueError: If ``retrievers`` is empty.
    """
    if not retrievers:
        raise ValueError("fuse_with_rrf requires at least one retriever.")

    if not weights or len(weights) != len(retrievers):
        weights = [1.0 / len(retrievers)] * len(retrievers)

    from langchain_classic.retrievers.ensemble import EnsembleRetriever

    logger.debug("Fusing %d retriever(s) via RRF (weights=%s)", len(retrievers), weights)

    return EnsembleRetriever(retrievers=retrievers, weights=weights)
