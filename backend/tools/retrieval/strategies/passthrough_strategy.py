"""
Passthrough retrieval strategy.

This is the default, no-op transformer: it returns the single source retriever
unchanged. It exists so the retrieval pipeline can be wired in without altering
current behaviour. Real strategies (reranking, ensemble/hybrid, multi-query,
etc.) are added later as additional implementations.
"""

from typing import List

from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_component import RetrievalTransformer
from tools.retrieval.retrieval_context import RetrievalContext
from utils.logger import get_logger

logger = get_logger(__name__)


class PassthroughStrategy(RetrievalTransformer):
    """Return the source retriever as-is, without any additional wrapping."""

    def apply(
        self,
        retrievers: List[BaseRetriever],
        ctx: RetrievalContext,
    ) -> BaseRetriever:
        logger.debug("PassthroughStrategy: returning source retriever unchanged")
        return retrievers[0]
