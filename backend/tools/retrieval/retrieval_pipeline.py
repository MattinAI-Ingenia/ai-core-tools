"""
Retrieval pipeline orchestrator.

Composes a final retriever from two ordered stages:

    search_methods[]  ->  (transformers[] applied in order)  ->  final retriever

* **Search methods** (dense, BM25, sparse, ...) each produce a retriever from
  the corpus. With more than one search method, a combining transformer
  (e.g. ensemble / RRF) is expected to fuse them.
* **Transformers** are strategies (rerank, ensemble, ...) applied
  left to right. Each one collapses the current list of retrievers into a new
  list (usually a single retriever) for the next transformer.

This is the single place that knows how the retrieval components fit together,
so adding a new search method or strategy never touches the call sites.
"""

from typing import List, Optional

from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.search_methods.search_method_factory import SearchMethodFactory
from tools.retrieval.strategies.strategy_factory import StrategyFactory
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """Build a final retriever from named search methods and transformers."""

    @staticmethod
    def build(
        ctx: RetrievalContext,
        search_method_names: List[str],
        transformer_names: Optional[List[str]] = None,
    ) -> BaseRetriever:
        """Assemble the retrieval pipeline and return the final retriever.

        Args:
            ctx: Shared retrieval context passed to every component.
            search_method_names: Ordered search methods to build (at least one).
            transformer_names: Ordered strategies to apply. When empty, the
                single search method retriever is returned as-is.

        Returns:
            The final composed ``BaseRetriever``.

        Raises:
            ValueError: If no search method is provided.
        """
        if not search_method_names:
            raise ValueError("RetrievalPipeline requires at least one search method.")

        logger.debug(
            "Building retrieval pipeline: search_methods=%s, transformers=%s",
            search_method_names,
            transformer_names,
        )

        retrievers: List[BaseRetriever] = [
            SearchMethodFactory.get_search_method(name).build(ctx) for name in search_method_names
        ]

        for name in transformer_names or []:
            transformer = StrategyFactory.get_strategy(name)
            retrievers = [transformer.apply(retrievers, ctx)]

        return retrievers[0]
