"""
Factory for retrieval strategies (transformers).

Mirrors the SearchMethodFactory / VectorStoreFactory pattern: it maps a strategy
name to a concrete RetrievalTransformer implementation, caching one instance per
strategy.

The 'rerank' (embeddings-based) and 'cross_encoder_rerank' (dedicated
cross-encoder model) strategies are implemented. Additional strategies
(hybrid, multi-query, ...) are reserved as future support and can be moved into
IMPLEMENTED_STRATEGIES once their implementation is added. When no strategy is
selected the retrieval pipeline simply returns the search-method retriever
unchanged, so there is no explicit no-op strategy.
"""

import threading
from typing import Dict, List

from tools.retrieval.retrieval_component import RetrievalTransformer
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategyFactory:

    # Supported retrieval strategies (including future planned support)
    SUPPORTED_STRATEGIES = {
        'rerank': 'Reranking with the silo embeddings',
        'cross_encoder_rerank': 'Reranking with a dedicated cross-encoder model',
        'hybrid': 'Hybrid dense + sparse search (future support)',
        'multi_query': 'Multi-query expansion (future support)',
    }

    # Strategies that are currently implemented and can be selected by users
    IMPLEMENTED_STRATEGIES = ('rerank', 'cross_encoder_rerank')

    _instances: Dict[str, RetrievalTransformer] = {}
    # Guards _instances: a strategy's construction (e.g. cross_encoder_rerank's
    # model load) can take long enough that two concurrent first-uses would
    # otherwise both miss the cache and each load their own copy of the model.
    _lock = threading.Lock()

    @staticmethod
    def get_strategy(strategy_name: str) -> RetrievalTransformer:
        """Return a cached strategy instance for the requested strategy name."""

        if not strategy_name:
            raise ValueError("A retrieval strategy name is required.")

        resolved_name = strategy_name.lower()

        if resolved_name not in StrategyFactory.SUPPORTED_STRATEGIES:
            supported = ', '.join(StrategyFactory.SUPPORTED_STRATEGIES.keys())
            raise ValueError(
                f"Unsupported retrieval strategy: {resolved_name}. Supported strategies: {supported}"
            )

        if resolved_name not in StrategyFactory.IMPLEMENTED_STRATEGIES:
            raise NotImplementedError(
                f"{resolved_name} strategy is planned but not yet implemented. Currently available: "
                f"{', '.join(StrategyFactory.IMPLEMENTED_STRATEGIES)}"
            )

        if resolved_name in StrategyFactory._instances:
            return StrategyFactory._instances[resolved_name]

        with StrategyFactory._lock:
            # Re-check: another thread may have finished constructing this
            # strategy while we were waiting for the lock.
            if resolved_name in StrategyFactory._instances:
                return StrategyFactory._instances[resolved_name]

            logger.info("Initializing retrieval strategy: %s", resolved_name)

            if resolved_name == 'rerank':
                instance = StrategyFactory._create_rerank_strategy()
            elif resolved_name == 'cross_encoder_rerank':
                instance = StrategyFactory._create_cross_encoder_rerank_strategy()
            else:
                # Guard clause for future implementations
                raise NotImplementedError(f"Retrieval strategy {resolved_name} is not implemented yet")

            StrategyFactory._instances[resolved_name] = instance
            return instance

    @staticmethod
    def get_available_strategy_options() -> List[Dict[str, str]]:
        """Expose implemented retrieval strategy choices with human-friendly labels."""

        options: List[Dict[str, str]] = []
        for key in StrategyFactory.IMPLEMENTED_STRATEGIES:
            label = StrategyFactory.SUPPORTED_STRATEGIES.get(key, key)
            options.append({
                'code': key,
                'label': label
            })
        return options

    @staticmethod
    def _create_rerank_strategy() -> RetrievalTransformer:
        """Create a RerankStrategy instance."""

        from tools.retrieval.strategies.rerank_strategy import RerankStrategy

        return RerankStrategy()

    @staticmethod
    def _create_cross_encoder_rerank_strategy() -> RetrievalTransformer:
        """Create a CrossEncoderRerankStrategy instance."""

        from tools.retrieval.strategies.cross_encoder_rerank_strategy import (
            CrossEncoderRerankStrategy,
        )

        return CrossEncoderRerankStrategy()
