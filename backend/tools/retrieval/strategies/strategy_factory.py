"""
Factory for retrieval strategies (transformers).

Mirrors the SearchMethodFactory / VectorStoreFactory pattern: it maps a strategy
name to a concrete RetrievalTransformer implementation, caching one instance per
strategy.

Only the 'rerank' strategy is implemented for now. Additional strategies
(hybrid, multi-query, ...) are reserved as future support and can be moved into
IMPLEMENTED_STRATEGIES once their implementation is added. When no strategy is
selected the retrieval pipeline simply returns the search-method retriever
unchanged, so there is no explicit no-op strategy.
"""

from typing import Dict, List

from tools.retrieval.retrieval_component import RetrievalTransformer
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategyFactory:

    # Supported retrieval strategies (including future planned support)
    SUPPORTED_STRATEGIES = {
        'rerank': 'Reranking with the silo embeddings',
        'hybrid': 'Hybrid dense + sparse search (future support)',
        'multi_query': 'Multi-query expansion (future support)',
    }

    # Strategies that are currently implemented and can be selected by users
    IMPLEMENTED_STRATEGIES = ('rerank',)

    _instances: Dict[str, RetrievalTransformer] = {}

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

        logger.info("Initializing retrieval strategy: %s", resolved_name)

        if resolved_name == 'rerank':
            instance = StrategyFactory._create_rerank_strategy()
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
