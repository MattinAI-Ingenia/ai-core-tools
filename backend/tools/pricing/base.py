"""Base class for provider-specific pricing fetchers."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class PricingProvider(ABC):
    """Abstract base class for fetching pricing from LLM/embedding providers.

    Subclasses implement provider-specific logic to fetch current prices
    from official APIs or web sources.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'openai', 'anthropic')."""
        pass

    @abstractmethod
    def fetch_llm_pricing(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Fetch LLM pricing for this provider.

        Returns:
            Dict mapping model_name → (input_usd_per_1m, output_usd_per_1m).
            Values can be None if not applicable.

        Raises:
            Exception if fetch fails (caller should log and skip).
        """
        pass

    @abstractmethod
    def fetch_embedding_pricing(self) -> Dict[str, float]:
        """Fetch embedding service pricing for this provider.

        Returns:
            Dict mapping model_name → input_usd_per_1m.

        Raises:
            Exception if fetch fails (caller should log and skip).
        """
        pass

    def fetch_all(self) -> Tuple[Dict, Dict]:
        """Fetch both LLM and embedding pricing.

        Returns:
            (llm_pricing, embedding_pricing) dicts. Empty dicts if fetch fails.
        """
        try:
            llm_pricing = self.fetch_llm_pricing()
            logger.info(f"[{self.provider_name}] Fetched {len(llm_pricing)} LLM models")
        except Exception as e:
            logger.warning(f"[{self.provider_name}] Failed to fetch LLM pricing: {e}")
            llm_pricing = {}

        try:
            embedding_pricing = self.fetch_embedding_pricing()
            logger.info(f"[{self.provider_name}] Fetched {len(embedding_pricing)} embedding models")
        except Exception as e:
            logger.warning(f"[{self.provider_name}] Failed to fetch embedding pricing: {e}")
            embedding_pricing = {}

        return llm_pricing, embedding_pricing
