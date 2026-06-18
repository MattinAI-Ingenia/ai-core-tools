"""Mistral pricing fetcher."""

from typing import Dict, Tuple, Optional
from .base import PricingProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class MistralPricingProvider(PricingProvider):
    """Fetch Mistral model pricing from known public list prices."""

    @property
    def provider_name(self) -> str:
        return "mistral"

    # Public list prices (USD per 1M tokens) as of June 2026
    # Source: https://mistral.ai/pricing/
    _LLM_PRICES = {
        "mistral-large":    (2.00,   6.00),
        "mistral-medium":   (0.40,   1.20),
        "mistral-small":    (0.10,   0.30),
        "mistral-small-4":  (0.10,   0.30),
        "ministral-3-3b":   (0.10,   0.10),
        "ministral-8b":     (0.10,   0.10),
        "pixtral":          (0.10,   0.30),
    }

    _EMBEDDING_PRICES = {
        "mistral-embed": 0.1,  # approximately
    }

    def fetch_llm_pricing(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Return known Mistral LLM pricing."""
        return self._LLM_PRICES.copy()

    def fetch_embedding_pricing(self) -> Dict[str, float]:
        """Return known Mistral embedding pricing."""
        return self._EMBEDDING_PRICES.copy()
