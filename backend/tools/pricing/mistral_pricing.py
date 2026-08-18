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

    # Public list prices (USD per 1M tokens) as of August 2026
    # Source: https://mistral.ai/pricing/api
    _LLM_PRICES = {
        "mistral-large":     (0.50,  1.50),
        "mistral-medium":    (1.50,  7.50),
        "mistral-small":     (0.15,  0.60),
        "mistral-small-4":   (0.15,  0.60),
        "codestral":         (0.30,  0.90),
        "ministral-3-3b":    (0.10,  0.10),
        "ministral-3-14b":   (0.20,  0.20),
        "ministral-8b":      (0.15,  0.15),
        "glm-5.2":           (1.40,  4.40),
        "pixtral":           (0.10,  0.30),  # not listed on current pricing page; kept from prior catalog
    }

    _EMBEDDING_PRICES = {
        "mistral-embed":   0.10,
        "codestral-embed": 0.15,
    }

    def fetch_llm_pricing(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Return known Mistral LLM pricing."""
        return self._LLM_PRICES.copy()

    def fetch_embedding_pricing(self) -> Dict[str, float]:
        """Return known Mistral embedding pricing."""
        return self._EMBEDDING_PRICES.copy()
