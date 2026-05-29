"""OpenAI pricing fetcher.

OpenAI doesn't expose pricing via API, so we maintain a curated list
of public list prices. Updated periodically as new models are released.
"""

from typing import Dict, Tuple, Optional
from .base import PricingProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIPricingProvider(PricingProvider):
    """Fetch OpenAI model pricing from known public list prices."""

    @property
    def provider_name(self) -> str:
        return "openai"

    # Public list prices (USD per 1M tokens) as of May 2026
    # Source: https://openai.com/pricing
    _LLM_PRICES = {
        # GPT-4o series
        "gpt-4o":           (2.50,   10.00),
        "gpt-4o-mini":      (0.15,    0.60),
        # GPT-4 Turbo
        "gpt-4-turbo":      (10.00,  30.00),
        # GPT-4
        "gpt-4":            (30.00,  60.00),
        # GPT-3.5 Turbo
        "gpt-3.5-turbo":    (0.50,    1.50),
    }

    _EMBEDDING_PRICES = {
        "text-embedding-3-small":  0.02,
        "text-embedding-3-large":  0.13,
        "text-embedding-ada-002":  0.10,
    }

    def fetch_llm_pricing(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Return known OpenAI LLM pricing."""
        return self._LLM_PRICES.copy()

    def fetch_embedding_pricing(self) -> Dict[str, float]:
        """Return known OpenAI embedding pricing."""
        return self._EMBEDDING_PRICES.copy()
