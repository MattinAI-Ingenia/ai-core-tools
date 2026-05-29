"""Anthropic pricing fetcher.

Anthropic pricing is fetched from their public pricing page or cached list.
"""

from typing import Dict, Tuple, Optional
from .base import PricingProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicPricingProvider(PricingProvider):
    """Fetch Anthropic model pricing from known public list prices."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    # Public list prices (USD per 1M tokens) as of May 2026
    # Source: https://www.anthropic.com/pricing
    _LLM_PRICES = {
        # Claude 4 series
        "claude-opus-4":     (15.00, 75.00),
        "claude-opus-4-7":   (15.00, 75.00),
        "claude-sonnet-4":   (3.00,  15.00),
        "claude-sonnet-4-6": (3.00,  15.00),
        # Claude 3.5 series
        "claude-3.5-sonnet": (3.00,  15.00),
        "claude-3.5-haiku":  (0.80,  4.00),
        # Claude 3 series
        "claude-3-haiku":    (0.25,  1.25),
        "claude-3-opus":     (15.00, 75.00),
        "claude-3-sonnet":   (3.00,  15.00),
    }

    _EMBEDDING_PRICES = {
        # Anthropic doesn't offer embeddings; would be empty
    }

    def fetch_llm_pricing(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Return known Anthropic LLM pricing."""
        return self._LLM_PRICES.copy()

    def fetch_embedding_pricing(self) -> Dict[str, float]:
        """Return known Anthropic embedding pricing (empty)."""
        return self._EMBEDDING_PRICES.copy()
