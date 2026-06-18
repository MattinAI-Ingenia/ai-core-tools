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

    # Public list prices (USD per 1M tokens) as of June 2026
    # Source: https://openai.com/api/pricing/
    _LLM_PRICES = {
        # GPT-5.5 series
        "gpt-5.5-pro":      (30.00,  180.00),
        "gpt-5.5":          (5.00,    30.00),
        # GPT-5.4 series
        "gpt-5.4-pro":      (30.00,  180.00),
        "gpt-5.4-nano":     (0.20,    1.25),
        "gpt-5.4-mini":     (0.75,    4.50),
        "gpt-5.4-cyber":    (2.50,   15.00),
        "gpt-5.4":          (2.50,   15.00),
        # GPT-5.3 series
        "gpt-5.3-codex":    (1.75,   14.00),
        # o-series reasoning
        "o3-deep-research":     (5.00,   20.00),
        "o4-mini-deep-research": (1.00,    4.00),
        "o4-mini":           (1.10,    4.40),
        "o3-pro":            (20.00,  80.00),
        "o3-mini":           (1.10,    4.40),
        "o3":                (2.00,    8.00),
        # GPT-4.1 series
        "gpt-4.1-nano":     (0.10,    0.40),
        "gpt-4.1-mini":     (0.40,    1.60),
        "gpt-4.1":          (2.00,    8.00),
        # GPT-4o series
        "gpt-4o-mini":      (0.15,    0.60),
        "gpt-4o":           (2.50,   10.00),
        # Legacy
        "gpt-4-turbo":      (10.00,  30.00),
        "gpt-4":            (30.00,  60.00),
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
