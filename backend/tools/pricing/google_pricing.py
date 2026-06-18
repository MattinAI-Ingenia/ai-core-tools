"""Google AI pricing fetcher."""

from typing import Dict, Tuple, Optional
from .base import PricingProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class GooglePricingProvider(PricingProvider):
    """Fetch Google AI model pricing from known public list prices."""

    @property
    def provider_name(self) -> str:
        return "google"

    # Public list prices (USD per 1M tokens) as of June 2026
    # Source: https://ai.google.dev/pricing (free tier for development, paid for production)
    # Context window prices vary; using per-token rates for up to 128K tokens.
    _LLM_PRICES = {
        # Gemini 3.1
        "gemini-3.1-flash-lite": (0.25,  1.50),
        # Gemini 2.0
        "gemini-2.0-flash":      (0.075, 0.30),
        # Gemini 1.5
        "gemini-1.5-pro":        (1.25,  5.00),
        "gemini-1.5-flash":      (0.075, 0.30),
        # Gemini 1.0
        "gemini-1.0-pro":        (0.50,  1.50),
    }

    _EMBEDDING_PRICES = {
        "text-embedding-004":  0.025,
    }

    def fetch_llm_pricing(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Return known Google Gemini pricing."""
        return self._LLM_PRICES.copy()

    def fetch_embedding_pricing(self) -> Dict[str, float]:
        """Return known Google embedding pricing."""
        return self._EMBEDDING_PRICES.copy()
