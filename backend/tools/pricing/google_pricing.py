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

    # Public list prices (USD per 1M tokens) as of August 2026
    # Source: https://ai.google.dev/pricing (free tier for development, paid for production)
    # Context/modality prices vary; using the base (lowest-tier, text) rate per model.
    _LLM_PRICES = {
        # Gemini 3.7 / 3.6 (promotional rate through 2026-12-31)
        "gemini-3.7-flash":      (0.75,  3.75),
        "gemini-3.6-flash":      (0.75,  3.75),
        # Gemini 3.5
        "gemini-3.5-flash":      (1.50,  9.00),
        "gemini-3.5-flash-lite": (0.30,  2.50),
        # Gemini 3.1
        "gemini-3.1-pro-preview": (2.00, 12.00),
        "gemini-3.1-flash-lite": (0.25,  1.50),
        # Gemini 2.5
        "gemini-2.5-pro":        (1.25, 10.00),
        "gemini-2.5-flash":      (0.30,  2.50),
        "gemini-2.5-flash-lite": (0.10,  0.40),
        # Gemini 2.0 (legacy)
        "gemini-2.0-flash":      (0.075, 0.30),
        # Gemini 1.5 (legacy)
        "gemini-1.5-pro":        (1.25,  5.00),
        "gemini-1.5-flash":      (0.075, 0.30),
        # Gemini 1.0 (legacy)
        "gemini-1.0-pro":        (0.50,  1.50),
    }

    _EMBEDDING_PRICES = {
        "gemini-embedding-001": 0.15,
        "gemini-embedding-2":   0.20,
        "text-embedding-004":   0.025,  # legacy id, kept for existing configs
    }

    def fetch_llm_pricing(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Return known Google Gemini pricing."""
        return self._LLM_PRICES.copy()

    def fetch_embedding_pricing(self) -> Dict[str, float]:
        """Return known Google embedding pricing."""
        return self._EMBEDDING_PRICES.copy()
