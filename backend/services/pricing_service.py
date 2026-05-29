"""Service for pricing operations."""

from typing import Optional, Tuple
from sqlalchemy.orm import Session
from tools.pricing import PricingCacheManager
from utils.logger import get_logger

logger = get_logger(__name__)


class PricingService:
    """High-level service for pricing operations."""

    @staticmethod
    def update_pricing_catalog(db: Session, force: bool = False) -> dict:
        """Fetch and cache pricing from all providers.

        Called periodically (e.g., daily via scheduled task) to keep prices up-to-date.

        Args:
            db: Database session
            force: If True, skip TTL checks and force refresh

        Returns:
            Result dict with 'success', 'failed', 'total', 'details' keys
        """
        logger.info("Updating pricing catalog...")
        result = PricingCacheManager.fetch_and_cache_all_pricing(db, force_refresh=force)
        logger.info(f"Pricing catalog updated: {result}")
        return result

    @staticmethod
    def get_llm_pricing(
        db: Session,
        model_name: str,
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Get LLM pricing (input, output) from cache.

        Falls back to None if model not found. The caller should have
        a hardcoded fallback for unknown models.

        Args:
            db: Database session
            model_name: Model identifier (e.g., 'gpt-4o')

        Returns:
            (input_usd_per_1m, output_usd_per_1m) or None
        """
        if not model_name:
            return None

        return PricingCacheManager.get_llm_pricing(db, model_name)

    @staticmethod
    def get_embedding_pricing(
        db: Session,
        model_name: str,
    ) -> Optional[float]:
        """Get embedding model pricing from cache.

        Falls back to None if model not found. The caller should have
        a hardcoded fallback for unknown models.

        Args:
            db: Database session
            model_name: Model identifier (e.g., 'text-embedding-3-small')

        Returns:
            input_usd_per_1m or None
        """
        if not model_name:
            return None

        return PricingCacheManager.get_embedding_pricing(db, model_name)
