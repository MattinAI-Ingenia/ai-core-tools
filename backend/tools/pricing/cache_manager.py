"""Pricing cache manager.

Orchestrates fetching from all providers and storing in the DB.
Provides cached lookups with fallback to hardcoded defaults.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from models.pricing_catalog import PricingCatalog
from utils.logger import get_logger
from .openai_pricing import OpenAIPricingProvider
from .anthropic_pricing import AnthropicPricingProvider
from .mistral_pricing import MistralPricingProvider
from .google_pricing import GooglePricingProvider

logger = get_logger(__name__)


class PricingCacheManager:
    """Manages pricing data in the database with provider-specific fetchers."""

    PROVIDERS = [
        OpenAIPricingProvider(),
        AnthropicPricingProvider(),
        MistralPricingProvider(),
        GooglePricingProvider(),
    ]

    @staticmethod
    def fetch_and_cache_all_pricing(db: Session, force_refresh: bool = False) -> dict:
        """Fetch pricing from all providers and cache in the database.

        Args:
            db: Database session
            force_refresh: If True, skip TTL check and force fetch

        Returns:
            Dict with keys: 'success', 'failed', 'total', 'details'
        """
        logger.info("Starting pricing catalog update...")
        results = {
            'success': 0,
            'failed': 0,
            'total': 0,
            'details': [],
        }

        for provider in PricingCacheManager.PROVIDERS:
            try:
                llm_pricing, emb_pricing = provider.fetch_all()

                # Update LLM pricing
                for model_name, (input_price, output_price) in llm_pricing.items():
                    PricingCacheManager._upsert_pricing(
                        db,
                        model_name=model_name,
                        provider=provider.provider_name,
                        input_price_per_1m=input_price,
                        output_price_per_1m=output_price,
                        embedding_price_per_1m=None,
                        source=f"{provider.provider_name}_api",
                    )
                    results['success'] += 1

                # Update embedding pricing
                for model_name, input_price in emb_pricing.items():
                    PricingCacheManager._upsert_pricing(
                        db,
                        model_name=model_name,
                        provider=provider.provider_name,
                        input_price_per_1m=input_price,
                        output_price_per_1m=None,
                        embedding_price_per_1m=input_price,
                        source=f"{provider.provider_name}_api",
                    )
                    results['success'] += 1

                results['total'] += len(llm_pricing) + len(emb_pricing)
                results['details'].append(
                    f"✓ {provider.provider_name}: "
                    f"{len(llm_pricing)} LLM + {len(emb_pricing)} embedding models"
                )

            except Exception as e:
                results['failed'] += 1
                results['details'].append(f"✗ {provider.provider_name}: {str(e)}")
                logger.error(f"Failed to fetch pricing from {provider.provider_name}: {e}")

        db.commit()
        logger.info(f"Pricing update complete: {results['success']} success, {results['failed']} failed")
        return results

    @staticmethod
    def _upsert_pricing(
        db: Session,
        model_name: str,
        provider: str,
        input_price_per_1m: Optional[float] = None,
        output_price_per_1m: Optional[float] = None,
        embedding_price_per_1m: Optional[float] = None,
        source: str = "unknown",
        currency: str = "USD",
    ) -> None:
        """Insert or update pricing record in the database."""
        existing = db.query(PricingCatalog).filter(
            PricingCatalog.model_name == model_name
        ).first()

        if existing:
            # Update existing record
            existing.input_price_per_1m = input_price_per_1m or existing.input_price_per_1m
            existing.output_price_per_1m = output_price_per_1m or existing.output_price_per_1m
            existing.embedding_price_per_1m = embedding_price_per_1m or existing.embedding_price_per_1m
            existing.last_updated = datetime.utcnow()
            existing.source = source
            existing.currency = currency
        else:
            # Insert new record
            new_record = PricingCatalog(
                model_name=model_name,
                provider=provider,
                input_price_per_1m=input_price_per_1m,
                output_price_per_1m=output_price_per_1m,
                embedding_price_per_1m=embedding_price_per_1m,
                last_updated=datetime.utcnow(),
                source=source,
                currency=currency,
            )
            db.add(new_record)

        db.flush()

    @staticmethod
    def get_llm_pricing(
        db: Session,
        model_name: str,
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Get LLM pricing (input, output) from cache, or None if not found.

        Args:
            db: Database session
            model_name: Model name to look up

        Returns:
            (input_usd_per_1m, output_usd_per_1m) or None
        """
        record = db.query(PricingCatalog).filter(
            PricingCatalog.model_name == model_name
        ).first()

        if record and (record.input_price_per_1m is not None or record.output_price_per_1m is not None):
            return (record.input_price_per_1m, record.output_price_per_1m)

        return None

    @staticmethod
    def get_embedding_pricing(
        db: Session,
        model_name: str,
    ) -> Optional[float]:
        """Get embedding pricing from cache, or None if not found."""
        record = db.query(PricingCatalog).filter(
            PricingCatalog.model_name == model_name
        ).first()

        if record and record.embedding_price_per_1m is not None:
            return record.embedding_price_per_1m

        return None
