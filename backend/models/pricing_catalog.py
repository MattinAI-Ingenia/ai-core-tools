"""Pricing catalog model for dynamic LLM and embedding service pricing."""

from sqlalchemy import Column, String, Float, DateTime, func
from sqlalchemy.orm import declarative_base
from db.database import Base


class PricingCatalog(Base):
    """Stores pricing information for LLM and embedding models.

    Updated periodically by PricingService from official provider APIs.
    Allows fallback to defaults if fetch fails.
    """
    __tablename__ = 'pricing_catalog'

    # Model identifier (e.g., "gpt-4o", "claude-3.5-sonnet", "text-embedding-3-small")
    model_name: str = Column(String(255), primary_key=True, nullable=False)

    # Provider name (openai, anthropic, mistral, google, custom)
    provider: str = Column(String(50), nullable=False, index=True)

    # Currency code (USD, EUR, etc.)
    currency: str = Column(String(3), nullable=False, default='USD', index=True)

    # LLM input pricing (per 1 million tokens in specified currency)
    input_price_per_1m: float | None = Column(Float, nullable=True)

    # LLM output pricing (per 1 million tokens in specified currency)
    output_price_per_1m: float | None = Column(Float, nullable=True)

    # Embedding pricing (per 1 million tokens in specified currency, input only)
    embedding_price_per_1m: float | None = Column(Float, nullable=True)

    # When this record was last fetched/updated from the provider
    last_updated: 'datetime' = Column(DateTime, nullable=False, server_default=func.now())

    # Source of this pricing data (e.g., "openai_api", "anthropic_docs", "mistral_api")
    source: str = Column(String(100), nullable=False)

    def __repr__(self):
        return (
            f"<PricingCatalog({self.model_name}, {self.provider}, "
            f"in=${self.input_price_usd_per_1m}, out=${self.output_price_usd_per_1m}, "
            f"emb=${self.embedding_price_usd_per_1m})>"
        )
