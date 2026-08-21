"""Regression test: concurrent upserts of the same model must not raise
UniqueViolation, and a partial-value update must not null out existing prices.
"""
from models.pricing_catalog import PricingCatalog
from tools.pricing.cache_manager import PricingCacheManager


def test_upsert_pricing_is_idempotent_and_preserves_unset_fields(db):
    PricingCacheManager._upsert_pricing(
        db, model_name="gpt-5.6-sol", provider="openai",
        input_price_per_1m=5.0, output_price_per_1m=30.0, source="openai_api",
    )
    # second call for the same model_name (simulates a second worker racing in)
    PricingCacheManager._upsert_pricing(
        db, model_name="gpt-5.6-sol", provider="openai",
        input_price_per_1m=5.0, output_price_per_1m=30.0, source="openai_api",
    )
    db.commit()

    row = db.query(PricingCatalog).filter_by(model_name="gpt-5.6-sol").one()
    assert row.input_price_per_1m == 5.0
    assert row.output_price_per_1m == 30.0

    # an embedding-only update for the same row must not wipe the LLM prices
    PricingCacheManager._upsert_pricing(
        db, model_name="gpt-5.6-sol", provider="openai",
        embedding_price_per_1m=1.0, source="openai_api",
    )
    db.commit()

    row = db.query(PricingCatalog).filter_by(model_name="gpt-5.6-sol").one()
    assert row.input_price_per_1m == 5.0
    assert row.embedding_price_per_1m == 1.0
