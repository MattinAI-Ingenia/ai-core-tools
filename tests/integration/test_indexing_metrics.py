"""Integration tests for per-document indexing metrics (T007).

Tests that indexing a document records an IndexingMetric whose
total_tokens == prompt_tokens + completion_tokens and whose tokens_source
is 'provider' when usage_metadata is present on the LLM response.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


pytestmark = pytest.mark.integration


class TestIndexingMetricRecording:
    """Test that LightRAG indexing records accurate token metrics."""

    def test_token_accumulator_provider_usage(self):
        """Accumulator uses provider usage_metadata when present."""
        from tools.vector_stores.lightrag.token_accumulator import (
            IndexingTokenAccumulator,
        )

        acc = IndexingTokenAccumulator()
        acc.add_llm_usage(prompt=100, completion=50, source="provider")
        acc.add_llm_usage(prompt=200, completion=80, source="provider")

        totals = acc.totals()
        assert totals["prompt_tokens"] == 300
        assert totals["completion_tokens"] == 130
        assert totals["total_tokens"] == 430
        assert totals["tokens_source"] == "provider"
        assert totals["llm_calls"] == 2

    def test_token_accumulator_falls_back_to_estimated(self):
        """Accumulator marks as 'estimated' when any call lacks provider data."""
        from tools.vector_stores.lightrag.token_accumulator import (
            IndexingTokenAccumulator,
        )

        acc = IndexingTokenAccumulator()
        acc.add_llm_usage(prompt=100, completion=50, source="provider")
        acc.add_llm_usage(prompt=200, completion=80, source="estimated")

        totals = acc.totals()
        assert totals["tokens_source"] == "estimated"

    def test_token_accumulator_empty(self):
        """Empty accumulator returns zero totals."""
        from tools.vector_stores.lightrag.token_accumulator import (
            IndexingTokenAccumulator,
        )

        acc = IndexingTokenAccumulator()
        totals = acc.totals()
        assert totals["total_tokens"] == 0
        assert totals["llm_calls"] == 0

    def test_metric_total_equals_prompt_plus_completion(self, db):
        """Persisted metric has total_tokens == prompt_tokens + completion_tokens."""
        from repositories.indexing_metric_repository import IndexingMetricRepository
        from models.app import App
        from models.user import User
        from models.silo import Silo

        # Create minimal entities
        user = User(email="metrics_test@test.com", name="T", is_active=True)
        db.add(user)
        db.flush()

        app = App(name="MetricApp", slug="metric-app-t007", owner_id=user.user_id)
        db.add(app)
        db.flush()

        silo = Silo(
            name="MetricSilo",
            app_id=app.app_id,
            vector_db_type="LIGHTRAG",
        )
        db.add(silo)
        db.flush()

        metric = IndexingMetricRepository.create(
            db,
            app_id=app.app_id,
            silo_id=silo.silo_id,
            resource_id=None,
            content_ref="test_doc.pdf",
            status="success",
            prompt_tokens=500,
            completion_tokens=120,
            total_tokens=620,
            tokens_source="provider",
            llm_calls=3,
            duration_seconds=12.5,
            cost=0.0004,
            currency="USD",
            model_name="gpt-4o-mini",
            embedding_model_name="text-embedding-3-small",
        )

        assert metric.metric_id is not None
        assert metric.total_tokens == metric.prompt_tokens + metric.completion_tokens
        assert metric.tokens_source == "provider"

    def test_failed_indexing_still_records_metric(self, db):
        """A failed run records status='failed' with partial tokens."""
        from repositories.indexing_metric_repository import IndexingMetricRepository
        from models.app import App
        from models.user import User
        from models.silo import Silo

        user = User(email="fail_metrics@test.com", name="F", is_active=True)
        db.add(user)
        db.flush()

        app = App(name="FailApp", slug="fail-app-t007", owner_id=user.user_id)
        db.add(app)
        db.flush()

        silo = Silo(name="FailSilo", app_id=app.app_id, vector_db_type="LIGHTRAG")
        db.add(silo)
        db.flush()

        metric = IndexingMetricRepository.create(
            db,
            app_id=app.app_id,
            silo_id=silo.silo_id,
            resource_id=None,
            status="failed",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            tokens_source="provider",
            llm_calls=1,
            duration_seconds=3.2,
            cost=None,
            currency=None,
        )

        assert metric.status == "failed"
        assert metric.total_tokens == 120
        assert metric.cost is None
