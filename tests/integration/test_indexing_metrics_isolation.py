"""Integration tests for metrics tenant isolation (T008).

Ensures GET metrics for a resource under the wrong app_id/silo_id
returns 403/404 and never leaks another app's metric.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


class TestIndexingMetricsIsolation:
    """Tenant isolation tests for the indexing metrics endpoints."""

    def _create_app_with_silo(self, db, email_prefix: str):
        """Helper: create user → app → silo → resource chain."""
        from models.user import User
        from models.app import App
        from models.silo import Silo
        from models.repository import Repository
        from models.resource import Resource

        user = User(email=f"{email_prefix}@iso-test.com", name=email_prefix, is_active=True)
        db.add(user)
        db.flush()

        app = App(name=f"{email_prefix}-app", slug=f"{email_prefix}-iso", owner_id=user.user_id)
        db.add(app)
        db.flush()

        silo = Silo(name=f"{email_prefix}-silo", app_id=app.app_id, vector_db_type="LIGHTRAG")
        db.add(silo)
        db.flush()

        repo = Repository(name=f"{email_prefix}-repo", silo_id=silo.silo_id, app_id=app.app_id)
        db.add(repo)
        db.flush()

        resource = Resource(
            name="doc.pdf",
            uri="doc.pdf",
            status="indexed",
            repository_id=repo.repository_id,
        )
        db.add(resource)
        db.flush()

        return user, app, silo, resource

    def test_metric_not_visible_across_apps(self, db):
        """A metric created for app A must not be returned for app B."""
        from repositories.indexing_metric_repository import IndexingMetricRepository

        _, app_a, silo_a, resource_a = self._create_app_with_silo(db, "iso-app-a")
        _, app_b, silo_b, resource_b = self._create_app_with_silo(db, "iso-app-b")

        # Create metric for app A / silo A / resource A
        IndexingMetricRepository.create(
            db,
            app_id=app_a.app_id,
            silo_id=silo_a.silo_id,
            resource_id=resource_a.resource_id,
            status="success",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            tokens_source="provider",
            duration_seconds=5.0,
        )

        # Querying with resource_a but scoped to silo_b must return None
        result = IndexingMetricRepository.get_latest_by_resource(
            db,
            resource_id=resource_a.resource_id,
            silo_id=silo_b.silo_id,  # wrong silo → cross-app
        )
        assert result is None, "Metric from silo_a leaked into silo_b query"

    def test_api_returns_404_for_wrong_silo(self, client, db):
        """GET metrics endpoint returns 404 when silo doesn't belong to app."""
        from models.user import User
        from models.app import App
        from models.silo import Silo

        user = User(email="api-iso@test.com", name="iso", is_active=True)
        db.add(user)
        db.flush()

        app = App(name="IsoApp", slug="iso-api-app", owner_id=user.user_id)
        db.add(app)
        db.flush()

        # Create silo belonging to a *different* fake app
        other_app = App(name="OtherApp", slug="iso-other-app", owner_id=user.user_id)
        db.add(other_app)
        db.flush()

        silo_other = Silo(name="OtherSilo", app_id=other_app.app_id, vector_db_type="LIGHTRAG")
        db.add(silo_other)
        db.flush()

        # Use dev login for the user
        login = client.post(
            "/internal/auth/dev-login",
            json={"email": user.email},
        )
        token = login.json().get("access_token") or login.json().get("token")
        headers = {"Authorization": f"Bearer {token}"}

        # Request metrics on silo_other via app — should be 403 or 404
        resp = client.get(
            f"/internal/apps/{app.app_id}/silos/{silo_other.silo_id}/resources/999/indexing-metrics",
            headers=headers,
        )
        assert resp.status_code in (403, 404), (
            f"Expected 403/404 for cross-app silo access, got {resp.status_code}: {resp.text}"
        )

    def test_silo_metrics_list_scoped_to_silo(self, db):
        """list_latest_by_silo only returns metrics for the requested silo."""
        from repositories.indexing_metric_repository import IndexingMetricRepository

        _, app_a, silo_a, res_a = self._create_app_with_silo(db, "list-silo-a")
        _, app_b, silo_b, res_b = self._create_app_with_silo(db, "list-silo-b")

        IndexingMetricRepository.create(
            db,
            app_id=app_a.app_id,
            silo_id=silo_a.silo_id,
            resource_id=res_a.resource_id,
            status="success",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            tokens_source="provider",
            duration_seconds=2.0,
        )

        rows = IndexingMetricRepository.list_latest_by_silo(db, silo_id=silo_b.silo_id)
        silo_ids = {r.silo_id for r in rows}
        assert silo_a.silo_id not in silo_ids, "Metric from silo_a leaked into silo_b listing"
