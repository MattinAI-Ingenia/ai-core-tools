"""Integration tests for knowledge-graph endpoint guard conditions (T019).

Tests:
- 409 for a non-LightRAG silo
- 403/404 for wrong app ownership
- 503 when the graph service raises RuntimeError (Neo4j unreachable)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch


pytestmark = pytest.mark.integration


class TestSiloGraphGuards:
    """HTTP guard conditions for GET /internal/apps/{app_id}/silos/{silo_id}/graph."""

    def _login(self, client, email: str) -> dict:
        resp = client.post("/internal/auth/dev-login", json={"email": email})
        token = resp.json().get("access_token") or resp.json().get("token")
        return {"Authorization": f"Bearer {token}"}

    def test_non_lightrag_silo_returns_409(self, client, db):
        """Graph endpoint returns 409 when the silo is not a LightRAG silo."""
        from models.user import User
        from models.app import App
        from models.silo import Silo

        user = User(email="guards-pgv@test.com", name="GuardsPGV", is_active=True)
        db.add(user)
        db.flush()

        app = App(name="GuardsPGVApp", slug="guards-pgv", owner_id=user.user_id)
        db.add(app)
        db.flush()

        silo = Silo(name="PGVSilo", app_id=app.app_id, vector_db_type="PGVECTOR")
        db.add(silo)
        db.flush()

        headers = self._login(client, user.email)
        resp = client.get(
            f"/internal/apps/{app.app_id}/silos/{silo.silo_id}/graph",
            headers=headers,
        )
        assert resp.status_code == 409, (
            f"Expected 409 for non-LightRAG silo, got {resp.status_code}"
        )

    def test_wrong_app_ownership_returns_403_or_404(self, client, db):
        """Graph endpoint returns 403 or 404 when the silo belongs to a different app."""
        from models.user import User
        from models.app import App
        from models.silo import Silo

        user = User(email="guards-xapp@test.com", name="GuardsXApp", is_active=True)
        db.add(user)
        db.flush()

        app_a = App(name="GuardsAppA", slug="guards-app-a", owner_id=user.user_id)
        app_b = App(name="GuardsAppB", slug="guards-app-b", owner_id=user.user_id)
        db.add(app_a)
        db.add(app_b)
        db.flush()

        silo_b = Silo(name="SiloB", app_id=app_b.app_id, vector_db_type="LIGHTRAG")
        db.add(silo_b)
        db.flush()

        headers = self._login(client, user.email)
        # Request silo_b via app_a — ownership mismatch
        resp = client.get(
            f"/internal/apps/{app_a.app_id}/silos/{silo_b.silo_id}/graph",
            headers=headers,
        )
        assert resp.status_code in (403, 404), (
            f"Expected 403/404 for cross-app graph access, got {resp.status_code}"
        )

    def test_neo4j_unreachable_returns_503(self, client, db):
        """Graph endpoint returns 503 when the graph service raises RuntimeError (Neo4j down)."""
        from models.user import User
        from models.app import App
        from models.silo import Silo

        user = User(email="guards-503@test.com", name="Guards503", is_active=True)
        db.add(user)
        db.flush()

        app = App(name="Guards503App", slug="guards-503", owner_id=user.user_id)
        db.add(app)
        db.flush()

        silo = Silo(name="LightRAGSilo503", app_id=app.app_id, vector_db_type="LIGHTRAG")
        db.add(silo)
        db.flush()

        headers = self._login(client, user.email)

        with patch(
            "services.silo_graph_service.SiloGraphService.get_silo_graph",
            side_effect=RuntimeError("Neo4j is unreachable"),
        ):
            resp = client.get(
                f"/internal/apps/{app.app_id}/silos/{silo.silo_id}/graph",
                headers=headers,
            )

        assert resp.status_code == 503, (
            f"Expected 503 when Neo4j unreachable, got {resp.status_code}"
        )
        assert "unreachable" in resp.json().get("detail", "").lower()
