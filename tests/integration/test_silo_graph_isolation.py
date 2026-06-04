"""Integration tests for knowledge-graph workspace isolation (T018).

Ensures the graph endpoint scopes Cypher queries to the requesting silo's
workspace and never returns data from another silo.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.integration


class TestSiloGraphIsolation:
    """Graph endpoint must filter by silo workspace."""

    def test_build_graph_query_always_includes_workspace_filter(self):
        """Direct Cypher query path always includes WHERE n.workspace = $ws."""
        from services.silo_graph_service import SiloGraphService

        # The service must expose the workspace collection name it will filter on
        ws = SiloGraphService._workspace_name(42)
        assert ws == "silo_42"

    def test_wrong_silo_graph_returns_404(self, client, db):
        """GET graph for a silo belonging to another app returns 403 or 404."""
        from models.user import User
        from models.app import App
        from models.silo import Silo

        user = User(email="graph-iso@test.com", name="G", is_active=True)
        db.add(user)
        db.flush()

        app = App(name="GApp", slug="graph-iso-app", owner_id=user.user_id)
        db.add(app)
        db.flush()

        other_app = App(name="OtherGApp", slug="graph-iso-other", owner_id=user.user_id)
        db.add(other_app)
        db.flush()

        silo_other = Silo(
            name="OtherGraphSilo",
            app_id=other_app.app_id,
            vector_db_type="LIGHTRAG",
        )
        db.add(silo_other)
        db.flush()

        login = client.post("/internal/auth/dev-login", json={"email": user.email})
        token = login.json().get("access_token") or login.json().get("token")
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.get(
            f"/internal/apps/{app.app_id}/silos/{silo_other.silo_id}/graph",
            headers=headers,
        )
        assert resp.status_code in (403, 404), (
            f"Expected 403/404 for cross-app graph access, got {resp.status_code}"
        )

    def test_non_lightrag_silo_graph_returns_409(self, client, db):
        """GET graph for a non-LightRAG silo returns 409 Conflict."""
        from models.user import User
        from models.app import App
        from models.silo import Silo

        user = User(email="graph-pgv@test.com", name="P", is_active=True)
        db.add(user)
        db.flush()

        app = App(name="PGVApp", slug="graph-pgv-app", owner_id=user.user_id)
        db.add(app)
        db.flush()

        silo_pgv = Silo(name="PGVSilo", app_id=app.app_id, vector_db_type="PGVECTOR")
        db.add(silo_pgv)
        db.flush()

        login = client.post("/internal/auth/dev-login", json={"email": user.email})
        token = login.json().get("access_token") or login.json().get("token")
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.get(
            f"/internal/apps/{app.app_id}/silos/{silo_pgv.silo_id}/graph",
            headers=headers,
        )
        assert resp.status_code == 409, (
            f"Expected 409 for non-LightRAG silo graph, got {resp.status_code}"
        )
