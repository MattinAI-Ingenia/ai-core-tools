"""Router-level unit tests for the operator global-memory erasure endpoint
(spec `user-memory-mem0`, FR-8, step_006).

- AC-31: requires ADMINISTRATOR or higher in the exact App scoped by
  `{app_id}`; an ADMINISTRATOR of a different App (or a lower-role member of
  this one) must be rejected.
- AC-62: the response must never vary with the purge outcome — no
  purge-count oracle. Verified against both an empty and a non-empty
  `PurgeResult`.

Mounts only `global_memory_router`/`me_global_memory_router` on a bare
`FastAPI()` app rather than the full application — the real app's lifespan
also initializes the memory store's connection pool and runs its `setup()`
migrations, which is orthogonal to what this file tests. `get_db` is
overridden to the real (savepoint-isolated) test session, so
`require_min_role`'s App/AppCollaborator role resolution runs for real
against factory-created rows; `get_current_user_oauth` is overridden with a
lightweight duck-typed identity, mirroring the `SimpleNamespace`-based
auth-context convention already used in
`tests/unit/routers/test_collaboration_administrator_can_invite.py`.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.database import get_db
from models.app_collaborator import CollaborationRole, CollaborationStatus
from routers.internal.auth_utils import get_current_user_oauth
from routers.internal.global_memory import global_memory_router, me_global_memory_router
from tools.memory.client import PurgeResult

from tests.factories import AppCollaboratorFactory, AppFactory, UserFactory, configure_factories


@pytest.fixture
def memory_app():
    app = FastAPI()
    app.include_router(global_memory_router, prefix="/apps/{app_id}/global-memory")
    app.include_router(me_global_memory_router)
    return app


@pytest.fixture
def memory_client(memory_app, db):
    async def _fake_db():
        yield db

    memory_app.dependency_overrides[get_db] = _fake_db
    with TestClient(memory_app, raise_server_exceptions=True) as test_client:
        yield test_client
    memory_app.dependency_overrides.pop(get_db, None)
    memory_app.dependency_overrides.pop(get_current_user_oauth, None)


def _authenticate_as(memory_app, user_id: int, email: str = "caller@example.com"):
    async def _dep():
        return SimpleNamespace(identity=SimpleNamespace(id=str(user_id), email=email))

    memory_app.dependency_overrides[get_current_user_oauth] = _dep


def _add_collaborator(db, app_obj, user_obj, role):
    collab = AppCollaboratorFactory(app=app_obj, user=user_obj, role=role, status=CollaborationStatus.ACCEPTED)
    db.flush()
    return collab


class TestOperatorEraseRoleGate:
    def test_administrator_of_this_app_succeeds(self, memory_app, memory_client, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        admin_user = UserFactory()
        target_user = UserFactory()
        _add_collaborator(db, app_obj, admin_user, CollaborationRole.ADMINISTRATOR)
        _authenticate_as(memory_app, admin_user.user_id)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=2, complete=True),
        ) as mock_delete:
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{target_user.user_id}")

        assert response.status_code == 204
        assert response.content == b""
        mock_delete.assert_called_once()
        called_args = mock_delete.call_args[0]
        assert called_args[1] == app_obj.app_id
        assert called_args[2] == target_user.user_id

    def test_owner_of_this_app_succeeds(self, memory_app, memory_client, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        target_user = UserFactory()
        _authenticate_as(memory_app, owner.user_id)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{target_user.user_id}")

        assert response.status_code == 204

    def test_editor_of_this_app_is_rejected(self, memory_app, memory_client, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        editor_user = UserFactory()
        target_user = UserFactory()
        _add_collaborator(db, app_obj, editor_user, CollaborationRole.EDITOR)
        _authenticate_as(memory_app, editor_user.user_id)

        response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{target_user.user_id}")

        assert response.status_code == 403

    def test_administrator_of_a_different_app_is_rejected(self, memory_app, memory_client, db):
        configure_factories(db)
        owner_a = UserFactory()
        app_a = AppFactory(owner=owner_a)
        owner_b = UserFactory()
        app_b = AppFactory(owner=owner_b)
        admin_of_b = UserFactory()
        target_user = UserFactory()
        _add_collaborator(db, app_b, admin_of_b, CollaborationRole.ADMINISTRATOR)
        _authenticate_as(memory_app, admin_of_b.user_id)

        # admin_of_b holds ADMINISTRATOR in app_b but no role at all in app_a.
        response = memory_client.delete(f"/apps/{app_a.app_id}/global-memory/users/{target_user.user_id}")

        assert response.status_code == 403

    def test_non_member_is_rejected(self, memory_app, memory_client, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        outsider = UserFactory()
        target_user = UserFactory()
        db.flush()
        _authenticate_as(memory_app, outsider.user_id)

        response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{target_user.user_id}")

        assert response.status_code == 403


class TestOperatorEraseNoPurgeCountOracle:
    """AC-62 — the response must be identical whether the scope held memory or not."""

    @pytest.mark.parametrize(
        "purge_result",
        [PurgeResult(deleted=0, complete=True), PurgeResult(deleted=41, complete=True)],
        ids=["empty_scope", "non_empty_scope"],
    )
    def test_response_is_identical_regardless_of_purge_count(self, memory_app, memory_client, db, purge_result):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        admin_user = UserFactory()
        target_user = UserFactory()
        _add_collaborator(db, app_obj, admin_user, CollaborationRole.ADMINISTRATOR)
        _authenticate_as(memory_app, admin_user.user_id)

        with patch("routers.internal.global_memory.delete_global_memory", return_value=purge_result):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{target_user.user_id}")

        assert response.status_code == 204
        assert response.content == b""


class TestOperatorEraseIncompleteSurfacesAsFailure:
    """Regression: an interrupted purge (`PurgeResult.complete=False`) must
    never be answered as a `204` success (NFR-5) — but the failure response
    must still carry no information about `result.deleted`, preserving AC-62.
    """

    def test_incomplete_purge_returns_503_not_204(self, memory_app, memory_client, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        admin_user = UserFactory()
        target_user = UserFactory()
        _add_collaborator(db, app_obj, admin_user, CollaborationRole.ADMINISTRATOR)
        _authenticate_as(memory_app, admin_user.user_id)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=1, complete=False),
        ):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{target_user.user_id}")

        assert response.status_code == 503
        assert "deleted" not in response.text
        assert "1" not in response.json().get("detail", "")
