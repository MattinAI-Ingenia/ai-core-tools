"""Router-level unit tests for the self-service "Preferences" view of
cross-session memory (spec `user-memory-mem0`, D-11): list the caller's own
facts in an App and erase one at a time.

- Identity always comes from the session: the `user_id` reaching the service
  is the authenticated one, whatever the request carries.
- VIEWER-level App members can use both endpoints, including a caller whose
  `platform_role` is the default `viewer` (mounted behind the same
  `require_editor_for_writes` gate production applies, mirroring
  `test_global_memory_self_service.py`).
- Non-members are rejected; store failures surface as 503, never as an empty
  list or a silent success.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from db.database import get_db
from models.app_collaborator import CollaborationRole, CollaborationStatus
from routers.internal.auth_utils import get_current_user_oauth, require_editor_for_writes
from routers.internal.global_memory import global_memory_router
from tools.memory.client import StoredFact

from tests.factories import AppCollaboratorFactory, AppFactory, UserFactory, configure_factories

_KEY = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def memory_app():
    app = FastAPI()
    app.include_router(
        global_memory_router,
        prefix="/apps/{app_id}/global-memory",
        dependencies=[Depends(require_editor_for_writes)],
    )
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


def _authenticate_as(memory_app, user):
    async def _dep():
        return SimpleNamespace(identity=SimpleNamespace(id=str(user.user_id), email=user.email))

    memory_app.dependency_overrides[get_current_user_oauth] = _dep


def _viewer_of_new_app(db):
    configure_factories(db)
    owner = UserFactory()
    app_obj = AppFactory(owner=owner)
    viewer = UserFactory()
    AppCollaboratorFactory(app=app_obj, user=viewer, role=CollaborationRole.VIEWER, status=CollaborationStatus.ACCEPTED)
    db.flush()
    return app_obj, viewer, owner


class TestListMyPreferences:
    def test_viewer_gets_their_own_facts(self, memory_app, memory_client, db):
        app_obj, viewer, _ = _viewer_of_new_app(db)
        _authenticate_as(memory_app, viewer)
        fact = StoredFact(key=_KEY, text="Wants responses in English", agent_id=3, created_at="2026-10-02T07:44:41+00:00")

        with patch("routers.internal.global_memory.list_my_global_memory", return_value=[fact]) as mock_list:
            response = memory_client.get(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 200, response.text
        assert response.json() == {
            "facts": [
                {"key": _KEY, "text": "Wants responses in English", "agent_id": 3, "created_at": "2026-10-02T07:44:41+00:00"}
            ]
        }
        args = mock_list.call_args[0]
        assert args[1] == app_obj.app_id
        assert args[2] == viewer.user_id

    def test_a_user_id_in_the_query_never_changes_whose_facts_are_listed(self, memory_app, memory_client, db):
        app_obj, viewer, owner = _viewer_of_new_app(db)
        _authenticate_as(memory_app, viewer)

        with patch("routers.internal.global_memory.list_my_global_memory", return_value=[]) as mock_list:
            response = memory_client.get(
                f"/apps/{app_obj.app_id}/global-memory/me",
                params={"user_id": owner.user_id},
                headers={"X-User-Id": str(owner.user_id)},
            )

        assert response.status_code == 200
        assert mock_list.call_args[0][2] == viewer.user_id

    def test_non_member_is_rejected(self, memory_app, memory_client, db):
        configure_factories(db)
        app_obj = AppFactory(owner=UserFactory())
        outsider = UserFactory()
        db.flush()
        _authenticate_as(memory_app, outsider)

        with patch("routers.internal.global_memory.list_my_global_memory") as mock_list:
            response = memory_client.get(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 403
        mock_list.assert_not_called()

    def test_store_failure_is_503_not_an_empty_list(self, memory_app, memory_client, db):
        app_obj, viewer, _ = _viewer_of_new_app(db)
        _authenticate_as(memory_app, viewer)

        with patch("routers.internal.global_memory.list_my_global_memory", side_effect=RuntimeError("store down")):
            response = memory_client.get(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 503


class TestEraseOneOfMyPreferences:
    def test_default_platform_viewer_can_delete_their_own_fact(self, memory_app, memory_client, db):
        """Goes through `require_editor_for_writes` exactly like production: the
        per-fact DELETE must be exempted the same way as `DELETE .../me`."""
        app_obj, viewer, _ = _viewer_of_new_app(db)
        assert viewer.platform_role == "viewer"
        _authenticate_as(memory_app, viewer)

        with patch("routers.internal.global_memory.delete_my_global_memory_fact", return_value=True) as mock_delete:
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me/facts/{_KEY}")

        assert response.status_code == 204, response.text
        args = mock_delete.call_args[0]
        assert args[1:] == (app_obj.app_id, viewer.user_id, _KEY)

    def test_unknown_key_is_404(self, memory_app, memory_client, db):
        app_obj, viewer, _ = _viewer_of_new_app(db)
        _authenticate_as(memory_app, viewer)

        with patch("routers.internal.global_memory.delete_my_global_memory_fact", return_value=False):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me/facts/{_KEY}")

        assert response.status_code == 404

    def test_ill_shaped_key_is_400(self, memory_app, memory_client, db):
        app_obj, viewer, _ = _viewer_of_new_app(db)
        _authenticate_as(memory_app, viewer)

        with patch(
            "routers.internal.global_memory.delete_my_global_memory_fact",
            side_effect=ValueError("Invalid fact key"),
        ):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me/facts/not-a-key")

        assert response.status_code == 400

    def test_store_failure_is_503(self, memory_app, memory_client, db):
        app_obj, viewer, _ = _viewer_of_new_app(db)
        _authenticate_as(memory_app, viewer)

        with patch(
            "routers.internal.global_memory.delete_my_global_memory_fact",
            side_effect=RuntimeError("store down"),
        ):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me/facts/{_KEY}")

        assert response.status_code == 503

    def test_non_member_is_rejected(self, memory_app, memory_client, db):
        configure_factories(db)
        app_obj = AppFactory(owner=UserFactory())
        outsider = UserFactory(platform_role="editor")
        db.flush()
        _authenticate_as(memory_app, outsider)

        with patch("routers.internal.global_memory.delete_my_global_memory_fact") as mock_delete:
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me/facts/{_KEY}")

        assert response.status_code == 403
        mock_delete.assert_not_called()

    def test_the_exemption_covers_only_delete_on_the_own_fact_path(self, memory_app, memory_client, db):
        """The viewer exemption is for `DELETE .../me/facts/{key}` only: the
        operator endpoint under the same gate still 403s a platform viewer."""
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        platform_viewer_admin = UserFactory()
        AppCollaboratorFactory(
            app=app_obj, user=platform_viewer_admin, role=CollaborationRole.ADMINISTRATOR,
            status=CollaborationStatus.ACCEPTED,
        )
        db.flush()
        _authenticate_as(memory_app, platform_viewer_admin)

        response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{owner.user_id}")

        assert response.status_code == 403
