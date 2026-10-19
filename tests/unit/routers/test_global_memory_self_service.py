"""Router-level unit tests for the self-service global-memory erasure
endpoints (spec `user-memory-mem0`, FR-8, D-11, step_006).

- AC-44: `/apps/{app_id}/global-memory/me` purges the caller's own scope,
  returns the count, is idempotent, and succeeds for a VIEWER-role caller.
- AC-45: with user X authenticated, attempts to target user Y via the
  request body, a query parameter, an extra path segment, and a spoofed
  header either are rejected outright or purge only X's own scope — the
  `user_id` reaching `delete_global_memory` / `delete_user_global_memory`
  must always be the one from the authenticated session, never from request
  input. Both the App-scoped (`/apps/{app_id}/global-memory/me`) and
  cross-App (`/me/global-memory`) variants are covered, since both declare
  the same "identity from session only" rule.

Kept in its own file (not folded into generic endpoint tests) so the
security auditor gating this step has a single place to look, per the plan.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from db.database import get_db
from models.app_collaborator import CollaborationRole, CollaborationStatus
from routers.internal.auth_utils import get_current_user_oauth, require_editor_for_writes
from routers.internal.global_memory import global_memory_router, me_global_memory_router
from tools.memory.client import PurgeResult

from tests.factories import AppCollaboratorFactory, AppFactory, UserFactory, configure_factories


@pytest.fixture
def memory_app():
    app = FastAPI()
    # `dependencies=[Depends(require_editor_for_writes)]` mirrors EXACTLY how
    # production mounts this router (`apps_router.include_router(...,
    # dependencies=[Depends(require_editor_for_writes)])` in
    # `backend/routers/internal/__init__.py`). Security-audit regression
    # (MEDIUM 1, spec `user-memory-mem0`): this fixture used to omit that
    # dependency entirely, so no test in this file could ever observe a
    # viewer-role 403 from it — a test that structurally cannot see a gate
    # can never fail when that gate misbehaves.
    app.include_router(
        global_memory_router,
        prefix="/apps/{app_id}/global-memory",
        dependencies=[Depends(require_editor_for_writes)],
    )
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


def _authenticate_as(memory_app, user):
    """Authenticate as `user` (a persisted `User` row).

    Uses the user's REAL `user_id`/`email` — `require_editor_for_writes`
    looks the caller up by email to read `platform_role` from the DB, so a
    fake/mismatched email would silently skip that check (`user` would come
    back `None`) instead of exercising it.
    """
    async def _dep():
        return SimpleNamespace(identity=SimpleNamespace(id=str(user.user_id), email=user.email))

    memory_app.dependency_overrides[get_current_user_oauth] = _dep


class TestAppScopedSelfServiceErasure:
    """AC-44."""

    def _setup(self, db, role=CollaborationRole.VIEWER):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        caller = UserFactory()
        AppCollaboratorFactory(app=app_obj, user=caller, role=role, status=CollaborationStatus.ACCEPTED)
        db.flush()
        return app_obj, caller

    def test_viewer_can_erase_own_memory_and_gets_the_count(self, memory_app, memory_client, db):
        app_obj, viewer_user = self._setup(db)
        _authenticate_as(memory_app, viewer_user)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=3, complete=True),
        ) as mock_delete:
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 200
        assert response.json() == {"purged": 3}
        mock_delete.assert_called_once_with(db, app_obj.app_id, viewer_user.user_id)

    def test_idempotent_for_a_caller_with_nothing_stored(self, memory_app, memory_client, db):
        app_obj, viewer_user = self._setup(db)
        _authenticate_as(memory_app, viewer_user)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 200
        assert response.json() == {"purged": 0}

    def test_non_member_is_rejected(self, memory_app, memory_client, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        outsider = UserFactory()
        db.flush()
        _authenticate_as(memory_app, outsider)

        response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 403

    def test_incomplete_purge_returns_503_not_a_success(self, memory_app, memory_client, db):
        """An interrupted purge must never be answered as a `200` success
        with a misleadingly clean `purged` count (NFR-5)."""
        app_obj, viewer_user = self._setup(db)
        _authenticate_as(memory_app, viewer_user)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=1, complete=False),
        ):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 503


class TestAppScopedSelfServiceCannotTargetAnotherUser:
    """AC-45 — App-scoped `/me` endpoint."""

    def _setup(self, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        user_x = UserFactory()
        user_y = UserFactory()
        AppCollaboratorFactory(app=app_obj, user=user_x, role=CollaborationRole.VIEWER, status=CollaborationStatus.ACCEPTED)
        AppCollaboratorFactory(app=app_obj, user=user_y, role=CollaborationRole.VIEWER, status=CollaborationStatus.ACCEPTED)
        db.flush()
        return app_obj, user_x, user_y

    def test_user_id_in_body_is_ignored(self, memory_app, memory_client, db):
        app_obj, user_x, user_y = self._setup(db)
        _authenticate_as(memory_app, user_x)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ) as mock_delete:
            response = memory_client.request(
                "DELETE",
                f"/apps/{app_obj.app_id}/global-memory/me",
                json={"user_id": user_y.user_id},
            )

        assert response.status_code == 200
        mock_delete.assert_called_once_with(db, app_obj.app_id, user_x.user_id)

    def test_user_id_in_query_param_is_ignored(self, memory_app, memory_client, db):
        app_obj, user_x, user_y = self._setup(db)
        _authenticate_as(memory_app, user_x)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ) as mock_delete:
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me?user_id={user_y.user_id}")

        assert response.status_code == 200
        mock_delete.assert_called_once_with(db, app_obj.app_id, user_x.user_id)

    def test_user_id_as_extra_path_segment_is_rejected(self, memory_app, memory_client, db):
        app_obj, user_x, user_y = self._setup(db)
        _authenticate_as(memory_app, user_x)

        response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me/{user_y.user_id}")

        assert response.status_code == 404

    def test_spoofed_user_id_header_is_ignored(self, memory_app, memory_client, db):
        app_obj, user_x, user_y = self._setup(db)
        _authenticate_as(memory_app, user_x)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ) as mock_delete:
            response = memory_client.delete(
                f"/apps/{app_obj.app_id}/global-memory/me",
                headers={"X-User-Id": str(user_y.user_id)},
            )

        assert response.status_code == 200
        mock_delete.assert_called_once_with(db, app_obj.app_id, user_x.user_id)

    def test_handler_declares_no_user_id_parameter_at_all(self):
        """Structural half of AC-45: no request-shaped input can ever reach a
        `user_id` the handler doesn't declare, because it declares none."""
        import inspect

        from routers.internal.global_memory import erase_my_global_memory

        params = inspect.signature(erase_my_global_memory).parameters
        assert "user_id" not in params


class TestCrossAppSelfServiceCannotTargetAnotherUser:
    """AC-45 — cross-App `/me/global-memory` endpoint (no App-membership gate at all)."""

    def test_user_id_in_body_query_and_header_are_all_ignored(self, memory_app, memory_client, db):
        configure_factories(db)
        user_x = UserFactory()
        user_y = UserFactory()
        db.flush()
        _authenticate_as(memory_app, user_x)

        with patch(
            "routers.internal.global_memory.delete_user_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ) as mock_delete:
            response = memory_client.request(
                "DELETE",
                f"/me/global-memory?user_id={user_y.user_id}",
                json={"user_id": user_y.user_id},
                headers={"X-User-Id": str(user_y.user_id)},
            )

        assert response.status_code == 200
        mock_delete.assert_called_once_with(db, user_x.user_id, commit=True)

    def test_user_id_as_extra_path_segment_is_rejected(self, memory_app, memory_client, db):
        configure_factories(db)
        user_x = UserFactory()
        user_y = UserFactory()
        db.flush()
        _authenticate_as(memory_app, user_x)

        response = memory_client.delete(f"/me/global-memory/{user_y.user_id}")

        assert response.status_code == 404

    def test_no_app_membership_required(self, memory_app, memory_client, db):
        """This endpoint's whole reason to exist: a user with memory in an
        App they do not belong to must still be able to reach it (R-24)."""
        configure_factories(db)
        lone_user = UserFactory()
        db.flush()
        _authenticate_as(memory_app, lone_user)

        with patch(
            "routers.internal.global_memory.delete_user_global_memory",
            return_value=PurgeResult(deleted=1, complete=True),
        ):
            response = memory_client.delete("/me/global-memory")

        assert response.status_code == 200
        assert response.json() == {"purged": 1}

    def test_handler_declares_no_user_id_parameter_at_all(self):
        import inspect

        from routers.internal.global_memory import erase_my_global_memory_everywhere

        params = inspect.signature(erase_my_global_memory_everywhere).parameters
        assert "user_id" not in params


class TestCrossAppSelfServiceIncompleteSurfacesAsFailure:
    """Regression (step_006 follow-up): a cross-App erasure whose
    enumeration (`list_user_scopes`) failed must not answer as a success —
    `delete_user_global_memory` already folds that into `PurgeResult.complete`;
    this endpoint must in turn never return `200` for `complete=False`."""

    def test_incomplete_purge_returns_503_not_200(self, memory_app, memory_client, db):
        configure_factories(db)
        lone_user = UserFactory()
        db.flush()
        _authenticate_as(memory_app, lone_user)

        with patch(
            "routers.internal.global_memory.delete_user_global_memory",
            return_value=PurgeResult(deleted=0, complete=False),
        ):
            response = memory_client.delete("/me/global-memory")

        assert response.status_code == 503


class TestAppsRouterWriteGateExemptsSelfServiceErasure:
    """Regression — security-audit MEDIUM 1 (spec `user-memory-mem0`).

    `require_editor_for_writes` (mounted on `apps_router` in production,
    `backend/routers/internal/__init__.py:23-27`) 403s ANY mutating verb for
    a caller whose `User.platform_role == 'viewer'` — the DB column default
    (`backend/models/user.py`) and what LOCAL signup assigns. `DELETE
    .../global-memory/me` is a DELETE, so a default-provisioned user hit a
    403 despite AC-44/FR-8(b) explicitly allowing a VIEWER to erase their own
    memory. This class exercises the fix (the `/global-memory/me` exemption
    in `require_editor_for_writes`) through the SAME gate production applies
    (see the `memory_app` fixture above) — a fixture without that gate
    structurally cannot fail on this regression, which is exactly how it
    slipped through originally.
    """

    def test_default_platform_role_viewer_is_not_blocked_by_the_apps_router_gate(
        self, memory_app, memory_client, db
    ):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        viewer_user = UserFactory()
        AppCollaboratorFactory(
            app=app_obj, user=viewer_user, role=CollaborationRole.VIEWER, status=CollaborationStatus.ACCEPTED
        )
        db.flush()
        assert viewer_user.platform_role == "viewer"  # the DB column default — not set explicitly here
        _authenticate_as(memory_app, viewer_user)

        with patch(
            "routers.internal.global_memory.delete_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ):
            response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/me")

        assert response.status_code == 200, response.text

    def test_an_ordinary_write_under_the_same_gate_is_still_blocked_for_a_viewer(
        self, memory_app, memory_client, db
    ):
        """Sanity check: the fix exempts only `.../global-memory/me` — the
        operator erasure endpoint is also a DELETE mounted under the same
        gate, is NOT exempted, and must still 403 a platform_role='viewer'
        caller even if they hold an app-level ADMINISTRATOR role."""
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        viewer_platform_admin = UserFactory()
        AppCollaboratorFactory(
            app=app_obj,
            user=viewer_platform_admin,
            role=CollaborationRole.ADMINISTRATOR,
            status=CollaborationStatus.ACCEPTED,
        )
        db.flush()
        assert viewer_platform_admin.platform_role == "viewer"
        _authenticate_as(memory_app, viewer_platform_admin)

        response = memory_client.delete(f"/apps/{app_obj.app_id}/global-memory/users/{owner.user_id}")

        assert response.status_code == 403
