"""Unit tests for the FR-8 cascade wiring into `AppService.delete_app` and
`UserService.delete_user` (spec `user-memory-mem0`, step_006).

`services.global_memory_service.delete_app_global_memory` /
`delete_user_global_memory` are mocked here. Security-audit MEDIUM 2 (step_006
follow-up) changed `AppService.delete_app`'s behaviour: a raised/incomplete
`delete_app_global_memory` now ABORTS the App deletion (raises
`GlobalMemoryPurgeIncompleteError`, App row untouched) instead of being
logged and swallowed — see `AppService.delete_app`'s step-0 comment for why.
`UserService.delete_user`'s OWN purge (of the user being deleted, as opposed
to the nested `AppService.delete_app` calls in `cascade_apps` mode) keeps the
original log-and-continue behaviour; those cases are still covered below.
The purge functions' own behaviour (real store, aggregation, idempotence) is
covered by `test_global_memory_erasure.py` and the integration suite.
"""
from unittest.mock import patch

import pytest

from repositories.app_repository import AppRepository
from services.app_service import AppService
from services.global_memory_service import GlobalMemoryPurgeIncompleteError
from services.user_service import UserService
from tools.memory.client import PurgeResult

from tests.factories import AppFactory, UserFactory, configure_factories


class TestAppServiceDeleteAppPurgesMemory:
    """AC-29."""

    def test_calls_delete_app_global_memory_with_the_apps_scope(self, db):
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        db.flush()
        app_id = app_obj.app_id

        with patch(
            "services.global_memory_service.delete_app_global_memory",
            return_value=PurgeResult(deleted=3, complete=True),
        ) as mock_purge:
            success = AppService(db).delete_app(app_id)

        assert success is True
        mock_purge.assert_called_once_with(db, app_id)

    def test_a_raising_memory_purge_aborts_app_deletion(self, db):
        """Regression (security-audit MEDIUM 2): a memory-store exception
        must abort the App deletion — the App must survive, retryable."""
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        # Committed (not just flushed): `delete_app`'s abort path calls
        # `self.db.rollback()`, which — in this test session's SAVEPOINT-based
        # transaction (see the `db` fixture) — would otherwise also discard an
        # uncommitted fixture insert, not just the App's own (nonexistent, at
        # step 0) mutations.
        db.commit()
        app_id = app_obj.app_id

        with patch(
            "services.global_memory_service.delete_app_global_memory",
            side_effect=RuntimeError("store unreachable"),
        ):
            with pytest.raises(GlobalMemoryPurgeIncompleteError):
                AppService(db).delete_app(app_id)

        assert AppRepository(db).get_by_id(app_id) is not None

    def test_an_incomplete_memory_purge_aborts_app_deletion(self, db):
        """Regression (security-audit MEDIUM 2): `complete=False` (no
        exception, just an interrupted walk) must also abort the deletion —
        the incompleteness must never be silently swallowed into a
        'successful' deletion that orphans surviving memory (NFR-5/AC-61)."""
        configure_factories(db)
        owner = UserFactory()
        app_obj = AppFactory(owner=owner)
        db.commit()  # see the sibling test above for why this must be committed, not just flushed
        app_id = app_obj.app_id

        with patch(
            "services.global_memory_service.delete_app_global_memory",
            return_value=PurgeResult(deleted=1, complete=False),
        ):
            with pytest.raises(GlobalMemoryPurgeIncompleteError):
                AppService(db).delete_app(app_id)

        assert AppRepository(db).get_by_id(app_id) is not None


class TestUserServiceDeleteUserPurgesMemory:
    """AC-30."""

    def test_block_mode_purges_memory_for_a_user_with_no_owned_apps(self, db):
        configure_factories(db)
        actor = UserFactory()
        target = UserFactory()
        db.flush()

        with patch(
            "services.global_memory_service.delete_user_global_memory",
            return_value=PurgeResult(deleted=2, complete=True),
        ) as mock_purge:
            success = UserService.delete_user(db, target.user_id, actor_user_id=actor.user_id, mode="block")

        assert success is True
        mock_purge.assert_called_once_with(db, target.user_id)

    def test_cascade_apps_mode_purges_the_deleted_users_memory(self, db):
        configure_factories(db)
        actor = UserFactory()
        target = UserFactory()
        AppFactory(owner=target)
        db.flush()

        with patch(
            "services.global_memory_service.delete_app_global_memory",
            return_value=PurgeResult(deleted=0, complete=True),
        ), patch(
            "services.global_memory_service.delete_user_global_memory",
            return_value=PurgeResult(deleted=1, complete=True),
        ) as mock_user_purge:
            success = UserService.delete_user(db, target.user_id, actor_user_id=actor.user_id, mode="cascade_apps")

        assert success is True
        mock_user_purge.assert_called_once_with(db, target.user_id)

    def test_cascade_apps_mode_aborts_if_an_owned_apps_memory_purge_is_incomplete(self, db):
        """Regression (security-audit MEDIUM 2, knock-on effect): `cascade_apps`
        deletes owned apps via `AppService.delete_app`, which now aborts on an
        incomplete memory purge — that abort must propagate out of
        `delete_user` too, leaving both the owned app and the target user
        intact (retryable), rather than reporting a successful user deletion
        that silently skipped an owned app's memory purge."""
        configure_factories(db)
        actor = UserFactory()
        target = UserFactory()
        owned_app = AppFactory(owner=target)
        # Committed, not just flushed — see the sibling AppService tests above
        # for why: `delete_app`'s abort path rolls back the current
        # transaction, which would otherwise discard these uncommitted rows too.
        db.commit()
        target_user_id = target.user_id
        owned_app_id = owned_app.app_id

        with patch(
            "services.global_memory_service.delete_app_global_memory",
            return_value=PurgeResult(deleted=1, complete=False),
        ):
            with pytest.raises(GlobalMemoryPurgeIncompleteError):
                UserService.delete_user(db, target_user_id, actor_user_id=actor.user_id, mode="cascade_apps")

        from repositories.user_repository import UserRepository

        assert AppRepository(db).get_by_id(owned_app_id) is not None
        assert UserRepository(db).get_by_id(target_user_id) is not None

    def test_transfer_apps_mode_purges_memory_in_the_surviving_app(self, db):
        """AC-30's named case: the App survives transfer, but the deleted
        user's memory in it must still be purged."""
        configure_factories(db)
        actor = UserFactory()
        target = UserFactory()
        recipient = UserFactory()
        owned_app = AppFactory(owner=target)
        db.flush()

        with patch(
            "services.global_memory_service.delete_user_global_memory",
            return_value=PurgeResult(deleted=1, complete=True),
        ) as mock_user_purge:
            success = UserService.delete_user(
                db,
                target.user_id,
                actor_user_id=actor.user_id,
                mode="transfer_apps",
                transfer_to_user_id=recipient.user_id,
            )

        assert success is True
        mock_user_purge.assert_called_once_with(db, target.user_id)
        # The App itself survives transfer_apps — only ownership moves.
        surviving_app = AppRepository(db).get_by_id(owned_app.app_id)
        assert surviving_app is not None
        assert surviving_app.owner_id == recipient.user_id

    def test_a_raising_memory_purge_does_not_abort_user_deletion(self, db):
        configure_factories(db)
        actor = UserFactory()
        target = UserFactory()
        db.flush()

        with patch(
            "services.global_memory_service.delete_user_global_memory",
            side_effect=RuntimeError("store unreachable"),
        ):
            success = UserService.delete_user(db, target.user_id, actor_user_id=actor.user_id, mode="block")

        assert success is True

    def test_an_incomplete_memory_purge_does_not_abort_user_deletion(self, db):
        configure_factories(db)
        actor = UserFactory()
        target = UserFactory()
        db.flush()

        with patch(
            "services.global_memory_service.delete_user_global_memory",
            return_value=PurgeResult(deleted=1, complete=False),
        ):
            success = UserService.delete_user(db, target.user_id, actor_user_id=actor.user_id, mode="block")

        assert success is True
