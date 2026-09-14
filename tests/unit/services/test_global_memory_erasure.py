"""Unit tests for `services.global_memory_service` — the FR-8 erasure surface
(spec `user-memory-mem0`, D-8, D-11, AD-10, step_006).

The memory store itself (`tools.memory.client`) is mocked throughout — per
NFR-8 it is the single seam this feature is built to be mockable at, so
these tests need no real Postgres/pgvector schema. See
`tests/integration/tools/test_global_memory_erasure_integration.py` for the
real-store proof of AC-26/AC-27/AC-28 (erasure actually removes data and is
scoped) and AC-42 (no checkpointer call).
"""
from unittest.mock import MagicMock, patch

import pytest

import services.global_memory_service as global_memory_service
from services.global_memory_service import (
    delete_app_global_memory,
    delete_global_memory,
    delete_user_global_memory,
)
from tools.memory.client import PurgeResult, ScopeListResult
from tools.memory.identity import MemoryIdentity


def _identity(app_id: int, user_id: int, agent_id: int = 0) -> MemoryIdentity:
    from tools.memory import settings as memory_settings

    root = memory_settings.namespace_root()
    return MemoryIdentity(
        app_id=app_id, user_id=user_id, agent_id=agent_id, namespace=(root, str(app_id), str(user_id))
    )


class TestDeleteGlobalMemory:
    def test_success_returns_the_purge_result(self):
        with patch.object(global_memory_service, "purge", return_value=PurgeResult(deleted=4, complete=True)) as mock_purge:
            result = delete_global_memory(MagicMock(), app_id=1, user_id=7)

        assert result == PurgeResult(deleted=4, complete=True)
        mock_purge.assert_called_once()
        identity_arg = mock_purge.call_args[0][0]
        assert isinstance(identity_arg, MemoryIdentity)
        assert (identity_arg.app_id, identity_arg.user_id) == (1, 7)

    def test_empty_scope_is_a_success_not_an_error(self):
        """AC-28."""
        with patch.object(global_memory_service, "purge", return_value=PurgeResult(deleted=0, complete=True)):
            result = delete_global_memory(MagicMock(), app_id=1, user_id=7)

        assert result == PurgeResult(deleted=0, complete=True)

    def test_second_call_is_idempotent(self):
        """AC-28 — a second call for an already-purged scope also succeeds with 0."""
        with patch.object(
            global_memory_service,
            "purge",
            side_effect=[PurgeResult(deleted=2, complete=True), PurgeResult(deleted=0, complete=True)],
        ) as mock_purge:
            first = delete_global_memory(MagicMock(), app_id=1, user_id=7)
            second = delete_global_memory(MagicMock(), app_id=1, user_id=7)

        assert first.deleted == 2
        assert second == PurgeResult(deleted=0, complete=True)
        assert mock_purge.call_count == 2

    def test_incomplete_purge_is_propagated_not_hidden(self):
        """A partial purge must never look like a clean success (NFR-5)."""
        with patch.object(global_memory_service, "purge", return_value=PurgeResult(deleted=1, complete=False)):
            result = delete_global_memory(MagicMock(), app_id=1, user_id=7)

        assert result.complete is False
        assert result.deleted == 1

    @pytest.mark.parametrize(
        "app_id,user_id",
        [(None, 7), (1, None), (0, 7), (1, -3), ("x", 7), (1, True)],
    )
    def test_rejects_ineligible_scope_shapes(self, app_id, user_id):
        with pytest.raises(ValueError):
            delete_global_memory(MagicMock(), app_id=app_id, user_id=user_id)

    def test_never_touches_the_checkpointer(self):
        """D-8/AC-42 — structural guard: this module's actual CODE (imports,
        attribute access, calls — not prose in docstrings, which legitimately
        explain the D-8 boundary by name) must never reference
        adelete_thread / CheckpointerCacheService / agent_cache_service."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(global_memory_service))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                identifiers.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                identifiers.add(node.module)
                identifiers.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)

        assert "adelete_thread" not in identifiers
        assert "CheckpointerCacheService" not in identifiers
        assert "agent_cache_service" not in identifiers

    def test_kill_switch_is_never_consulted(self):
        """Erasure must work even when the deployment kill switch is off —
        `is_global_memory_enabled` must not be called by the erasure
        functions themselves (docstring prose explaining why is fine and
        expected).

        Scoped to the three erasure functions specifically, not the whole
        module: since step_007 this module also hosts the FR-6 read-path
        policy (`prepare_turn_memory`/`get_memory_context`), which correctly
        *does* consult the kill switch as its very first check (AD-6 step 1,
        AC-33) — that is a different code path with a different contract,
        not a regression of this one.
        """
        import ast
        import inspect

        erasure_function_names = {
            "delete_global_memory",
            "delete_app_global_memory",
            "delete_user_global_memory",
        }
        tree = ast.parse(inspect.getsource(global_memory_service))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in erasure_function_names:
                identifiers |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                identifiers |= {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}

        assert "is_global_memory_enabled" not in identifiers


class TestDeleteAppGlobalMemory:
    def test_aggregates_deleted_counts_across_scopes(self):
        scopes = [_identity(1, 7), _identity(1, 8)]
        with patch.object(
            global_memory_service, "list_app_scopes", return_value=ScopeListResult(scopes=scopes, complete=True)
        ), patch.object(
            global_memory_service,
            "purge",
            side_effect=[PurgeResult(deleted=3, complete=True), PurgeResult(deleted=5, complete=True)],
        ):
            result = delete_app_global_memory(MagicMock(), app_id=1)

        assert result == PurgeResult(deleted=8, complete=True)

    def test_one_incomplete_scope_makes_the_whole_result_incomplete(self):
        scopes = [_identity(1, 7), _identity(1, 8)]
        with patch.object(
            global_memory_service, "list_app_scopes", return_value=ScopeListResult(scopes=scopes, complete=True)
        ), patch.object(
            global_memory_service,
            "purge",
            side_effect=[PurgeResult(deleted=3, complete=True), PurgeResult(deleted=1, complete=False)],
        ):
            result = delete_app_global_memory(MagicMock(), app_id=1)

        assert result.complete is False
        assert result.deleted == 4

    def test_no_scopes_is_a_success(self):
        with patch.object(
            global_memory_service, "list_app_scopes", return_value=ScopeListResult(scopes=[], complete=True)
        ):
            result = delete_app_global_memory(MagicMock(), app_id=1)

        assert result == PurgeResult(deleted=0, complete=True)

    def test_enumeration_comes_from_the_store_not_from_app_rows(self):
        """AC-29 (service-level half) — must call `list_app_scopes`; the
        superseded design's inference from AppCollaborator/Conversation must
        never be reintroduced (guarded structurally in the cascade tests too)."""
        with patch.object(
            global_memory_service, "list_app_scopes", return_value=ScopeListResult(scopes=[], complete=True)
        ) as mock_list:
            delete_app_global_memory(MagicMock(), app_id=5)

        mock_list.assert_called_once_with(5)

    def test_rejects_ineligible_app_id(self):
        with pytest.raises(ValueError):
            delete_app_global_memory(MagicMock(), app_id=-1)

    def test_failed_enumeration_with_zero_scopes_is_not_a_clean_success(self):
        """The step_006 gap this closes: `list_app_scopes` failing outright
        returns zero scopes, so the purge loop below runs zero times — but
        that must never collapse to `PurgeResult(deleted=0, complete=True)`,
        which would report a GDPR erasure whose enumeration failed as a
        clean, complete success (NFR-5)."""
        with patch.object(
            global_memory_service, "list_app_scopes", return_value=ScopeListResult(scopes=[], complete=False)
        ), patch.object(global_memory_service, "purge") as mock_purge:
            result = delete_app_global_memory(MagicMock(), app_id=1)

        mock_purge.assert_not_called()
        assert result == PurgeResult(deleted=0, complete=False)

    def test_failed_enumeration_combined_with_scopes_still_collected(self):
        """A page failure mid-walk can still have collected some scopes
        (`ScopeListResult.scopes` is a partial list, not `[]`) — those
        collected scopes are still purged, but the aggregate must still be
        `complete=False` because enumeration itself did not finish."""
        scopes = [_identity(1, 7)]
        with patch.object(
            global_memory_service, "list_app_scopes", return_value=ScopeListResult(scopes=scopes, complete=False)
        ), patch.object(global_memory_service, "purge", return_value=PurgeResult(deleted=2, complete=True)):
            result = delete_app_global_memory(MagicMock(), app_id=1)

        assert result == PurgeResult(deleted=2, complete=False)


class TestDeleteUserGlobalMemory:
    def test_aggregates_across_every_app(self):
        scopes = [_identity(1, 7), _identity(2, 7)]
        with patch.object(
            global_memory_service, "list_user_scopes", return_value=ScopeListResult(scopes=scopes, complete=True)
        ) as mock_list, patch.object(
            global_memory_service,
            "purge",
            side_effect=[PurgeResult(deleted=1, complete=True), PurgeResult(deleted=2, complete=True)],
        ):
            result = delete_user_global_memory(MagicMock(), user_id=7)

        mock_list.assert_called_once_with(7)
        assert result == PurgeResult(deleted=3, complete=True)

    def test_enumeration_uses_suffix_listing_not_membership(self):
        """AC-61/R-24 (service-level half) — `list_user_scopes` is the only
        enumeration source; no AppCollaborator query happens here."""
        with patch.object(
            global_memory_service, "list_user_scopes", return_value=ScopeListResult(scopes=[], complete=True)
        ) as mock_list:
            delete_user_global_memory(MagicMock(), user_id=42)

        mock_list.assert_called_once_with(42)

    def test_no_scopes_is_a_success(self):
        with patch.object(
            global_memory_service, "list_user_scopes", return_value=ScopeListResult(scopes=[], complete=True)
        ):
            result = delete_user_global_memory(MagicMock(), user_id=7)

        assert result == PurgeResult(deleted=0, complete=True)

    def test_rejects_ineligible_user_id(self):
        with pytest.raises(ValueError):
            delete_user_global_memory(MagicMock(), user_id=-1)

    def test_failed_enumeration_with_zero_scopes_is_not_a_clean_success(self):
        """Regression for the step_006 gap: a cross-App erasure whose
        enumeration failed must report `complete=False`, not an inflated
        clean `deleted=0` success (NFR-5)."""
        with patch.object(
            global_memory_service, "list_user_scopes", return_value=ScopeListResult(scopes=[], complete=False)
        ), patch.object(global_memory_service, "purge") as mock_purge:
            result = delete_user_global_memory(MagicMock(), user_id=7)

        mock_purge.assert_not_called()
        assert result == PurgeResult(deleted=0, complete=False)

    def test_failed_enumeration_combined_with_scopes_still_collected(self):
        scopes = [_identity(1, 7)]
        with patch.object(
            global_memory_service, "list_user_scopes", return_value=ScopeListResult(scopes=scopes, complete=False)
        ), patch.object(global_memory_service, "purge", return_value=PurgeResult(deleted=1, complete=True)):
            result = delete_user_global_memory(MagicMock(), user_id=7)

        assert result == PurgeResult(deleted=1, complete=False)
