"""
Cross-session user memory — right-to-erasure service functions (spec
`user-memory-mem0`, FR-8, D-8, D-11, AD-10, step_006).

**A full data-subject erasure is two calls, not one**: `delete_global_memory`
(this module) purges only the durable memory *facts* held in the store —
it never touches conversation history. Conversation history is an
already-existing, separate capability:
`ConversationService.delete_conversation` (`backend/services/conversation_service.py:274`),
which owns `adelete_thread` via `backend/services/agent_cache_service.py:168`.
This module must **never** call `adelete_thread`, `CheckpointerCacheService`,
or anything in `agent_cache_service.py` (D-8, AC-42) — duplicating that
capability here would create two competing history-deletion mechanisms.
Neither call implies the other; an operator serving a GDPR request needs both.

**Authorization boundary (the step_006 decision).** Every function below
builds its `MemoryIdentity` via `MemoryIdentity.for_operator_purge(...)`
(`tools/memory/identity.py`), never via `resolve_identity`: erasure must
succeed for a target who no longer holds any role in the App, was never a
member of it, or would fail `resolve_identity`'s eligibility checks entirely
— a GDPR request cannot be conditioned on the same eligibility a chat turn
requires. `for_operator_purge` therefore carries **no authorization of its
own** (`MemoryIdentity` is shape-proof, not authorization-proof) — this
module's callers are solely responsible for authorization: the router's role
dependency for the operator endpoint, the session-derived identity for the
self-service endpoints, and the App-row lookup that already gates
`AppService.delete_app` / `UserService.delete_user`.

**Idempotence and completeness (AC-26/AC-28, NFR-5).** A scope with nothing
stored is a success reporting `deleted=0` — never an error. But a partial
failure must never be indistinguishable from that: every `PurgeResult` (and
the `PurgeResult`-shaped aggregate this module returns for App-wide /
user-wide erasure) carries `complete: bool` precisely so a half-completed
walk is never read as a clean success. Callers (routers, cascades) must
propagate that distinction rather than collapsing it to a bare count.

This applies one level up too: `delete_app_global_memory`/
`delete_user_global_memory` enumerate their scopes via
`tools.memory.client.list_app_scopes`/`list_user_scopes`, which themselves
return a `ScopeListResult(scopes, complete)` rather than a bare list for the
identical reason — an enumeration that failed outright must never look like
"this app/user holds no memory" (zero scopes collected either way). Both
functions fold `ScopeListResult.complete` into their own aggregate
`complete` *before* iterating any scopes, so a failed enumeration alone
(even with zero scopes to purge) already yields `complete=False`.

The read/write policy functions this module will also host
(`get_memory_context` / `prepare_turn_memory` / the write path) are added by
step_007/step_008 — this module currently contains only the erasure surface.
"""
from sqlalchemy.orm import Session

from tools.memory.client import PurgeResult, ScopeListResult, list_app_scopes, list_user_scopes, purge
from tools.memory.identity import MemoryIdentity, is_eligible_app_id, is_eligible_user_id
from tools.memory.settings import FAIL_ERROR, log_memory_event
from utils.logger import get_logger

logger = get_logger(__name__)


class GlobalMemoryPurgeIncompleteError(Exception):
    """A global-memory purge step did not complete, and the caller has
    decided that must abort the surrounding operation rather than be
    silently logged and swallowed (security-audit MEDIUM 2, spec
    `user-memory-mem0`, step_006 follow-up).

    Kept as its own exception type — never folded into a bare `RuntimeError`
    or the generic cascade-deletion `except Exception` — specifically so a
    router can tell "the memory store hiccuped, retry the deletion" (503,
    safe to retry) apart from "an ordinary cascade-deletion bug" (500, needs
    investigation). See `AppService.delete_app`'s step-0 usage: it is raised
    there ONLY before any row for that App has been touched, which is what
    makes re-raising (instead of continuing) cheap and safe.
    """


def delete_global_memory(db: Session, app_id: int, user_id: int) -> PurgeResult:
    """Purge every durable memory fact held for the exact `(app_id, user_id)` scope.

    Idempotent (AC-28): a scope with no memory succeeds and reports
    `PurgeResult(deleted=0, complete=True)` — a normal outcome, not an error.

    Works regardless of the deployment-wide kill switch. The kill switch
    (`AICT_GLOBAL_MEMORY_ENABLED`) only gates the per-turn read/write hooks
    (step_007/step_008); an operator serving a GDPR erasure request must never
    be blockable by that operational flag. `purge()` itself
    (`tools/memory/client.py`) is a module-level function independent of the
    kill switch and the shared connection pool for exactly this reason.

    Never calls `adelete_thread` or anything in
    `backend/services/agent_cache_service.py` (D-8, AC-42) — see the module
    docstring for the "erasure = this + conversation deletion" boundary.

    Args:
        db: SQLAlchemy session. Unused by the purge itself (`purge()` opens
            its own short-lived sync connection, per `tools/memory/client.py`);
            kept as an explicit parameter for interface consistency with the
            read/write policy functions this module gains in step_007/step_008,
            and so a future audit-log write can be added here without changing
            every call site's signature.
        app_id: The exact App scope to purge.
        user_id: The exact user scope to purge.

    Returns:
        The `PurgeResult` from `tools.memory.client.purge`. `complete=False`
        means the walk was interrupted — `deleted` is then a partial count,
        never a guaranteed total — and callers (the routers, the cascades)
        must surface that as a failed/partial erasure, never as an
        unqualified `200`/`204` success.

    Raises:
        ValueError: `app_id` or `user_id` is not a positive-int-shaped scope.
    """
    if not is_eligible_app_id(app_id) or not is_eligible_user_id(user_id):
        raise ValueError(
            "delete_global_memory requires a positive-int app_id and user_id; "
            f"got app_id={app_id!r}, user_id={user_id!r}"
        )

    identity = MemoryIdentity.for_operator_purge(app_id=int(app_id), user_id=int(user_id))
    result = purge(identity)

    log_memory_event(
        "erase",
        app_id=int(app_id),
        agent_id=None,
        active=True,
        count=result.deleted,
        reason=None if result.complete else FAIL_ERROR,
        level="info" if result.complete else "error",
    )
    return result


def delete_app_global_memory(db: Session, app_id: int) -> PurgeResult:
    """Purge durable memory for every scope the store itself reports under `app_id`.

    Enumerates directly from the store via `list_app_scopes` (AD-10) —
    **never** by inferring candidate users from `App.owner_id` /
    `AppCollaborator` / `Conversation`, which is the superseded design's
    workaround for a missing prefix-listing API and could silently miss a
    former collaborator whose memory outlived their App membership. Because
    enumeration comes from the store and not from this App's own rows, this
    purge has **no ordering dependency** on when in `AppService.delete_app`'s
    cascade it runs relative to agent/collaborator deletion.

    Aggregates every enumerated scope's `PurgeResult`: `deleted` is the sum
    across scopes, and `complete` is `True` only if **every** scope's own
    walk completed AND `list_app_scopes`'s own enumeration walk completed —
    one interrupted scope, or an enumeration that never finished (even one
    that collected zero scopes), makes the whole call `complete=False`, so a
    caller can never read a partially-failed or never-enumerated App-wide
    purge as a clean success (the same rule `purge()` itself follows for a
    single scope, folded up one level for `list_app_scopes`'s own
    `ScopeListResult.complete`).

    Args:
        db: SQLAlchemy session. Unused; see `delete_global_memory`'s
            docstring for why it is still an explicit parameter.
        app_id: The App to purge every held memory scope for.

    Returns:
        The aggregated `PurgeResult` across every scope `list_app_scopes`
        returned for `app_id`. `complete=False` here can mean either "one or
        more scopes' purge did not finish" or "enumeration itself did not
        finish" — either way it must be surfaced as a failed/partial
        erasure, never a `200`/`204` success.

    Raises:
        ValueError: `app_id` is not a positive-int-shaped scope.
    """
    if not is_eligible_app_id(app_id):
        raise ValueError(f"delete_app_global_memory requires a positive-int app_id; got {app_id!r}")

    scope_result: ScopeListResult = list_app_scopes(int(app_id))
    scopes = scope_result.scopes
    total_deleted = 0
    all_complete = scope_result.complete

    for scope in scopes:
        result = purge(scope)
        total_deleted += result.deleted
        all_complete = all_complete and result.complete
        log_memory_event(
            "erase",
            app_id=int(app_id),
            agent_id=None,
            active=True,
            count=result.deleted,
            reason=None if result.complete else FAIL_ERROR,
            level="debug" if result.complete else "error",
        )

    log_memory_event(
        "erase_app_summary",
        app_id=int(app_id),
        agent_id=None,
        active=True,
        count=total_deleted,
        reason=None if all_complete else FAIL_ERROR,
        level="info" if all_complete else "error",
    )
    return PurgeResult(deleted=total_deleted, complete=all_complete)


def delete_user_global_memory(db: Session, user_id: int) -> PurgeResult:
    """Purge durable memory for `user_id` across **every** App it exists in.

    Enumerates via `list_user_scopes` — a namespace **suffix** listing
    (AD-10) — never from App ownership or `AppCollaborator` rows. This is
    what makes the function correct under all three `UserService.delete_user`
    modes (`block`, `cascade_apps`, `transfer_apps`): a scope whose
    membership row has *already* been removed (e.g. mid-cascade, after
    collaborations are dropped) is never missed, because nothing here reads
    membership at all. It is also what powers the cross-App self-service
    endpoint (`DELETE /internal/me/global-memory`) for users who have facts
    in an App they were never even a member of (R-24/AC-61) — membership
    gating would lock out exactly the people that endpoint exists for.

    Aggregation semantics mirror `delete_app_global_memory`: summed
    `deleted`, `complete` only if every enumerated scope's purge completed
    AND `list_user_scopes`'s own enumeration walk completed — an enumeration
    that never finished (even one that collected zero scopes) makes the
    whole call `complete=False` rather than a clean, misleading success.

    Args:
        db: SQLAlchemy session. Unused; see `delete_global_memory`'s
            docstring for why it is still an explicit parameter.
        user_id: The user whose memory is purged, in every App it is found.

    Returns:
        The aggregated `PurgeResult` across every scope `list_user_scopes`
        returned for `user_id`. `complete=False` here can mean either "one
        or more scopes' purge did not finish" or "enumeration itself did not
        finish" — either way it must be surfaced as a failed/partial
        erasure, never a `200`/`204` success.

    Raises:
        ValueError: `user_id` is not a positive-int-shaped scope.
    """
    if not is_eligible_user_id(user_id):
        raise ValueError(f"delete_user_global_memory requires a positive-int user_id; got {user_id!r}")

    scope_result: ScopeListResult = list_user_scopes(int(user_id))
    scopes = scope_result.scopes
    total_deleted = 0
    all_complete = scope_result.complete

    for scope in scopes:
        result = purge(scope)
        total_deleted += result.deleted
        all_complete = all_complete and result.complete
        log_memory_event(
            "erase",
            app_id=scope.app_id,
            agent_id=None,
            active=True,
            count=result.deleted,
            reason=None if result.complete else FAIL_ERROR,
            level="debug" if result.complete else "error",
        )

    log_memory_event(
        "erase_user_summary",
        app_id=None,
        agent_id=None,
        active=True,
        count=total_deleted,
        reason=None if all_complete else FAIL_ERROR,
        level="info" if all_complete else "error",
    )
    return PurgeResult(deleted=total_deleted, complete=all_complete)
