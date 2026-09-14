"""
Right-to-erasure endpoints for cross-session user memory (spec
`user-memory-mem0`, FR-8, D-8, D-11, step_006).

Three surfaces, all delegating to the **same** service functions
(`services.global_memory_service.delete_global_memory` /
`delete_user_global_memory`) — no endpoint reimplements the purge:

1. **Operator erasure** — `DELETE /internal/apps/{app_id}/global-memory/users/{user_id}`,
   gated `ADMINISTRATOR`+ in `{app_id}` (an ADMINISTRATOR of another App can
   never use this to erase here). Returns `204 No Content` with no body —
   **never** a purge count — so a varying response cannot be used as a
   destructive oracle for which user PKs hold memory in this App, including
   non-members (AC-62).
2. **Self-service, current App** — `DELETE /internal/apps/{app_id}/global-memory/me`,
   any authenticated App member (VIEWER+). The identity purged comes
   exclusively from the session (see `erase_my_global_memory`'s docstring).
3. **Cross-App self-service** — `DELETE /internal/me/global-memory`,
   registered directly on the top-level internal router (see
   `backend/routers/internal/__init__.py`), **not** nested under
   `apps_router`: authentication only, no App-membership dependency at all,
   because the people this endpoint exists for may hold memory in an App
   they never joined and cannot even discover its `app_id` (R-24/AC-61).

**Incompleteness surfaces as `503`, never a swallowed success.** Every one
of the three handlers below now checks `result.complete` and raises
`HTTP_503_SERVICE_UNAVAILABLE` when it is `False` — an interrupted purge
(a single exact scope) or, for the cross-App handler, an enumeration that
never finished (`tools.memory.client.list_user_scopes` returning
`ScopeListResult(complete=False)`, folded into `delete_user_global_memory`'s
own `PurgeResult.complete`) must never be answered as a `204`/`200` success
(NFR-5). This does **not** reintroduce the AC-62 purge-count oracle on the
operator endpoint: `204` is returned whether the scope held nothing or held
memory that was actually purged — the only other outcome is `503`, which
reports that the *operation* failed, not anything about whether the target
scope holds (or held) data. A generic detail message is used on every `503`
below for the same reason — it must never vary with `result.deleted`.

None of these three call `adelete_thread` or anything in
`backend/services/agent_cache_service.py` (D-8) — a full data-subject
erasure is memory purge **plus** conversation deletion (existing, separate,
`ConversationService.delete_conversation`); see
`global_memory_service`'s module docstring.

Routers only translate HTTP <-> service call <-> response; all erasure logic
lives in `services.global_memory_service`.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from lks_idprovider.models.auth import AuthContext
from sqlalchemy.orm import Session

from db.database import get_db
from routers.controls.role_authorization import AppRole, require_min_role
from schemas.global_memory_schemas import GlobalMemoryPurgeResponseSchema
from services.global_memory_service import delete_global_memory, delete_user_global_memory
from utils.logger import get_logger

from .auth_utils import get_current_user_oauth

logger = get_logger(__name__)

# Mounted under `apps_router` with prefix "/{app_id}/global-memory"
# (`backend/routers/internal/apps.py`) — the operator and current-App
# self-service endpoints.
global_memory_router = APIRouter()

# Mounted directly on `internal_router`, no prefix (`backend/routers/internal/__init__.py`)
# — the cross-App self-service endpoint. Authentication only, deliberately
# not nested under `apps_router` (there is no single `{app_id}` to nest under).
me_global_memory_router = APIRouter(tags=["Global Memory"])


@global_memory_router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Operator erasure of a user's global memory in this App",
    description=(
        "Purges every durable global-memory fact stored for this exact (app_id, "
        "user_id) scope. Requires ADMINISTRATOR or higher in THIS App — an "
        "ADMINISTRATOR of a different App can never use this endpoint to erase "
        "here. Returns 204 with no body regardless of whether the scope held any "
        "memory: a varying response would let an administrator enumerate, "
        "destructively, which user PKs have memory in this App, including people "
        "who are not (or are no longer) members. This purges durable memory "
        "facts ONLY — it does not delete conversation history. A full "
        "data-subject erasure is this call plus conversation deletion (a "
        "separate, already-existing capability)."
    ),
    tags=["Global Memory"],
)
async def erase_user_global_memory(
    app_id: int,
    user_id: int,
    role: Annotated[AppRole, Depends(require_min_role("administrator"))],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Operator-gated erasure. See the module docstring for the AC-62
    no-purge-count-oracle rule — the response shape (204 vs 503) must never
    vary with `result.deleted`, only with `result.complete`. `result.deleted`
    itself is still never returned to the caller. A failed or partial purge
    is also surfaced loudly server-side, via `delete_global_memory`'s own
    ERROR-level `log_memory_event` call.
    """
    try:
        result = delete_global_memory(db, app_id, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.error(
            "erase_user_global_memory: unexpected error app_id=%s user_id=%s by=%s",
            app_id, user_id, auth_context.identity.email,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to erase global memory due to an unexpected error.",
        )

    if not result.complete:
        logger.error(
            "erase_user_global_memory: incomplete purge app_id=%s user_id=%s by=%s",
            app_id, user_id, auth_context.identity.email,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Global memory erasure could not be completed. Please retry.",
        )

    logger.info(
        "global_memory:erase_by_operator app_id=%s user_id=%s by_user_id=%s",
        app_id, user_id, auth_context.identity.id,
    )
    # Deliberately no body — see the AC-62 rule above.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@global_memory_router.delete(
    "/me",
    response_model=GlobalMemoryPurgeResponseSchema,
    summary="Self-service erasure of the caller's own global memory in this App",
    description=(
        "Purges every durable global-memory fact stored about the caller in this "
        "App. Available to any authenticated member of this App (VIEWER or "
        "higher) — the action only ever affects the caller's own data. Idempotent: "
        "a caller with nothing stored gets purged=0, which is a success. This "
        "does NOT delete conversation history — memory blocks already injected "
        "into existing conversation transcripts are unaffected; use the existing "
        "conversation-deletion endpoint for that. Memory is shared across every "
        "memory-enabled agent in this App, so this action clears it for all of "
        "them, not just the current chat."
    ),
    tags=["Global Memory"],
)
async def erase_my_global_memory(
    app_id: int,
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
) -> GlobalMemoryPurgeResponseSchema:
    """Self-service erasure, current App.

    The identity purged is **always** the caller's own, taken exclusively
    from `auth_context` (the authenticated session) — this handler
    deliberately declares no `user_id` parameter anywhere (body, path, query,
    or header), and must never read one opportunistically. A caller-supplied
    identifier here would be a privilege-escalation vector, not a
    convenience: the `/me` literal in the path exists precisely so there is
    no identifier to tamper with (D-11, AC-45). If an operator variant of
    this action is ever needed, it must stay the **separate**
    `erase_user_global_memory` handler above, gated by its own
    ADMINISTRATOR+ role dependency — reusing this handler with an "optional
    user_id" is exactly the mistake AC-45 exists to catch.
    """
    session_user_id = int(auth_context.identity.id)
    try:
        result = delete_global_memory(db, app_id, session_user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.error(
            "erase_my_global_memory: unexpected error app_id=%s user_id=%s",
            app_id, session_user_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to erase global memory due to an unexpected error.",
        )

    if not result.complete:
        logger.error(
            "erase_my_global_memory: incomplete purge app_id=%s user_id=%s",
            app_id, session_user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Global memory erasure could not be completed. Please retry.",
        )
    return GlobalMemoryPurgeResponseSchema(purged=result.deleted)


@me_global_memory_router.delete(
    "/me/global-memory",
    response_model=GlobalMemoryPurgeResponseSchema,
    summary="Cross-App self-service erasure of the caller's own global memory",
    description=(
        "Purges every durable global-memory fact stored about the caller, across "
        "EVERY App — including Apps the caller does not belong to, or no longer "
        "belongs to. Authentication only: no App-membership check, because the "
        "people this endpoint exists for are exactly the ones who cannot "
        "discover which app_id holds their data. Enumerates by namespace suffix "
        "from the memory store itself, never from AppCollaborator rows, so a "
        "scope whose membership has already been removed is never missed. "
        "Idempotent: nothing stored is a success (purged=0). Does NOT delete "
        "conversation history in any App."
    ),
)
async def erase_my_global_memory_everywhere(
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
) -> GlobalMemoryPurgeResponseSchema:
    """Cross-App self-service erasure.

    Same identity rule as `erase_my_global_memory`: the user purged is
    exclusively `auth_context`'s own session identity, never anything from
    the request body/query/headers.
    """
    session_user_id = int(auth_context.identity.id)
    try:
        result = delete_user_global_memory(db, session_user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.error(
            "erase_my_global_memory_everywhere: unexpected error user_id=%s",
            session_user_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to erase global memory due to an unexpected error.",
        )

    if not result.complete:
        # Covers both an interrupted purge AND an enumeration
        # (`list_user_scopes`) that never finished across Apps — either way
        # this cross-App erasure must not be answered as a success.
        logger.error(
            "erase_my_global_memory_everywhere: incomplete purge user_id=%s",
            session_user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Global memory erasure could not be completed. Please retry.",
        )
    return GlobalMemoryPurgeResponseSchema(purged=result.deleted)
