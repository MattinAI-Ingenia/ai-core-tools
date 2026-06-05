"""Typed domain exceptions for user-deletion orchestration.

These are plain Python exceptions — no FastAPI or HTTPException dependency.
The router layer (step_006) maps them to the appropriate HTTP status codes:

    UserNotFoundError          → 404
    SelfDeletionError          → 400
    OmniadminDeletionError     → 403
    OwnedAppsPresentError      → 409

Design note (AD-7): keeping HTTP semantics out of the service layer means the
same errors can be raised from within a background task or another service
without importing FastAPI internals.
"""
from __future__ import annotations

from typing import List, TypedDict


# ---------------------------------------------------------------------------
# Typed payloads
# ---------------------------------------------------------------------------

class OwnedAppSummary(TypedDict):
    """Minimal app descriptor returned in the 409 owned-apps payload."""

    app_id: int
    name: str


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class UserDeletionError(Exception):
    """Base class for all user-deletion domain errors.

    Args:
        message: Human-readable description surfaced to the API caller.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# Concrete errors
# ---------------------------------------------------------------------------

class UserNotFoundError(UserDeletionError):
    """Raised when the target user_id does not exist in the database.

    Maps to HTTP 404.

    Args:
        user_id: The ID that was not found.
    """

    def __init__(self, user_id: int) -> None:
        super().__init__(f"User {user_id} not found.")
        self.user_id = user_id


class SelfDeletionError(UserDeletionError):
    """Raised when the acting administrator attempts to delete their own account.

    Maps to HTTP 400.

    Args:
        user_id: The shared actor/target user_id.
    """

    def __init__(self, user_id: int) -> None:
        super().__init__("An administrator cannot delete their own account.")
        self.user_id = user_id


class OmniadminDeletionError(UserDeletionError):
    """Raised when the target user is an omniadmin and therefore undeletable.

    Maps to HTTP 403.

    Args:
        email: The omniadmin e-mail address, for structured-logging context only.
          Never forwarded verbatim to the API response body.
    """

    def __init__(self, email: str) -> None:
        super().__init__("Cannot delete an administrator account.")
        self.email = email


class OwnedAppsPresentError(UserDeletionError):
    """Raised when the target user owns apps and mode is 'block'.

    Maps to HTTP 409.  The ``owned_apps`` list is included in the 409 response
    body so the caller can present a transfer/cascade dialog to the admin.

    Args:
        owned_apps: List of dicts, each with keys ``app_id`` (int) and
            ``name`` (str), representing the apps the user currently owns.
    """

    def __init__(self, owned_apps: List[OwnedAppSummary]) -> None:
        count = len(owned_apps)
        super().__init__(
            f"User owns {count} app(s). Choose 'cascade_apps' to delete them "
            "or 'transfer_apps' to reassign ownership before deleting the user."
        )
        self.owned_apps: List[OwnedAppSummary] = owned_apps
