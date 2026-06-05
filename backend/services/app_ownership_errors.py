"""Typed domain exceptions for app ownership transfer.

These are plain Python exceptions — no FastAPI or HTTPException dependency.
The router layer maps them to the appropriate HTTP status codes:

    AppNotFoundError                → 404
    TransferRecipientInvalidError   → 400
    TierLimitExceededError          → 409
    OwnershipOfferNotFoundError     → 404
    NotOfferRecipientError          → 403

Design note (AD-7): keeping HTTP semantics out of the service layer means the
same errors can be raised from within a background task or another service
without importing FastAPI internals.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AppOwnershipError(Exception):
    """Base class for all app-ownership-transfer domain errors.

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

class AppNotFoundError(AppOwnershipError):
    """Raised when the target app_id does not exist in the database.

    Maps to HTTP 404.

    Args:
        app_id: The app ID that was not found.
    """

    def __init__(self, app_id: int) -> None:
        super().__init__(f"App {app_id} not found.")
        self.app_id = app_id


class TransferRecipientInvalidError(AppOwnershipError):
    """Raised when the proposed new owner fails any validation check.

    Maps to HTTP 400.

    Reasons include: user does not exist, user is not active, or the user
    is already the current owner of the app.

    Args:
        reason: Short human-readable explanation (logged and returned to caller).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TierLimitExceededError(AppOwnershipError):
    """Raised when a transfer would push the recipient over their app-count limit.

    Maps to HTTP 409.

    Args:
        user_id: The prospective new owner whose limit would be exceeded.
        detail: Optional tier-enforcement message (from TierEnforcementService).
    """

    def __init__(self, user_id: int, detail: str = "") -> None:
        msg = f"Transfer would exceed the app limit for user {user_id}."
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)
        self.user_id = user_id
        self.detail = detail


class OwnershipOfferNotFoundError(AppOwnershipError):
    """Raised when the pending OWNER offer (AppCollaborator row) cannot be found.

    Maps to HTTP 404.

    Args:
        collaboration_id: The collaboration row ID that was not found, if known.
        app_id: The app ID associated with the missing offer, if known.
    """

    def __init__(
        self,
        collaboration_id: int | None = None,
        app_id: int | None = None,
    ) -> None:
        if collaboration_id is not None:
            msg = f"Ownership offer {collaboration_id} not found."
        elif app_id is not None:
            msg = f"No pending ownership offer found for app {app_id}."
        else:
            msg = "Ownership offer not found."
        super().__init__(msg)
        self.collaboration_id = collaboration_id
        self.app_id = app_id


class NotOfferRecipientError(AppOwnershipError):
    """Raised when an actor tries to accept/decline an offer that is not addressed to them.

    Maps to HTTP 403.

    Args:
        actor_user_id: The user who attempted the action.
        collaboration_id: The collaboration row they attempted to act on.
    """

    def __init__(self, actor_user_id: int, collaboration_id: int) -> None:
        super().__init__(
            f"User {actor_user_id} is not the recipient of ownership offer {collaboration_id}."
        )
        self.actor_user_id = actor_user_id
        self.collaboration_id = collaboration_id
