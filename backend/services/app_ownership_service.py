"""App ownership transfer service.

Provides the single source of truth for all ``App.owner_id`` reassignment logic.
Two public entry points:

- ``transfer_direct``: Administrative-direct path (omniadmin-initiated).  Used
  standalone via the admin API endpoint and internally by ``UserService.delete_user``
  in ``transfer_apps`` mode.

- ``_reassign_owner``: Private primitive called by ``transfer_direct`` (and by
  ``delete_user`` in the loop).  **Never commits**.  The caller owns the transaction
  boundary (AD-2).

Architecture constraints enforced here:
- ``_reassign_owner`` never calls ``db.commit()`` (AD-2).  It may call
  ``db.flush()`` so that intra-transaction queries (e.g. tier count) see the
  staged changes; the caller still owns the commit boundary.
- Tier enforcement (SaaS only / no-op self-managed) via
  ``TierEnforcementService.check_app_limit``.
- Collaborator hygiene: any pre-existing collaborator row for the new owner on
  this app is removed inside the same transaction (FR-C5).
- ``FreezeService.apply_freeze``: called in ``transfer_direct`` **after** the
  commit so it operates on stable DB state.  See inline note for the delete-user
  path.
"""
from __future__ import annotations

from typing import Tuple

from fastapi import HTTPException as _HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from models.app import App
from models.user import User
from models.app_collaborator import AppCollaborator
from repositories.app_collaboration_repository import AppCollaborationRepository
from services.app_ownership_errors import (
    AppNotFoundError,
    TierLimitExceededError,
    TransferRecipientInvalidError,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class AppOwnershipService:
    """Service for administrative-direct app ownership transfer.

    All methods are synchronous to match the rest of the codebase (UserService,
    AppService, etc. are also sync).  Each method receives ``db`` as its first
    argument rather than storing it at construction time, mirroring the
    ``UserService`` static-method pattern.
    """

    # ------------------------------------------------------------------ #
    # Private primitive — NEVER commits                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reassign_owner(
        db: Session,
        app: App,
        new_owner_id: int,
        *,
        actor_user_id: int,
    ) -> None:
        """Stage the owner reassignment for ``app`` onto ``new_owner_id``.

        This is the single source of truth for all transfer invariants.  It
        **must not** call ``db.commit()`` — the caller owns the transaction
        boundary (AD-2).  It may call ``db.flush()`` to make staged changes
        visible to subsequent queries within the same open transaction (e.g. the
        post-transfer app count for the tier check).

        Invariants enforced (in order):
        a. ``new_owner_id`` resolves to a real, active, non-current-owner user.
        b. SaaS tier: ``TierEnforcementService.check_app_limit`` for the new owner
           (no-op in self-managed mode).  Raises ``fastapi.HTTPException`` with
           status 403 in SaaS over-limit; we catch that and re-raise as the
           typed ``TierLimitExceededError`` so the service layer stays free of
           FastAPI imports from the caller's perspective.
        c. ``app.owner_id = new_owner_id`` is staged on the ORM object.
        d. Any ``AppCollaborator`` row for the new owner on this app is deleted
           (ownership supersedes collaboration — FR-C5).

        Args:
            db: Synchronous SQLAlchemy session.  Must be in an open transaction.
            app: The ``App`` ORM instance to reassign.  Loaded by the caller.
            new_owner_id: PK of the user who will become the new owner.
            actor_user_id: PK of the administrator performing the action.
                Used for structured logging only; authorisation is enforced at
                the router layer via ``require_admin``.

        Raises:
            TransferRecipientInvalidError: Recipient does not exist, is not
                active, or is already the current owner of the app (→ 400).
            TierLimitExceededError: Transfer would push the new owner over their
                SaaS app-count limit (→ 409).
        """
        # ---- a. Validate the recipient ----------------------------------------
        recipient: User | None = db.execute(
            select(User).where(User.user_id == new_owner_id)
        ).scalar_one_or_none()

        if recipient is None:
            raise TransferRecipientInvalidError(
                f"User {new_owner_id} does not exist."
            )

        if not recipient.is_active:
            raise TransferRecipientInvalidError(
                f"User {new_owner_id} is not active and cannot receive app ownership."
            )

        if app.owner_id == new_owner_id:
            raise TransferRecipientInvalidError(
                f"User {new_owner_id} is already the owner of app {app.app_id}."
            )

        # ---- b. Tier enforcement BEFORE staging (mirror creation semantics) ---
        # check_app_limit counts the recipient's CURRENT owned apps and raises if
        # they are already at/over their limit — exactly as app *creation* does.
        # We therefore check BEFORE staging this app's owner_id so receiving one
        # app is semantically identical to creating one (no off-by-one).
        #
        # db.flush() first so that prior owner_id reassignments staged earlier in
        # THIS transaction (the multi-app delete_user(transfer_apps) loop) are
        # visible to the COUNT — the session is autoflush=False, so without this
        # the count would miss apps already transferred in the same transaction
        # and a multi-app transfer could silently exceed the limit.
        #
        # check_app_limit raises fastapi.HTTPException on violation; re-raise as
        # the typed TierLimitExceededError so callers stay FastAPI-free. No-op in
        # self-managed mode. On any error the caller's rollback reverts staged state.
        db.flush()

        from services.tier_enforcement_service import TierEnforcementService  # noqa: PLC0415 — circular-safe lazy import

        try:
            TierEnforcementService.check_app_limit(db, new_owner_id)
        except _HTTPException as tier_exc:
            raise TierLimitExceededError(new_owner_id, tier_exc.detail)

        # ---- c. Stage owner_id (after the limit check passes) -----------------
        app.owner_id = new_owner_id

        # ---- d. Collaborator hygiene ------------------------------------------
        # If the new owner already has a collaborator row on this app (any status,
        # any role), remove it.  Ownership supersedes collaboration (FR-C5).
        collab_repo = AppCollaborationRepository(db)
        existing_collab: AppCollaborator | None = (
            collab_repo.get_collaboration_by_app_and_user(app.app_id, new_owner_id)
        )
        if existing_collab is not None:
            logger.info(
                f"transfer_owner: removing pre-existing collaborator row "
                f"collab_id={existing_collab.id} app_id={app.app_id} "
                f"user_id={new_owner_id} actor_user_id={actor_user_id}"
            )
            db.delete(existing_collab)

        logger.info(
            f"transfer_owner: staged owner_id={new_owner_id} for app_id={app.app_id} "
            f"actor_user_id={actor_user_id}"
        )

    # ------------------------------------------------------------------ #
    # Public: standalone administrative-direct transfer                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def transfer_direct(
        db: Session,
        app_id: int,
        new_owner_id: int,
        *,
        actor_user_id: int,
    ) -> Tuple[App, int]:
        """Reassign ``app_id`` to ``new_owner_id`` immediately (no handshake).

        This is the administrative-direct transfer path (FR-C1/FR-C3).  It is
        used by the admin endpoint ``POST /internal/admin/apps/{app_id}/transfer``
        and also internally when ``UserService.delete_user`` is called with
        ``mode='transfer_apps'`` *as a convenience wrapper for standalone calls*.

        The ``delete_user`` path calls ``_reassign_owner`` directly inside its
        own loop to keep all reassignments and the user deletion in one
        transaction.  ``transfer_direct`` owns its own commit and is therefore
        *not* used inside that loop.

        Post-commit:
        - ``FreezeService.apply_freeze`` is triggered for the new owner so the
          transferred app inherits the new owner's tier-freeze state (FR-C7).
          This is called **after** commit so it reads stable DB state.

        Args:
            db: Synchronous SQLAlchemy session.
            app_id: PK of the app to transfer.
            new_owner_id: PK of the user who will become the new owner.
            actor_user_id: PK of the administrator performing the action.

        Returns:
            A tuple ``(app, previous_owner_id)`` — the refreshed ``App`` after the
            transfer and the owner_id captured from the FOR UPDATE-locked row before
            reassignment (authoritative under concurrency, for the response/audit).

        Raises:
            AppNotFoundError: ``app_id`` does not exist in the database (→ 404).
            TransferRecipientInvalidError: Recipient validation failed (→ 400).
            TierLimitExceededError: SaaS tier limit exceeded for new owner (→ 409).
        """
        logger.info(
            f"transfer_direct: start app_id={app_id} new_owner_id={new_owner_id} "
            f"actor_user_id={actor_user_id}"
        )

        # Load and lock the app row to prevent concurrent transfers.
        app: App | None = db.execute(
            select(App).where(App.app_id == app_id).with_for_update()
        ).scalar_one_or_none()

        if app is None:
            raise AppNotFoundError(app_id)

        previous_owner_id: int = app.owner_id

        try:
            AppOwnershipService._reassign_owner(
                db, app, new_owner_id, actor_user_id=actor_user_id
            )
            db.commit()
            db.refresh(app)
        except Exception:
            db.rollback()
            logger.error(
                f"transfer_direct: rollback app_id={app_id} new_owner_id={new_owner_id} "
                f"actor_user_id={actor_user_id}",
                exc_info=True,
            )
            raise

        logger.info(
            f"transfer_direct: complete app_id={app_id} "
            f"previous_owner_id={previous_owner_id} new_owner_id={new_owner_id} "
            f"actor_user_id={actor_user_id}"
        )

        # FR-C7: Re-evaluate freeze/tier for the new owner now that the transfer
        # is committed.  FreezeService.apply_freeze requires the new owner's
        # effective tier, which is resolved inside the service.  We call it
        # inside a guarded block: a freeze failure must never roll back a
        # successfully committed transfer — it is a best-effort post-processing
        # step.  A WARNING is logged so ops can investigate without user impact.
        try:
            from services.freeze_service import FreezeService
            from deployment_mode import is_self_managed

            if not is_self_managed():
                # Determine the new owner's effective tier to pass to apply_freeze.
                from repositories.subscription_repository import SubscriptionRepository

                sub_repo = SubscriptionRepository(db)
                sub = sub_repo.get_by_user_id(new_owner_id)
                if sub:
                    effective_tier = sub.admin_override_tier or (
                        sub.tier.value if sub.tier else "free"
                    )
                    FreezeService.apply_freeze(db, new_owner_id, effective_tier)
                    db.commit()
        except Exception as freeze_exc:
            # Reset any partially-staged freeze changes; the transfer itself is
            # already committed and must NOT be reverted by a freeze failure.
            db.rollback()
            logger.warning(
                f"transfer_direct: FreezeService post-commit re-eval failed for "
                f"new_owner_id={new_owner_id} app_id={app_id} — {freeze_exc}. "
                f"Transfer is committed; freeze state may need manual recalculation."
            )

        return app, previous_owner_id
