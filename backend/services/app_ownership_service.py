"""App ownership transfer service.

Provides the single source of truth for all ``App.owner_id`` reassignment logic.
Four public entry points:

- ``transfer_direct``: Administrative-direct path (omniadmin-initiated).  Used
  standalone via the admin API endpoint and internally by ``UserService.delete_user``
  in ``transfer_apps`` mode.

- ``offer``: Voluntary path — owner/omniadmin creates a PENDING OWNER
  ``AppCollaborator`` row for the intended recipient (AD-5).  No ownership
  change occurs until ``accept`` is called.

- ``accept``: Recipient-only.  Calls ``_reassign_owner``, demotes the previous
  owner to ADMINISTRATOR, and removes the pending offer row.

- ``decline`` / ``cancel``: Recipient declines or owner/omniadmin cancels the
  pending offer.  Sets status to DECLINED (decline) or deletes the row (cancel).

- ``_reassign_owner``: Private primitive called by ``transfer_direct`` and by
  ``accept``.  **Never commits**.  The caller owns the transaction boundary (AD-2).

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

from datetime import datetime
from typing import Tuple

from fastapi import HTTPException as _HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from models.app import App
from models.user import User
from models.app_collaborator import AppCollaborator, CollaborationRole, CollaborationStatus
from repositories.app_collaboration_repository import AppCollaborationRepository
from services.app_ownership_errors import (
    AppNotFoundError,
    NotOfferRecipientError,
    OwnershipOfferNotFoundError,
    TierLimitExceededError,
    TransferRecipientInvalidError,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class AppOwnershipService:
    """Service for app ownership transfer — both administrative-direct and voluntary.

    All methods are synchronous to match the rest of the codebase (UserService,
    AppService, etc. are also sync).  Each method receives ``db`` as its first
    argument rather than storing it at construction time, mirroring the
    ``UserService`` static-method pattern.

    The private ``_reassign_owner`` primitive is the single source of truth for
    all reassignment invariants (recipient validation, tier enforcement, collaborator
    hygiene).  Both ``transfer_direct`` (admin path) and ``accept`` (voluntary path)
    call it — never duplicating the logic.
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

    # ------------------------------------------------------------------ #
    # Public: voluntary transfer — offer / accept / decline / cancel     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def offer(
        db: Session,
        app_id: int,
        new_owner_id: int,
        *,
        actor_user_id: int,
    ) -> AppCollaborator:
        """Create or refresh a pending ownership offer for ``app_id`` to ``new_owner_id``.

        Models the offer as an ``AppCollaborator(role=OWNER, status=PENDING)`` row
        (AD-5 — no new table/migration).  Role resolution ignores collaborator OWNER
        rows until ``accept`` flips ``App.owner_id``, so the offer grants no access.

        This method is idempotent: if a PENDING OWNER offer already exists for the
        same ``(app_id, new_owner_id)`` pair it is refreshed (``invited_by`` and
        ``invited_at`` updated) rather than creating a duplicate.

        Authorization (owner or omniadmin) is enforced at the router via
        ``require_min_role(AppRole.OWNER)``; this method only validates the recipient.

        Args:
            db: Synchronous SQLAlchemy session.
            app_id: PK of the app for which ownership is being offered.
            new_owner_id: PK of the user being offered ownership.
            actor_user_id: PK of the current owner or omniadmin making the offer.

        Returns:
            The created or refreshed ``AppCollaborator`` offer row.

        Raises:
            AppNotFoundError: ``app_id`` does not exist (→ 404).
            TransferRecipientInvalidError: Recipient does not exist, is not active,
                or is already the current owner (→ 400).
        """
        logger.info(
            f"offer: start app_id={app_id} new_owner_id={new_owner_id} "
            f"actor_user_id={actor_user_id}"
        )

        # Load and lock the app row to prevent concurrent offers/transfers.
        app: App | None = db.execute(
            select(App).where(App.app_id == app_id).with_for_update()
        ).scalar_one_or_none()

        if app is None:
            raise AppNotFoundError(app_id)

        # Validate recipient — reuse the same validation checks as _reassign_owner
        # (exists / active / not current owner) but do NOT reassign owner_id here.
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
                f"User {new_owner_id} is already the owner of app {app_id}."
            )

        try:
            # Single active pending OWNER offer per app. Load every pending OWNER
            # offer for this app under a lock: refresh the one addressed to THIS
            # recipient (idempotent), and supersede (delete) any pending offer made
            # to a DIFFERENT recipient — so cancel()/accept() never face multiple
            # competing offers (which would make cancel's scalar_one_or_none raise
            # and let a stale offer still be accepted).
            existing_offers = db.execute(
                select(AppCollaborator)
                .where(
                    AppCollaborator.app_id == app_id,
                    AppCollaborator.role == CollaborationRole.OWNER,
                    AppCollaborator.status == CollaborationStatus.PENDING,
                )
                .with_for_update()
            ).scalars().all()

            same_recipient_offer: AppCollaborator | None = None
            for off in existing_offers:
                if int(off.user_id) == int(new_owner_id):
                    same_recipient_offer = off
                else:
                    db.delete(off)  # supersede a stale offer to a different recipient
                    logger.info(
                        f"offer: superseded prior PENDING OWNER offer collab_id={off.id} "
                        f"app_id={app_id} prior_recipient={off.user_id} actor_user_id={actor_user_id}"
                    )

            if same_recipient_offer is not None:
                # Refresh the existing offer row (idempotent — no duplicate).
                same_recipient_offer.status = CollaborationStatus.PENDING
                same_recipient_offer.invited_by = actor_user_id
                same_recipient_offer.invited_at = datetime.now()
                same_recipient_offer.accepted_at = None
                db.add(same_recipient_offer)
                db.commit()
                db.refresh(same_recipient_offer)
                offer_row = same_recipient_offer
                logger.info(
                    f"offer: refreshed existing PENDING OWNER row collab_id={same_recipient_offer.id} "
                    f"app_id={app_id} new_owner_id={new_owner_id} actor_user_id={actor_user_id}"
                )
            else:
                # A non-OWNER collaboration for this user (if any) is left in place;
                # the OWNER offer is a separate row, cleaned up during accept (via
                # _reassign_owner's collaborator hygiene) or decline/cancel.
                new_collab = AppCollaborator(
                    app_id=app_id,
                    user_id=new_owner_id,
                    role=CollaborationRole.OWNER,
                    status=CollaborationStatus.PENDING,
                    invited_by=actor_user_id,
                    invited_at=datetime.now(),
                )
                db.add(new_collab)
                db.commit()
                db.refresh(new_collab)
                offer_row = new_collab
                logger.info(
                    f"offer: created PENDING OWNER row collab_id={new_collab.id} "
                    f"app_id={app_id} new_owner_id={new_owner_id} actor_user_id={actor_user_id}"
                )
        except Exception:
            db.rollback()
            logger.error(
                f"offer: rollback app_id={app_id} new_owner_id={new_owner_id} "
                f"actor_user_id={actor_user_id}",
                exc_info=True,
            )
            raise

        return offer_row

    @staticmethod
    def accept(
        db: Session,
        collaboration_id: int,
        *,
        actor_user_id: int,
        app_id: int | None = None,
    ) -> Tuple[App, int]:
        """Accept a pending ownership offer (recipient-only).

        Enforces:
        - The ``AppCollaborator`` row exists, has ``role=OWNER``, ``status=PENDING``,
          and ``user_id == actor_user_id`` (only the intended recipient may accept).
        - Calls ``_reassign_owner`` (the single source of truth) which validates the
          recipient again, checks tier limits, stages ``App.owner_id``, and removes
          any prior collaborator row for the recipient (which is the offer row itself
          in this case — no double-delete needed afterward).
        - Demotes the previous owner to an ADMINISTRATOR collaborator (role=ADMINISTRATOR,
          status=ACCEPTED) so they retain access by default after the handover (FR-D3).
        - Commits in one transaction.

        Args:
            db: Synchronous SQLAlchemy session.
            collaboration_id: PK of the ``AppCollaborator`` PENDING OWNER offer row.
            actor_user_id: PK of the user accepting the offer (must equal offer's ``user_id``).

        Returns:
            A tuple ``(app, previous_owner_id)`` — the refreshed ``App`` after the
            transfer and the former owner's PK (captured under the FOR UPDATE lock,
            authoritative under concurrency).

        Raises:
            OwnershipOfferNotFoundError: Row does not exist or is not a PENDING OWNER offer (→ 404).
            NotOfferRecipientError: Actor is not the recipient named in the offer (→ 403).
            TransferRecipientInvalidError: Recipient validation failed (→ 400).
            TierLimitExceededError: Tier limit exceeded for the new owner (→ 409).
        """
        logger.info(
            f"accept: start collaboration_id={collaboration_id} actor_user_id={actor_user_id}"
        )

        # Load and lock the offer row.
        offer: AppCollaborator | None = db.execute(
            select(AppCollaborator)
            .where(AppCollaborator.id == collaboration_id)
            .with_for_update()
        ).scalar_one_or_none()

        if offer is None or offer.role != CollaborationRole.OWNER or offer.status != CollaborationStatus.PENDING:
            raise OwnershipOfferNotFoundError(collaboration_id=collaboration_id)

        # Bind the offer to the {app_id} in the URL (when provided): an offer can only
        # be accepted through its own app's path — prevents cross-app path confusion.
        if app_id is not None and int(offer.app_id) != int(app_id):
            raise OwnershipOfferNotFoundError(collaboration_id=collaboration_id)

        if int(offer.user_id) != int(actor_user_id):
            raise NotOfferRecipientError(actor_user_id, collaboration_id)

        # From here on use the offer's authoritative app_id (equal to the path
        # app_id when that was provided, per the binding check above).
        app_id = offer.app_id
        recipient_id: int = int(offer.user_id)

        # Load and lock the app row.
        app: App | None = db.execute(
            select(App).where(App.app_id == app_id).with_for_update()
        ).scalar_one_or_none()

        if app is None:
            raise AppNotFoundError(app_id)

        previous_owner_id: int = app.owner_id

        try:
            # _reassign_owner validates the recipient, checks the tier limit, stages
            # app.owner_id = recipient_id, and removes any pre-existing collaborator row
            # for the recipient via collab_repo.get_collaboration_by_app_and_user (FR-C5).
            #
            # IMPORTANT: because the offer row has user_id=recipient_id on this app,
            # _reassign_owner's collaborator-hygiene step will find and db.delete()
            # the offer row itself.  We therefore must NOT call db.delete(offer) again
            # after this point — doing so would attempt a double-delete on an already
            # pending-delete ORM object.
            #
            # This is the correct, safe path: _reassign_owner is the single source of
            # truth for "recipient's collaborator rows are cleared", and it clears the
            # offer row as part of that hygiene.  The offer is gone after this call.
            AppOwnershipService._reassign_owner(
                db, app, recipient_id, actor_user_id=actor_user_id
            )

            # Demote the previous owner to ADMINISTRATOR so they retain access (FR-D3).
            # We query AFTER the _reassign_owner flush so that we see any rows that
            # were already committed for the previous owner.  Note: the previous owner
            # cannot be the recipient (validated above), so there is no conflict.
            collab_repo = AppCollaborationRepository(db)
            prev_owner_collab: AppCollaborator | None = (
                collab_repo.get_collaboration_by_app_and_user(app_id, previous_owner_id)
            )

            if prev_owner_collab is not None:
                # Reuse / update the existing row to ADMINISTRATOR / ACCEPTED.
                prev_owner_collab.role = CollaborationRole.ADMINISTRATOR
                prev_owner_collab.status = CollaborationStatus.ACCEPTED
                prev_owner_collab.accepted_at = datetime.now()
                db.add(prev_owner_collab)
                logger.info(
                    f"accept: updated previous owner row collab_id={prev_owner_collab.id} "
                    f"to ADMINISTRATOR/ACCEPTED app_id={app_id} user_id={previous_owner_id}"
                )
            else:
                # Create a new ADMINISTRATOR row for the previous owner.
                demoted_collab = AppCollaborator(
                    app_id=app_id,
                    user_id=previous_owner_id,
                    role=CollaborationRole.ADMINISTRATOR,
                    status=CollaborationStatus.ACCEPTED,
                    invited_by=actor_user_id,
                    invited_at=datetime.now(),
                    accepted_at=datetime.now(),
                )
                db.add(demoted_collab)
                logger.info(
                    f"accept: created ADMINISTRATOR row for previous owner user_id={previous_owner_id} "
                    f"app_id={app_id} actor_user_id={actor_user_id}"
                )

            db.commit()
            db.refresh(app)
        except Exception:
            db.rollback()
            logger.error(
                f"accept: rollback collaboration_id={collaboration_id} "
                f"actor_user_id={actor_user_id}",
                exc_info=True,
            )
            raise

        logger.info(
            f"accept: complete app_id={app_id} previous_owner_id={previous_owner_id} "
            f"new_owner_id={recipient_id} actor_user_id={actor_user_id}"
        )

        return app, previous_owner_id

    @staticmethod
    def decline(
        db: Session,
        collaboration_id: int,
        *,
        actor_user_id: int,
        app_id: int | None = None,
    ) -> None:
        """Decline a pending ownership offer (recipient-only).

        Sets the offer row status to DECLINED.  Matches the existing collaboration
        decline behaviour (status flip rather than row deletion) so the inviter can
        see the offer was explicitly declined.

        Args:
            db: Synchronous SQLAlchemy session.
            collaboration_id: PK of the ``AppCollaborator`` PENDING OWNER offer row.
            actor_user_id: PK of the user declining (must equal offer's ``user_id``).

        Raises:
            OwnershipOfferNotFoundError: Row does not exist or is not a PENDING OWNER offer (→ 404).
            NotOfferRecipientError: Actor is not the recipient named in the offer (→ 403).
        """
        logger.info(
            f"decline: start collaboration_id={collaboration_id} actor_user_id={actor_user_id}"
        )

        offer: AppCollaborator | None = db.execute(
            select(AppCollaborator)
            .where(AppCollaborator.id == collaboration_id)
            .with_for_update()
        ).scalar_one_or_none()

        if offer is None or offer.role != CollaborationRole.OWNER or offer.status != CollaborationStatus.PENDING:
            raise OwnershipOfferNotFoundError(collaboration_id=collaboration_id)

        if app_id is not None and int(offer.app_id) != int(app_id):
            raise OwnershipOfferNotFoundError(collaboration_id=collaboration_id)

        if int(offer.user_id) != int(actor_user_id):
            raise NotOfferRecipientError(actor_user_id, collaboration_id)

        try:
            offer.status = CollaborationStatus.DECLINED
            db.add(offer)
            db.commit()
        except Exception:
            db.rollback()
            logger.error(
                f"decline: rollback collaboration_id={collaboration_id} "
                f"actor_user_id={actor_user_id}",
                exc_info=True,
            )
            raise

        logger.info(
            f"decline: complete collaboration_id={collaboration_id} "
            f"app_id={offer.app_id} actor_user_id={actor_user_id}"
        )

    @staticmethod
    def cancel(
        db: Session,
        app_id: int,
        *,
        actor_user_id: int,
    ) -> None:
        """Cancel the pending ownership offer for an app (owner/omniadmin only).

        Deletes the PENDING OWNER ``AppCollaborator`` row.  Authorization
        (owner or omniadmin) is enforced at the router via
        ``require_min_role(AppRole.OWNER)``; this method only locates and removes
        the offer row.

        Args:
            db: Synchronous SQLAlchemy session.
            app_id: PK of the app whose pending offer should be cancelled.
            actor_user_id: PK of the owner or omniadmin performing the cancellation.

        Raises:
            OwnershipOfferNotFoundError: No PENDING OWNER offer exists for this app (→ 404).
        """
        logger.info(
            f"cancel: start app_id={app_id} actor_user_id={actor_user_id}"
        )

        # Find ALL pending OWNER offer rows for this app. offer() enforces a single
        # active offer, but deleting every match here is robust against any legacy
        # multi-offer rows and avoids a MultipleResultsFound on scalar_one_or_none.
        offers = db.execute(
            select(AppCollaborator)
            .where(
                AppCollaborator.app_id == app_id,
                AppCollaborator.role == CollaborationRole.OWNER,
                AppCollaborator.status == CollaborationStatus.PENDING,
            )
            .with_for_update()
        ).scalars().all()

        if not offers:
            raise OwnershipOfferNotFoundError(app_id=app_id)

        try:
            for off in offers:
                db.delete(off)
            db.commit()
        except Exception:
            db.rollback()
            logger.error(
                f"cancel: rollback app_id={app_id} actor_user_id={actor_user_id}",
                exc_info=True,
            )
            raise

        logger.info(
            f"cancel: complete app_id={app_id} cancelled_count={len(offers)} "
            f"actor_user_id={actor_user_id}"
        )
