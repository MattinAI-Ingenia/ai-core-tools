from sqlalchemy.orm import Session
from sqlalchemy import select
from models.user import User, PlatformRole
from models.api_key import APIKey
from models.app import App
from repositories.user_repository import UserRepository
from repositories.app_repository import AppRepository
from repositories.app_collaboration_repository import AppCollaborationRepository
from services.user_deletion_errors import (
    OmniadminDeletionError,
    OwnedAppsPresentError,
    OwnedAppSummary,
    SelfDeletionError,
    UserNotFoundError,
)
from services.app_ownership_errors import TransferRecipientInvalidError, TierLimitExceededError
from typing import Literal, Optional, Tuple, List, Dict, Any
from utils.config import is_omniadmin
from utils.logger import get_logger

class UserService:
    
    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def _user_to_dict(user: User, include_full_details: bool = True) -> Dict[str, Any]:
        """
        Convert User object to dictionary format
        
        Args:
            user: User instance to convert
            include_full_details: If True, includes all fields (apps, keys, etc.)
                                 If False, includes only basic fields
            
        Returns:
            Dictionary representation of the user
        """
        user_dict = {
            'user_id': user.user_id,
            'email': user.email,
            'name': user.name,
            'created_at': user.create_date.isoformat() if user.create_date else None,
        }
        
        if include_full_details:
            user_dict.update({
                'owned_apps_count': len(user.owned_apps) if user.owned_apps else 0,
                'api_keys_count': len(user.api_keys) if user.api_keys else 0,
                'is_active': user.is_active if hasattr(user, 'is_active') else True,
                'is_omniadmin': is_omniadmin(user.email),
                'platform_role': user.platform_role if user.platform_role else 'editor',
            })
        
        return user_dict
    
    @staticmethod
    def _get_user_or_raise(db: Session, user_id: int) -> User:
        """
        Get user by ID or raise ValueError if not found
        
        Args:
            db: Database session
            user_id: ID of the user
            
        Returns:
            User instance
            
        Raises:
            ValueError: If user not found
        """
        user_repo = UserRepository(db)
        user = user_repo.get_by_id(user_id)
        
        if not user:
            raise ValueError("User not found")
        
        return user
    
    # ============================================================================
    # PUBLIC METHODS
    # ============================================================================
    
    @staticmethod
    def get_or_create_user(db: Session, email: str, name: str = None) -> Tuple[User, bool]:
        """
        Get existing user or create new user if doesn't exist.
        Returns: (user, created) where created is True if user was just created
        """
        user_repo = UserRepository(db)
        
        # Try to find existing user
        user = user_repo.get_by_email(email)
        
        if user:
            # User exists, update name if provided and different
            user = user_repo.update(user, name)
            return user, False
        
        # Create new user
        new_user = user_repo.create(email, name)
        return new_user, True
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> User:
        """Get user by ID"""
        user_repo = UserRepository(db)
        return user_repo.get_by_id(user_id)
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> User:
        """Get user by email"""
        user_repo = UserRepository(db)
        return user_repo.get_by_email(email)
    
    @staticmethod
    def get_user_accessible_apps(user_id: int):
        """Get all apps user has access to (owned + collaborated)"""
        # TODO: Implement this method
        pass
    
    # ============================================================================
    # ADMIN METHODS
    # ============================================================================
    
    @staticmethod
    def get_all_users(db: Session, page: int = 1, per_page: int = 10) -> Tuple[List[Dict], int]:
        """
        Get all users with pagination
        
        Args:
            db: Database session
            page: Page number (1-based)
            per_page: Number of users per page
            
        Returns:
            Tuple of (users_list, total_count)
        """
        user_repo = UserRepository(db)
        users, total = user_repo.get_all_paginated(page, per_page)
        
        # Convert to dict format
        users_list = [UserService._user_to_dict(user) for user in users]
        
        return users_list, total
    
    @staticmethod
    def get_user_by_id_with_relations(db: Session, user_id: int) -> User:
        """
        Get user by ID with related data
        
        Args:
            db: Database session
            user_id: ID of the user
            
        Returns:
            User instance or None if not found
        """
        user_repo = UserRepository(db)
        return user_repo.get_by_id_with_relations(user_id)
    
    @staticmethod
    def search_users(db: Session, query: str, page: int = 1, per_page: int = 10) -> Tuple[List[Dict], int]:
        """
        Search users by name or email
        
        Args:
            db: Database session
            query: Search query
            page: Page number (1-based)
            per_page: Number of users per page
            
        Returns:
            Tuple of (users_list, total_count)
        """
        user_repo = UserRepository(db)
        users, total = user_repo.search_users(query, page, per_page)
        
        # Convert to dict format
        users_list = [UserService._user_to_dict(user) for user in users]
        
        return users_list, total
    
    @staticmethod
    def delete_user(
        db: Session,
        user_id: int,
        *,
        actor_user_id: int,
        mode: Literal["block", "cascade_apps", "transfer_apps"] = "block",
        transfer_to_user_id: Optional[int] = None,
    ) -> bool:
        """Delete a user with ordered, safe orchestration.

        Mirrors the structure of ``AppService.delete_app`` and implements the
        four-class deletion taxonomy defined in AD-1 of the user-deletion spec:

        - **Class A** (credentials, refresh_tokens, subscription, usage): handled
          by DB-level ``ON DELETE CASCADE``; ``passive_deletes=True`` on the ORM
          relationships prevents the ORM from emitting a conflicting SET NULL.
        - **Class B** (owned apps): handled explicitly by ``AppService.delete_app``
          (cascade) or the transfer service (transfer); never a DB cascade.
        - **Class C** (collaboration memberships, API keys in non-owned apps):
          removed explicitly in this method before ``db.delete(user)``.
        - **Class D** (invited_by, Conversation.user_id, crawl_job.triggered_by):
          handled by DB-level ``ON DELETE SET NULL``; ORM does not touch them.

        Atomicity (AD-2):
        - ``mode='cascade_apps'``: each ``AppService.delete_app`` call commits
          per-app (irreversible vector-store drops). The user row is deleted LAST
          so a mid-failure leaves the user present with any remaining owned apps
          intact — the operation is resumable.
        - ``mode='transfer_apps'``: step_004 will wire this as a single
          transaction. The stub raises ``NotImplementedError`` until then.
        - The terminal phase (class-C cleanup + ``db.delete(user)`` + commit) is
          always a single Postgres transaction.

        Concurrency (AD-6):
        - A ``SELECT … FOR UPDATE`` row lock on the target ``User`` row is acquired
          at entry so that two concurrent delete requests serialise cleanly.

        Args:
            db: Synchronous SQLAlchemy session (caller owns the transaction
                boundary for the terminal phase).
            user_id: Primary key of the user to delete.
            actor_user_id: Primary key of the administrator performing the action.
                Used for self-delete guardrail and structured logging.
            mode: Deletion strategy for owned apps.
                ``'block'`` (default) — reject with ``OwnedAppsPresentError`` if
                the user owns any apps; no mutation occurs.
                ``'cascade_apps'`` — call ``AppService.delete_app`` for every
                owned app, then delete the user.
                ``'transfer_apps'`` — placeholder; raises ``NotImplementedError``
                until step_004 wires ``AppOwnershipService.transfer_direct``.
            transfer_to_user_id: Target owner for ``mode='transfer_apps'``.
                Ignored for other modes (reserved for step_004).

        Returns:
            ``True`` on successful deletion.

        Raises:
            UserNotFoundError: Target ``user_id`` does not exist (→ 404).
            SelfDeletionError: ``actor_user_id == user_id`` (→ 400).
            OmniadminDeletionError: Target user is an omniadmin (→ 403).
            OwnedAppsPresentError: User owns apps and ``mode='block'`` (→ 409).
            NotImplementedError: ``mode='transfer_apps'`` is not yet wired
                (step_004 will replace this with real logic).
        """
        # AppService is imported lazily to avoid a circular import
        # (app_service → child services → user_service). user_deletion_errors has
        # no such cycle and is imported at module top-level.
        from services.app_service import AppService

        logger = get_logger(__name__)

        # ------------------------------------------------------------------
        # Step 1 — Acquire row lock + load the target user (AD-6).
        #
        # Using SQLAlchemy 2.x select() + with_for_update() matches the pattern
        # established by refresh_token_repository.get_by_jti_for_update.
        # ------------------------------------------------------------------
        user: Optional[User] = db.execute(
            select(User).where(User.user_id == user_id).with_for_update()
        ).scalar_one_or_none()

        if user is None:
            logger.warning(
                f"delete_user: target not found — user_id={user_id}, actor_user_id={actor_user_id}"
            )
            raise UserNotFoundError(user_id)

        # Capture immutable scalars up front: the per-app commits inside
        # AppService.delete_app (cascade_apps) expire this ORM instance
        # (expire_on_commit=True), so any later attribute read would trigger an
        # implicit reload. Read email once here and use the local thereafter.
        user_email: str = user.email

        # ------------------------------------------------------------------
        # Step 2 — Guardrails (no mutation on rejection). Logged at WARNING.
        # ------------------------------------------------------------------
        if actor_user_id == user_id:
            logger.warning(
                f"delete_user: self-deletion rejected — user_id={user_id}, actor_user_id={actor_user_id}"
            )
            raise SelfDeletionError(user_id)

        if is_omniadmin(user_email):
            logger.warning(
                f"delete_user: omniadmin deletion rejected — user_id={user_id}, actor_user_id={actor_user_id}"
            )
            raise OmniadminDeletionError(user_email)

        # Only now is the deletion actually proceeding — emit the start log after
        # the guardrails pass so a rejected request never looks like an initiated
        # deletion in the audit trail (NFR-6).
        logger.info(
            f"delete_user: start — user_id={user_id} mode={mode} actor_user_id={actor_user_id}"
        )

        # ------------------------------------------------------------------
        # Step 3 — Owned-apps gate (Class B).
        #
        # AppRepository.get_by_owner(user_id) returns List[App] ordered by
        # create_date DESC.  We fetch once, build summaries, and branch.
        # ------------------------------------------------------------------
        app_repo = AppRepository(db)
        owned_apps = app_repo.get_by_owner(user_id)

        if owned_apps:
            owned_app_summaries: List[OwnedAppSummary] = [
                {"app_id": app.app_id, "name": app.name} for app in owned_apps
            ]

            if mode == "block":
                logger.warning(
                    f"delete_user: owned-apps-present block — user_id={user_id} "
                    f"owned_app_count={len(owned_apps)} actor_user_id={actor_user_id}"
                )
                raise OwnedAppsPresentError(owned_app_summaries)

            if mode == "transfer_apps":
                # AD-2: transfer path is FULLY ATOMIC — all reassignments plus the
                # terminal user deletion run as a single Postgres transaction.
                # We call AppOwnershipService._reassign_owner (the private primitive)
                # directly — NOT transfer_direct — because transfer_direct commits
                # and would break the single-transaction guarantee.

                if transfer_to_user_id is None:
                    raise TransferRecipientInvalidError(
                        "transfer_to_user_id must be provided when mode='transfer_apps'."
                    )

                if transfer_to_user_id == user_id:
                    raise TransferRecipientInvalidError(
                        "transfer_to_user_id cannot be the same user that is being deleted."
                    )

                # Validate the recipient ONCE up front (clean error before any mutation).
                # _reassign_owner also validates per-app, but doing it here lets us fail
                # fast before touching any App row.
                recipient = db.execute(
                    select(User).where(User.user_id == transfer_to_user_id)
                ).scalar_one_or_none()

                if recipient is None:
                    raise TransferRecipientInvalidError(
                        f"User {transfer_to_user_id} does not exist."
                    )

                if not recipient.is_active:
                    raise TransferRecipientInvalidError(
                        f"User {transfer_to_user_id} is not active and cannot receive app ownership."
                    )

                # Lazy import to avoid a circular import (same reason as AppService below).
                from services.app_ownership_service import AppOwnershipService

                # Re-fetch the owned apps WITH row locks (AD-6). The summary list
                # above came from a plain get_by_owner() read holding no lock;
                # _reassign_owner mutates owner_id, so we must lock each App row to
                # stop a concurrent transfer_direct from racing a double-reassignment.
                locked_owned_apps = db.execute(
                    select(App).where(App.owner_id == user_id).with_for_update()
                ).scalars().all()

                # AD-2: this whole loop must be atomic with the terminal phase. It
                # runs BEFORE the Step-4 try/except, so guard it here: any failure
                # mid-loop must roll back ALL staged transfers (not leave apps
                # 1..N-1 reassigned) before propagating — the service stays atomic
                # regardless of whether the caller owns the session lifecycle.
                try:
                    for app in locked_owned_apps:
                        logger.info(
                            f"delete_user(transfer_apps): staging transfer of app_id={app.app_id} "
                            f"'{app.name}' to user_id={transfer_to_user_id} for user_id={user_id}"
                        )
                        # _reassign_owner stages changes but does NOT commit — the entire
                        # loop plus the terminal phase below form one atomic transaction.
                        AppOwnershipService._reassign_owner(
                            db,
                            app,
                            transfer_to_user_id,
                            actor_user_id=actor_user_id,
                        )
                except Exception:
                    db.rollback()
                    logger.error(
                        f"delete_user(transfer_apps): rollback — staged transfers reverted "
                        f"for user_id={user_id} recipient={transfer_to_user_id} actor_user_id={actor_user_id}",
                        exc_info=True,
                    )
                    raise
                # Fall through to the terminal phase (Step 4) which performs the
                # single db.commit() covering all staged transfers + user deletion.

            if mode == "cascade_apps":
                # Each delete_app call commits per-app internally (AD-2). The user
                # row is deleted LAST so a failure mid-loop leaves the user intact
                # with only the already-deleted apps gone (resumable).
                app_svc = AppService(db)
                for app in owned_apps:
                    app_id_to_delete = app.app_id
                    app_name = app.name
                    logger.info(
                        f"delete_user(cascade_apps): deleting owned app {app_id_to_delete} "
                        f"'{app_name}' for user_id={user_id}"
                    )
                    success = app_svc.delete_app(app_id_to_delete)
                    if not success:
                        # AppService.delete_app swallows real failures and returns
                        # False for BOTH "already gone" and "failed mid-cascade".
                        # Distinguish them: if the app still exists, this was a real
                        # failure — abort BEFORE db.delete(user) so the app is not
                        # orphaned and the operation stays resumable (AD-2/AC-13).
                        if app_repo.get_by_id(app_id_to_delete) is not None:
                            logger.error(
                                f"delete_user(cascade_apps): delete_app failed for "
                                f"app_id={app_id_to_delete}; aborting before user deletion "
                                f"to avoid orphaning the app."
                            )
                            raise RuntimeError(
                                f"Failed to delete owned app {app_id_to_delete} during "
                                f"cascade deletion of user {user_id}."
                            )
                        logger.warning(
                            f"delete_user(cascade_apps): app_id={app_id_to_delete} already "
                            f"absent — treating as concurrently removed and continuing"
                        )

        # ------------------------------------------------------------------
        # Step 4 — Terminal phase: Class-C explicit cleanup + db.delete(user).
        #
        # At this point the user owns zero apps (either had none, or cascade_apps
        # removed them). Class C — access grants:
        #   a) AppCollaborator rows where user_id == target (membership in others' apps).
        #   b) APIKey rows owned by the target in apps the user does NOT own
        #      (keys in owned apps were already removed by delete_app step 11).
        # Class D (invited_by, Conversation.user_id, crawl_job rows) is handled by
        # DB ON DELETE SET NULL — the ORM must NOT touch those columns.
        # ------------------------------------------------------------------
        try:
            # Re-acquire the row lock and a fresh instance. In cascade_apps the
            # per-app commits released the entry lock and expired `user`; re-locking
            # closes the delete-races-mutation window for db.delete(user) and yields
            # a clean UserNotFoundError if the user was concurrently removed (AD-6).
            user = db.execute(
                select(User).where(User.user_id == user_id).with_for_update()
            ).scalar_one_or_none()
            if user is None:
                raise UserNotFoundError(user_id)

            # --- 4a: Remove collaboration memberships (Class C) ---
            collab_repo = AppCollaborationRepository(db)
            memberships = collab_repo.get_collaborations_by_user(user_id)
            for membership in memberships:
                logger.info(
                    f"delete_user: removing collaboration id={membership.id} "
                    f"app_id={membership.app_id} user_id={user_id}"
                )
                db.delete(membership)

            db.flush()  # Ensure memberships are gone before we inspect API keys.

            # --- 4b: Remove surviving API keys (Class C) ---
            # Queried directly: APIKeyService.delete_api_key is scoped by app_id and
            # would need a per-key lookup. Owned-app keys are already removed by
            # delete_app (step 11), so this catches only keys held in other apps.
            surviving_keys = db.execute(
                select(APIKey).where(APIKey.user_id == user_id)
            ).scalars().all()
            for api_key in surviving_keys:
                logger.info(
                    f"delete_user: removing APIKey key_id={api_key.key_id} "
                    f"app_id={api_key.app_id} user_id={user_id}"
                )
                db.delete(api_key)

            db.flush()  # Ensure FK dependents are gone before deleting the user row.

            # --- 4c: Delete the user row ---
            # DB ON DELETE CASCADE handles: user_credentials, refresh_tokens,
            #   subscription, usage_records, MarketplaceUsage, AgentMarketplaceRating.
            # DB ON DELETE SET NULL handles: AppCollaborator.invited_by,
            #   Conversation.user_id, crawl_job.triggered_by_user_id.
            # passive_deletes=True on all mapped relationships prevents ORM SET NULL.
            db.delete(user)
            db.commit()

            logger.info(
                f"delete_user: complete — user_id={user_id} mode={mode} actor_user_id={actor_user_id}"
            )
            return True

        except Exception:
            db.rollback()
            logger.error(
                f"delete_user: rollback — user_id={user_id} mode={mode} "
                f"actor_user_id={actor_user_id}",
                exc_info=True,
            )
            raise
    
    @staticmethod
    def get_user_stats(db: Session) -> Dict[str, Any]:
        """
        Get system-wide user statistics
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with user statistics
        """
        user_repo = UserRepository(db)
        
        # Total users count
        total_users = user_repo.get_total_count()
        
        # Recent users (last 30 days)
        recent_users = user_repo.get_recent_users_count(30)
        
        # Users with apps
        users_with_apps = user_repo.get_users_with_apps_count()
        
        # Recent users list (last 10)
        recent_users_list = user_repo.get_recent_users_list(30, 10)
        
        recent_users_data = [
            UserService._user_to_dict(user, include_full_details=False) 
            for user in recent_users_list
        ]
        
        return {
            'total_users': total_users,
            'recent_users': recent_users,
            'users_with_apps': users_with_apps,
            'recent_users_list': recent_users_data
        }
    
    @staticmethod
    def activate_user(db: Session, user_id: int, admin_email: str) -> User:
        """
        Activate a user account
        
        Args:
            db: Database session
            user_id: ID of the user to activate
            admin_email: Email of the admin performing the action
            
        Returns:
            Updated User instance
            
        Raises:
            ValueError: If user cannot be activated
        """
        logger = get_logger(__name__)
        user = UserService._get_user_or_raise(db, user_id)
        
        if user.is_active:
            raise ValueError("User is already active")
        
        # Activate the user
        user.is_active = True
        db.commit()
        db.refresh(user)
        
        logger.info(f"User activated - Admin: {admin_email}, Target User: {user.email} (ID: {user_id})")
        
        return user
    
    @staticmethod
    def deactivate_user(db: Session, user_id: int, admin_email: str) -> User:
        """
        Deactivate a user account
        
        Args:
            db: Database session
            user_id: ID of the user to deactivate
            admin_email: Email of the admin performing the action
            
        Returns:
            Updated User instance
            
        Raises:
            ValueError: If user cannot be deactivated
        """
        logger = get_logger(__name__)
        user = UserService._get_user_or_raise(db, user_id)
        
        if not user.is_active:
            raise ValueError("User is already inactive")
        
        # Prevent deactivating admin users
        if is_omniadmin(user.email):
            raise ValueError("Cannot deactivate admin users")
        
        # Prevent self-deactivation
        if user.email == admin_email:
            raise ValueError("Cannot deactivate your own account")
        
        # Deactivate the user
        user.is_active = False
        db.commit()
        db.refresh(user)
        
        logger.info(f"User deactivated - Admin: {admin_email}, Target User: {user.email} (ID: {user_id})")
        
        return user
    
    @staticmethod
    def set_platform_role(db: Session, user_id: int, role: str, admin_email: str) -> dict:
        from models.app import App
        logger = get_logger(__name__)
        valid_roles = {r.value for r in PlatformRole}
        if role not in valid_roles:
            raise ValueError(f"Invalid platform role '{role}'. Must be one of: {', '.join(valid_roles)}")

        user = UserService._get_user_or_raise(db, user_id)

        if user.email == admin_email:
            raise ValueError("Cannot change your own platform role")

        user.platform_role = role
        db.commit()
        db.refresh(user)

        logger.info(f"Platform role set to '{role}' - Admin: {admin_email}, Target: {user.email} (ID: {user_id})")

        warnings: List[str] = []
        if role == 'viewer':
            owned = db.query(App).filter(App.owner_id == user_id).count()
            if owned:
                warnings.append(f"User owns {owned} app(s). They retain ownership but cannot modify them while a viewer.")

        return {"user": user, "warnings": warnings}

    @staticmethod
    def get_active_users_count(db: Session) -> int:
        """
        Get count of active users
        
        Args:
            db: Database session
            
        Returns:
            Count of active users
        """
        return db.query(User).filter(User.is_active == True).count()
    
    @staticmethod
    def get_inactive_users_count(db: Session) -> int:
        """
        Get count of inactive users
        
        Args:
            db: Database session
            
        Returns:
            Count of inactive users
        """
        return db.query(User).filter(User.is_active == False).count() 