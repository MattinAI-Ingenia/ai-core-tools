"""Credential lifecycle service for LOCAL auth mode.

This module is the single source of truth for password hashing, account
lockout, set-password tokens, and the authenticate/change-password flows used
exclusively by LOCAL auth endpoints.

It is intentionally decoupled from:
- SaaS subscription logic (no SubscriptionRepository imports).
- HTTP concerns (no FastAPI, no HTTPException — raises typed domain exceptions
  that the HTTP layer maps to status codes).

Concurrency / event-loop safety
--------------------------------
bcrypt hashing is CPU-bound and takes ~250 ms.  All hash and verify operations
are dispatched to a thread-pool via ``fastapi.concurrency.run_in_threadpool``
so the async event loop is never blocked.

Token storage strategy
-----------------------
Set-password tokens (first-time setup + admin-triggered resets) reuse the
``reset_token`` / ``reset_token_expiry`` columns on ``UserCredential`` with a
distinct itsdangerous salt ``_TOKEN_SALT_SET_PASSWORD``.  On consumption the
columns are cleared, making the token single-use.  Expiry defaults to 48 hours
(configurable via ``LOCAL_SET_PASSWORD_TOKEN_MAX_AGE_HOURS``).

Lockout formula
---------------
Threshold: ``LOCAL_LOCKOUT_THRESHOLD`` (default 5 failed attempts).
Base backoff: ``LOCAL_LOCKOUT_BASE_SECONDS`` (default 60 s).
Exponent: ``failed_attempts - threshold``.

    locked_until = now + base * 2^(failed_attempts - threshold)

Examples (threshold=5, base=60):
  - 5th fail:  60 * 2^0 = 60 s
  - 6th fail:  60 * 2^1 = 120 s
  - 7th fail:  60 * 2^2 = 240 s
  - 10th fail: 60 * 2^5 = 1920 s (~32 min)

Omniadmins (``is_omniadmin(email)``) are exempt from lockout so a platform
administrator cannot be locked out of their own instance.

SMTP-off verification bypass (FR-C6/AC-4)
------------------------------------------
When ``smtp_configured()`` returns False (no outbound email) the
``is_verified`` gate on login is skipped.  This lets self-hosted deployments
(no SMTP) work without the verification handshake.  When SMTP is configured
the gate is enforced.
"""

import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi.concurrency import run_in_threadpool
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from models.user import User
from repositories.user_credential_repository import UserCredentialRepository
from repositories.user_repository import UserRepository
from schemas.local_auth_schemas import check_password_not_email
from services.auth.refresh_service import RefreshService
from services.email import smtp_configured
from utils.config import Config, is_omniadmin
from utils.logger import get_logger
from utils.secret_key import get_secret_key

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOCKOUT_THRESHOLD: int = Config.get_int_env_var("LOCAL_LOCKOUT_THRESHOLD", default=5)
_LOCKOUT_BASE_SECONDS: int = Config.get_int_env_var("LOCAL_LOCKOUT_BASE_SECONDS", default=60)

_TOKEN_SALT_SET_PASSWORD = "local-set-password-v1"
_SET_PASSWORD_TOKEN_MAX_AGE_SECONDS: int = (
    Config.get_int_env_var("LOCAL_SET_PASSWORD_TOKEN_MAX_AGE_HOURS", default=48) * 3600
)

# A fixed dummy hash evaluated against unknown-email login attempts to keep
# response timing uniform and prevent user-enumeration via timing.  Generated
# once at module import time so the cost is paid once, not per request.
_DUMMY_HASH: str = bcrypt.hashpw(b"__dummy_sentinel__", bcrypt.gensalt(rounds=12)).decode("utf-8")


# ---------------------------------------------------------------------------
# Domain exceptions (HTTP layer maps these to status codes)
# ---------------------------------------------------------------------------


class CredentialError(Exception):
    """Generic credential failure.  Maps to HTTP 401 Unauthorized."""


class AccountLockedError(Exception):
    """Account temporarily locked after excessive failures.  Maps to HTTP 429.

    Attributes:
        locked_until: UTC-aware datetime at which the lockout expires.  Callers
            should use this to populate a ``Retry-After`` response header.
            ``None`` only when the lockout timestamp is unavailable (should not
            occur in normal operation).
    """

    def __init__(self, message: str, locked_until: Optional[datetime] = None) -> None:
        super().__init__(message)
        self.locked_until: Optional[datetime] = locked_until


class TokenError(Exception):
    """Set-password token is invalid, expired, or already used.  Maps to HTTP 400."""


class UserAlreadyExistsError(Exception):
    """Admin create: email already registered.  Maps to HTTP 409 Conflict."""


class PasswordPolicyError(Exception):
    """Password rejected by policy (e.g. matches email local-part).

    Intentionally NOT a subclass of ``CredentialError`` so that HTTP handlers
    can distinguish a policy rejection (400) from an authentication failure
    (401) without relying on exception message text.  Maps to HTTP 400.
    """


class InactiveAccountError(CredentialError):
    """Account is deactivated.  Maps to HTTP 403 Forbidden."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_secret_key())


def _sync_hash(plain: str) -> str:
    """Synchronous bcrypt hash (runs in thread-pool via ``hash_password``)."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def _sync_verify(plain: str, hashed: str) -> bool:
    """Synchronous bcrypt verify (runs in thread-pool via ``verify_password``)."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# Public async helpers
# ---------------------------------------------------------------------------


async def hash_password(plain: str) -> str:
    """Hash a plaintext password off the event loop.

    Args:
        plain: The candidate password string.

    Returns:
        bcrypt hash string suitable for storage.
    """
    return await run_in_threadpool(_sync_hash, plain)


async def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash off the event loop.

    Args:
        plain: The candidate password string.
        hashed: The stored bcrypt hash.

    Returns:
        True when the password matches, False otherwise.
    """
    return await run_in_threadpool(_sync_verify, plain, hashed)


# ---------------------------------------------------------------------------
# Lockout helpers
# ---------------------------------------------------------------------------


def _compute_lockout_duration(failed_attempts: int) -> timedelta:
    """Compute exponential backoff lockout duration.

    Args:
        failed_attempts: The current failed-attempt counter (already incremented
            to the value that triggered the lock).

    Returns:
        The ``timedelta`` to add to ``now`` to set ``locked_until``.
    """
    exponent = min(max(0, failed_attempts - _LOCKOUT_THRESHOLD), 10)
    seconds = _LOCKOUT_BASE_SECONDS * (2 ** exponent)
    return timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# CredentialService
# ---------------------------------------------------------------------------


class CredentialService:
    """Hardened credential lifecycle service for LOCAL auth mode.

    All methods are async.  The synchronous DB session is used directly (as per
    the project pattern in other services) — only the CPU-bound bcrypt calls
    are dispatched to a thread-pool.
    """

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    async def authenticate(db: Session, email: str, password: str) -> User:
        """Authenticate a user by email and password.

        Implements anti-enumeration (always runs a bcrypt verify even when the
        user is not found), exponential lockout on repeated failures, and the
        SMTP-off verification bypass.

        Args:
            db: Active database session.
            email: The submitted email address.
            password: The submitted plaintext password.

        Returns:
            The authenticated ``User`` ORM instance.

        Raises:
            AccountLockedError: When the account is locked out.
            CredentialError: For all other credential failures (wrong password,
                unknown user, unverified account, inactive account).  The same
                exception type is used deliberately to prevent distinguishing
                failure modes from the outside.
        """
        user_repo = UserRepository(db)
        cred_repo = UserCredentialRepository(db)

        user: Optional[User] = user_repo.get_by_email(email)

        if user is None or user.auth_method != "local":
            # Run a dummy verify to keep timing uniform even when the user does
            # not exist, preventing enumeration via response time.
            await run_in_threadpool(_sync_verify, password, _DUMMY_HASH)
            logger.warning("auth:login_failed reason=unknown_user email=%s", email)
            raise CredentialError("Invalid email or password.")

        cred = cred_repo.get_by_user_id(user.user_id)
        if cred is None:
            await run_in_threadpool(_sync_verify, password, _DUMMY_HASH)
            logger.warning("auth:login_failed reason=no_credential user_id=%s", user.user_id)
            raise CredentialError("Invalid email or password.")

        # Lockout check — omniadmins are exempt.
        # IMPORTANT: This exemption prevents an admin self-lockout DoS.  It REQUIRES
        # the route-level per-IP + per-account login rate limit landing in step_007;
        # without that rate limit, omniadmin accounts have no brute-force protection.
        if not is_omniadmin(email):
            now = datetime.now(timezone.utc)
            locked = cred.locked_until
            if locked is not None and locked.tzinfo is None:
                locked = locked.replace(tzinfo=timezone.utc)
            if locked is not None and locked > now:
                logger.warning(
                    "auth:login_rejected reason=account_locked user_id=%s locked_until=%s",
                    user.user_id,
                    cred.locked_until.isoformat(),
                )
                raise AccountLockedError(
                    "Account is temporarily locked due to too many failed attempts. "
                    "Please try again later.",
                    locked_until=locked,  # UTC-aware; used for Retry-After header
                )

        # Verify password off the event loop.
        password_ok: bool = await run_in_threadpool(_sync_verify, password, cred.hashed_password)

        if not password_ok:
            # Increment failed_attempts and potentially set lockout.
            if cred.failed_attempts is None:
                cred.failed_attempts = 0
            cred.failed_attempts += 1

            if not is_omniadmin(email) and cred.failed_attempts >= _LOCKOUT_THRESHOLD:
                duration = _compute_lockout_duration(cred.failed_attempts)
                cred.locked_until = datetime.now(timezone.utc) + duration
                logger.warning(
                    "auth:account_locked user_id=%s failed_attempts=%s locked_until=%s",
                    user.user_id,
                    cred.failed_attempts,
                    cred.locked_until.isoformat(),
                )
            else:
                logger.warning(
                    "auth:login_failed reason=wrong_password user_id=%s failed_attempts=%s",
                    user.user_id,
                    cred.failed_attempts,
                )

            db.flush()
            db.commit()
            raise CredentialError("Invalid email or password.")

        # SMTP-off verification bypass (FR-C6/AC-4): when no SMTP is configured,
        # skip the is_verified gate so self-hosted installs work without email.
        if smtp_configured() and not cred.is_verified:
            logger.warning(
                "auth:login_failed reason=email_not_verified user_id=%s", user.user_id
            )
            raise CredentialError(
                "Please verify your email address before logging in."
            )

        if not user.is_active:
            logger.warning(
                "auth:login_failed reason=inactive_account user_id=%s", user.user_id
            )
            raise InactiveAccountError("This account has been deactivated.")

        # Success: reset lockout counters.
        cred.failed_attempts = 0
        cred.locked_until = None
        db.flush()
        db.commit()

        logger.info("auth:login_success user_id=%s", user.user_id)
        return user

    # ------------------------------------------------------------------
    # Admin user management
    # ------------------------------------------------------------------

    @staticmethod
    async def admin_create_user(db: Session, email: str, name: str) -> User:
        """Create a new LOCAL auth user account (admin-initiated).

        The new user has no password set — the admin must call
        ``issue_set_password_token`` to generate a first-time setup link.

        Does NOT create a Subscription record; subscription lifecycle belongs
        to the SaaS domain and is outside the scope of LOCAL auth.

        Args:
            db: Active database session.
            email: New user's email address.
            name: Display name.

        Returns:
            The newly created ``User`` ORM instance.

        Raises:
            UserAlreadyExistsError: When a user with this email already exists.
        """
        user_repo = UserRepository(db)
        existing = user_repo.get_by_email(email)
        if existing is not None:
            raise UserAlreadyExistsError(
                f"A user with email '{email}' already exists."
            )

        # email_verified=True when SMTP is off (no verification handshake needed).
        user = User(
            email=email,
            name=name,
            auth_method="local",
            is_active=True,
            email_verified=not smtp_configured(),
        )
        db.add(user)
        db.flush()

        # Credential row with a locked placeholder password.  The placeholder is
        # a freshly hashed random string that cannot be guessed; it is replaced
        # when the user completes the set-password flow.
        placeholder_hash = await run_in_threadpool(
            _sync_hash, secrets.token_urlsafe(32)
        )
        cred_repo = UserCredentialRepository(db)
        cred_repo.create(user_id=user.user_id, hashed_password=placeholder_hash)

        db.commit()
        db.refresh(user)

        logger.info("auth:admin_create_user user_id=%s email=%s", user.user_id, email)
        return user

    @staticmethod
    async def admin_set_password(db: Session, user_id: int, new_password: str) -> None:
        """Forcibly set a user's password (admin panel emergency reset).

        Resets lockout state and revokes all existing refresh sessions so the
        user must re-authenticate with the new password.

        Args:
            db: Active database session.
            user_id: Target user's numeric PK.
            new_password: The new plaintext password (already policy-validated
                by the schema before reaching this method).

        Raises:
            CredentialError: When the user or credential record is not found.
        """
        cred_repo = UserCredentialRepository(db)
        cred = cred_repo.get_by_user_id(user_id)
        if cred is None:
            logger.warning("auth:admin_set_password credential_not_found user_id=%s", user_id)
            raise CredentialError("Credential record not found.")

        hashed = await hash_password(new_password)
        cred_repo.update_password(user_id, hashed)

        # An admin-set password counts as completed account setup: mark the
        # credential verified so it is not treated as "pending setup" (e.g. the
        # omniadmin bootstrap would otherwise re-issue a set-password link on
        # every restart for an account that already has a usable password).
        cred.is_verified = True

        # Reset lockout state.
        cred.failed_attempts = 0
        cred.locked_until = None
        db.flush()

        # Revoke all active refresh sessions (force re-login with new password).
        RefreshService.revoke_all(db, user_id)

        db.commit()
        logger.info("auth:admin_set_password user_id=%s", user_id)

    # ------------------------------------------------------------------
    # Set-password token flow
    # ------------------------------------------------------------------

    @staticmethod
    def issue_set_password_token(db: Session, user_id: int) -> str:
        """Issue a single-use time-limited set-password token.

        The raw token is stored (SHA-1 is NOT used here — the itsdangerous
        serialiser signs with HMAC-SHA1 against SECRET_KEY) as a marker in
        ``reset_token`` / ``reset_token_expiry`` so it can be invalidated on
        consumption.  The token encodes the user's email, which is verified
        on decode.

        Args:
            db: Active database session.
            user_id: The target user's numeric PK.

        Returns:
            The raw token string.  The caller (router) returns it to the admin
            in the response body; when SMTP is configured the router also sends
            it via ``EmailSender``.

        Raises:
            CredentialError: When the user or credential record is not found.
        """
        user_repo = UserRepository(db)
        user = user_repo.get_by_id(user_id)
        if user is None:
            logger.warning("auth:issue_set_password_token user_not_found user_id=%s", user_id)
            raise CredentialError("User not found.")

        cred_repo = UserCredentialRepository(db)
        cred = cred_repo.get_by_user_id(user_id)
        if cred is None:
            logger.warning(
                "auth:issue_set_password_token credential_not_found user_id=%s", user_id
            )
            raise CredentialError("Credential record not found.")

        token = _get_serializer().dumps(user.email, salt=_TOKEN_SALT_SET_PASSWORD)
        expiry = datetime.now(timezone.utc) + timedelta(seconds=_SET_PASSWORD_TOKEN_MAX_AGE_SECONDS)

        # Store the raw token as the single-use marker.
        cred.reset_token = token
        cred.reset_token_expiry = expiry
        db.flush()
        db.commit()

        logger.info(
            "auth:set_password_token_issued user_id=%s expires_at=%s",
            user_id,
            expiry.isoformat(),
        )
        return token

    @staticmethod
    async def consume_set_password_token(
        db: Session, token: str, new_password: str
    ) -> User:
        """Validate a set-password token and apply the new password.

        Validates:
        1. itsdangerous signature and max-age.
        2. The decoded email resolves to an existing user.
        3. The stored ``reset_token`` marker matches the presented token
           (single-use: the marker is cleared on consumption).

        Args:
            db: Active database session.
            token: The raw token string from the set-password link.
            new_password: The new plaintext password (policy-validated by
                schema before reaching this method).

        Returns:
            The ``User`` ORM instance after password is set and verified.

        Raises:
            TokenError: On any validation failure.
        """
        try:
            email: str = _get_serializer().loads(
                token,
                salt=_TOKEN_SALT_SET_PASSWORD,
                max_age=_SET_PASSWORD_TOKEN_MAX_AGE_SECONDS,
            )
        except SignatureExpired:
            logger.warning("auth:set_password_token_rejected reason=expired")
            raise TokenError("Set-password link has expired.")
        except BadSignature:
            logger.warning("auth:set_password_token_rejected reason=bad_signature")
            raise TokenError("Set-password link is invalid.")

        user_repo = UserRepository(db)
        user = user_repo.get_by_email(email)
        if user is None:
            logger.warning("auth:set_password_token_rejected reason=user_not_found email=%s", email)
            raise TokenError("Set-password link is invalid.")

        cred_repo = UserCredentialRepository(db)
        cred = cred_repo.get_by_user_id(user.user_id)
        if cred is None or not hmac.compare_digest(cred.reset_token or "", token):
            # Token has already been consumed or was never issued.
            logger.warning(
                "auth:set_password_token_rejected reason=already_used_or_not_found user_id=%s",
                user.user_id,
            )
            raise TokenError("Set-password link has already been used or is invalid.")

        # Optional: check email local-part match at service level.
        check_password_not_email(new_password, email)

        hashed = await hash_password(new_password)

        # Apply new password.
        cred_repo.update_password(user.user_id, hashed)

        # Mark the user as verified (they clicked the admin-issued link).
        cred_repo.mark_verified(user.user_id)

        # Reset lockout state (new password means clean slate).  update_password
        # already cleared the single-use reset_token marker on the same row.
        cred.failed_attempts = 0
        cred.locked_until = None

        # Sync User.email_verified
        user.email_verified = True
        db.flush()
        db.commit()
        db.refresh(user)

        logger.info("auth:set_password_token_consumed user_id=%s", user.user_id)
        return user

    # ------------------------------------------------------------------
    # Self-service change-password
    # ------------------------------------------------------------------

    @staticmethod
    async def change_password(
        db: Session, user: User, current_password: str, new_password: str
    ) -> None:
        """Change an authenticated user's password.

        Verifies the current password (off-loop), applies policy to the new
        password (already validated by schema), then revokes all refresh
        sessions to force re-authentication on all devices.

        Args:
            db: Active database session.
            user: The authenticated ``User`` ORM instance (from the request's
                auth dependency).
            current_password: The user's existing plaintext password.
            new_password: The new plaintext password (already policy-validated
                by schema before reaching this method).

        Raises:
            CredentialError: When current_password does not match.
        """
        cred_repo = UserCredentialRepository(db)
        cred = cred_repo.get_by_user_id(user.user_id)
        if cred is None:
            raise CredentialError("No credential record found for this account.")

        current_ok: bool = await run_in_threadpool(
            _sync_verify, current_password, cred.hashed_password
        )
        if not current_ok:
            logger.warning(
                "auth:change_password_failed reason=wrong_current user_id=%s", user.user_id
            )
            raise CredentialError("Current password is incorrect.")

        # Check new password is not the same as the email local-part.
        check_password_not_email(new_password, user.email)

        hashed = await hash_password(new_password)
        cred_repo.update_password(user.user_id, hashed)

        # Revoke all refresh sessions (FR-C7).
        RefreshService.revoke_all(db, user.user_id)

        db.commit()
        logger.info("auth:change_password_success user_id=%s", user.user_id)
