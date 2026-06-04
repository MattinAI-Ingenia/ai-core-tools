"""Refresh-token lifecycle service for LOCAL auth mode.

Implements the AD-4 rotation + reuse-detection protocol:

Wire format
-----------
The opaque value stored in the httpOnly cookie (wired in step_004/006) is::

    "<jti>.<opaque_secret>"

where:
- ``jti``          — uuid4 hex, used as the DB lookup key (indexed).
- ``opaque_secret``— ``secrets.token_urlsafe(48)``, the high-entropy value
                     whose SHA-256 digest is stored in ``token_hash``.

Only the SHA-256 digest of ``opaque_secret`` is ever persisted.  The raw
opaque value is never logged or stored.

Rotation safety
---------------
``rotate()`` acquires a ``WITH FOR UPDATE`` row lock before inspecting
``rotated_at``.  This means two concurrent rotation requests for the same
token serialise at the DB level:

- The first request sees ``rotated_at IS NULL`` → proceeds normally.
- The second request sees ``rotated_at IS NOT NULL`` → reuse detected →
  entire family revoked.

There is no false-positive: the only way ``rotated_at`` is set is by a
successful prior rotation of that exact token, which is a true reuse signal.
"""

import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from models.user import User
from repositories.refresh_token_repository import RefreshTokenRepository
from repositories.user_repository import UserRepository
from utils.local_auth_tokens import mint_access_token
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REFRESH_TTL_DAYS: int = int(os.getenv("LOCAL_REFRESH_TTL_DAYS", "14"))
#: Exported so that auth_cookies.py can share this single source of truth
#: for the cookie max-age without re-reading the env var independently.
REFRESH_TTL_DAYS: int = _REFRESH_TTL_DAYS
_OPAQUE_BYTES: int = 48  # yields 64 url-safe base64 chars

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RefreshTokenError(Exception):
    """Raised when a presented refresh token cannot be accepted.

    Callers (HTTP layer — step_004/006) should map this to HTTP 401.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_opaque() -> str:
    """Generate a cryptographically random opaque token value."""
    return secrets.token_urlsafe(_OPAQUE_BYTES)


def _new_jti() -> str:
    """Generate a random JTI for DB lookup."""
    return uuid.uuid4().hex


def _hash_opaque(opaque: str) -> str:
    """Return the SHA-256 hex digest of an opaque token value.

    Args:
        opaque: The raw opaque token string.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(opaque.encode()).hexdigest()


def _compose_wire(jti: str, opaque: str) -> str:
    """Compose the wire value stored in the cookie.

    Format: ``"<jti>.<opaque>"``.  The dot separator is safe because JTI is
    a hex string (no dots) and opaque is url-safe base64 (no dots).

    Args:
        jti: The token identifier.
        opaque: The raw opaque secret.

    Returns:
        The combined string to hand to the client.
    """
    return f"{jti}.{opaque}"


def _parse_wire(wire: str) -> tuple[str, str]:
    """Parse the wire value back into (jti, opaque).

    Args:
        wire: The value presented by the client.

    Returns:
        Tuple of ``(jti, opaque)``.

    Raises:
        RefreshTokenError: If the wire format is invalid.
    """
    parts = wire.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise RefreshTokenError("Invalid refresh token format.")
    return parts[0], parts[1]


def _load_user(db: Session, user_id: int) -> User:
    """Fetch a user by ID and assert they are active.

    Args:
        db: Active database session.
        user_id: Numeric PK of the user.

    Returns:
        The active ``User`` instance.

    Raises:
        RefreshTokenError: If the user is not found or inactive.
    """
    user_repo = UserRepository(db)
    user: Optional[User] = user_repo.get_by_id(user_id)
    if not user or not user.is_active:
        raise RefreshTokenError("User account not found or deactivated.")
    return user


# ---------------------------------------------------------------------------
# Public service API
# ---------------------------------------------------------------------------


class RefreshService:
    """Service layer for refresh-token issuance, rotation, and revocation."""

    @staticmethod
    def issue_session(
        db: Session,
        user: User,
        *,
        user_agent: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> tuple[str, str, datetime]:
        """Issue a fresh access + refresh token pair for a newly authenticated user.

        Creates a new rotation family.  Should be called after successful
        password validation in the login flow.

        Args:
            db: Active database session.
            user: The authenticated ``User`` ORM instance.
            user_agent: Optional HTTP User-Agent for audit logging.
            ip: Optional client IP address for audit logging.

        Returns:
            A three-tuple of:
            - ``access_token``: Signed JWT string ready to send to the client.
            - ``refresh_token_plain``: Wire-format refresh token (``jti.opaque``)
              to store in an httpOnly cookie.
            - ``refresh_expires_at``: Timezone-aware UTC datetime of refresh
              token expiry; use to set the cookie's ``Max-Age`` / ``Expires``.
        """
        access_token, _ = mint_access_token(
            user_id=user.user_id,
            email=user.email,
            name=user.name,
        )

        family_id = uuid.uuid4().hex
        jti = _new_jti()
        opaque = _new_opaque()
        refresh_expires_at = datetime.now(timezone.utc) + timedelta(days=_REFRESH_TTL_DAYS)

        RefreshTokenRepository.create(
            db,
            jti=jti,
            user_id=user.user_id,
            family_id=family_id,
            token_hash=_hash_opaque(opaque),
            expires_at=refresh_expires_at,
            user_agent=user_agent,
            ip=ip,
        )
        db.commit()

        logger.info(
            "auth:session_issued user_id=%s family_id=%s",
            user.user_id,
            family_id,
        )

        return access_token, _compose_wire(jti, opaque), refresh_expires_at

    @staticmethod
    def rotate(
        db: Session,
        presented_refresh_plain: str,
        *,
        user_agent: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> tuple[str, str, datetime]:
        """Exchange a valid refresh token for a new access + refresh token pair.

        The rotation is performed inside a single transaction with a row-level
        lock to prevent concurrent rotation races.  See module docstring for
        the safety analysis.

        Args:
            db: Active database session.
            presented_refresh_plain: Wire-format value from the client cookie.
            user_agent: Optional HTTP User-Agent (applied to the new token row).
            ip: Optional client IP (applied to the new token row).

        Returns:
            A three-tuple identical in shape to ``issue_session``.

        Raises:
            RefreshTokenError: When the token is missing, expired, revoked,
                already rotated (reuse), or the opaque hash does not match.
        """
        jti, opaque = _parse_wire(presented_refresh_plain)

        row = RefreshTokenRepository.get_by_jti_for_update(db, jti)

        if row is None:
            logger.warning("auth:rotate_rejected reason=not_found jti=%s", jti)
            raise RefreshTokenError("Refresh token not found.")

        now = datetime.now(timezone.utc)

        # Postgres stores ``expires_at`` in a naive ``DateTime`` column, so the
        # value read back is timezone-naive.  Coerce it to UTC-aware before the
        # comparison; otherwise ``naive < aware`` raises ``TypeError`` on every
        # rotation against a real database.
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at < now:
            logger.warning(
                "auth:rotate_rejected reason=expired user_id=%s family_id=%s",
                row.user_id,
                row.family_id,
            )
            raise RefreshTokenError("Refresh token has expired.")

        if row.revoked_at is not None:
            logger.warning(
                "auth:rotate_rejected reason=revoked user_id=%s family_id=%s",
                row.user_id,
                row.family_id,
            )
            raise RefreshTokenError("Refresh token has been revoked.")

        if row.rotated_at is not None:
            # Token has already been used → reuse detected.
            # Revoke the entire family to force re-authentication.
            revoked_count = RefreshTokenRepository.revoke_family(db, row.family_id)
            db.commit()
            logger.warning(
                "auth:reuse_detected user_id=%s family_id=%s revoked=%s",
                row.user_id,
                row.family_id,
                revoked_count,
            )
            raise RefreshTokenError("Refresh token reuse detected; session invalidated.")

        # Constant-time hash comparison to prevent timing attacks.
        if not hmac.compare_digest(_hash_opaque(opaque), row.token_hash):
            logger.warning(
                "auth:rotate_rejected reason=hash_mismatch user_id=%s",
                row.user_id,
            )
            raise RefreshTokenError("Refresh token is invalid.")

        # Mark the consumed token as rotated (non-null rotated_at is the reuse
        # sentinel — must be done before the new row is created).
        RefreshTokenRepository.mark_rotated(db, row)

        # Load and validate the user before minting new tokens.
        user = _load_user(db, row.user_id)

        # Mint new access token.
        access_token, _ = mint_access_token(
            user_id=user.user_id,
            email=user.email,
            name=user.name,
        )

        # Issue successor refresh token in the same family.
        new_jti = _new_jti()
        new_opaque = _new_opaque()
        refresh_expires_at = now + timedelta(days=_REFRESH_TTL_DAYS)

        RefreshTokenRepository.create(
            db,
            jti=new_jti,
            user_id=user.user_id,
            family_id=row.family_id,
            token_hash=_hash_opaque(new_opaque),
            expires_at=refresh_expires_at,
            user_agent=user_agent,
            ip=ip,
        )
        db.commit()

        logger.info(
            "auth:token_rotated user_id=%s family_id=%s",
            user.user_id,
            row.family_id,
        )

        return access_token, _compose_wire(new_jti, new_opaque), refresh_expires_at

    @staticmethod
    def revoke_current(db: Session, presented_refresh_plain: str) -> None:
        """Revoke the presented refresh token (single-device logout).

        Parses the wire value, revokes only the identified JTI.  Does not
        revoke the rest of the family (other devices remain logged in).

        Args:
            db: Active database session.
            presented_refresh_plain: Wire-format value from the client cookie.
        """
        try:
            jti, _ = _parse_wire(presented_refresh_plain)
        except RefreshTokenError:
            # Malformed token — nothing to revoke; treat as success.
            return

        RefreshTokenRepository.revoke_jti(db, jti)
        db.commit()
        logger.info("auth:logout jti=%s", jti)

    @staticmethod
    def revoke_all(db: Session, user_id: int) -> None:
        """Revoke all refresh tokens for a user (logout-all / account suspension).

        Args:
            db: Active database session.
            user_id: Numeric PK of the user whose sessions should be ended.
        """
        count = RefreshTokenRepository.revoke_all_for_user(db, user_id)
        db.commit()
        logger.info("auth:logout_all user_id=%s revoked=%s", user_id, count)
