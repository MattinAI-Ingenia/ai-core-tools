"""Data-access layer for RefreshToken records.

All methods accept an explicit ``db`` session and perform only flush operations
(no commit), keeping transaction boundaries in the calling service.

The ``get_by_jti_for_update`` method acquires a row-level lock via
``with_for_update()`` so that concurrent rotation requests are serialised at the
DB level rather than racing in application code.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from models.refresh_token import RefreshToken
from utils.logger import get_logger

logger = get_logger(__name__)


class RefreshTokenRepository:
    """Repository for RefreshToken CRUD and bulk revocation operations."""

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        db: Session,
        *,
        jti: str,
        user_id: int,
        family_id: str,
        token_hash: str,
        expires_at: datetime,
        user_agent: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> RefreshToken:
        """Persist a new refresh-token record.

        Args:
            db: Active database session.
            jti: Unique token identifier (uuid4 hex, used for DB lookup).
            user_id: FK to the owning User row.
            family_id: Rotation family identifier (uuid4 hex).
            token_hash: SHA-256 hex digest of the opaque token value; the raw
                token is never stored.
            expires_at: Absolute expiry timestamp (timezone-aware UTC).
            user_agent: Optional HTTP User-Agent string for audit purposes.
            ip: Optional client IP address for audit purposes.

        Returns:
            The newly created ``RefreshToken`` ORM instance.
        """
        row = RefreshToken(
            jti=jti,
            user_id=user_id,
            family_id=family_id,
            token_hash=token_hash,
            expires_at=expires_at,
            user_agent=user_agent,
            ip=ip,
        )
        db.add(row)
        db.flush()
        return row

    @staticmethod
    def mark_rotated(db: Session, row: RefreshToken) -> None:
        """Set ``rotated_at`` on the given row to the current UTC time.

        A non-null ``rotated_at`` on a presented token means the token has
        already been exchanged — reuse detected.

        Args:
            db: Active database session.
            row: The ``RefreshToken`` instance to mark.
        """
        row.rotated_at = datetime.now(timezone.utc)
        db.flush()

    @staticmethod
    def revoke_jti(db: Session, jti: str) -> None:
        """Bulk-UPDATE ``revoked_at`` on the single row identified by ``jti``.

        Only rows where ``revoked_at IS NULL`` are affected.

        Args:
            db: Active database session.
            jti: Token identifier to revoke.
        """
        now = datetime.now(timezone.utc)
        db.execute(
            update(RefreshToken)
            .where(RefreshToken.jti == jti, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        db.flush()

    @staticmethod
    def revoke_family(db: Session, family_id: str) -> int:
        """Revoke every non-yet-revoked token in a rotation family.

        Called on reuse detection to invalidate the entire token lineage,
        forcing the user to re-authenticate.

        Args:
            db: Active database session.
            family_id: The family identifier shared across the rotation chain.

        Returns:
            The number of rows updated.
        """
        now = datetime.now(timezone.utc)
        result = db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.family_id == family_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        db.flush()
        return result.rowcount

    @staticmethod
    def revoke_all_for_user(db: Session, user_id: int) -> int:
        """Revoke every non-yet-revoked token for a given user.

        Used by logout-all / account suspension flows.

        Args:
            db: Active database session.
            user_id: Numeric PK of the user whose tokens should be invalidated.

        Returns:
            The number of rows updated.
        """
        now = datetime.now(timezone.utc)
        result = db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        db.flush()
        return result.rowcount

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @staticmethod
    def get_by_jti(db: Session, jti: str) -> Optional[RefreshToken]:
        """Fetch a refresh-token row by its JTI without acquiring a row lock.

        Args:
            db: Active database session.
            jti: The unique token identifier.

        Returns:
            The matching ``RefreshToken`` or ``None`` if not found.
        """
        return db.execute(
            select(RefreshToken).where(RefreshToken.jti == jti)
        ).scalar_one_or_none()

    @staticmethod
    def get_by_jti_for_update(db: Session, jti: str) -> Optional[RefreshToken]:
        """Fetch a refresh-token row by JTI and acquire a pessimistic row lock.

        This method is used during token rotation.  The ``WITH FOR UPDATE``
        lock serialises concurrent rotation requests for the same token so that
        only one succeeds; the second sees ``rotated_at`` already set and
        triggers reuse detection rather than a false-positive duplicate rotation.

        Args:
            db: Active database session (must be inside an open transaction).
            jti: The unique token identifier.

        Returns:
            The matching ``RefreshToken`` with a row lock held, or ``None``.
        """
        return db.execute(
            select(RefreshToken)
            .where(RefreshToken.jti == jti)
            .with_for_update()
        ).scalar_one_or_none()
