"""First-boot omniadmin provisioning for LOCAL auth mode.

This module provides ``bootstrap_omniadmins``, called once at application
startup (in-process, within the FastAPI lifespan) when ``AICT_LOGIN=LOCAL`` and
``AUTH_BOOTSTRAP_OMNIADMINS=true`` (default).

Security model
--------------
The bootstrap runs **server-side only** — there is no HTTP endpoint and no
time-window race condition.  The one-time set-password URL is delivered
**out-of-band via the backend logs** so only an operator with log-level access
to the running container/process can see it.

This is the ONLY location in the codebase where a set-password token is
intentionally written to a log line.  It is acceptable here because:

1. The log channel is the designated out-of-band delivery medium for a
   server-side first-boot flow.  The operator *is* the administrator, and
   log access already grants full infrastructure control.
2. The token is single-use (consumed on first ``/set-password`` submit) and
   time-limited (default 48 h, ``LOCAL_SET_PASSWORD_TOKEN_MAX_AGE_HOURS``).
3. The WARNING level makes the line visible and auditable while keeping it
   distinct from normal INFO noise.
4. This pattern is analogous to the nginx-ui / Gitea first-run admin-token
   approach, hardened by the absence of any network-exposed bootstrap
   endpoint (cf. nginx-ui CVE-2024-23835).

Multi-worker / multi-replica safety (PostgreSQL advisory lock)
--------------------------------------------------------------
The Dockerfile launches uvicorn with ``--workers ${UVICORN_WORKERS:-2}``, and
Kubernetes may run several replicas — all sharing the same PostgreSQL instance.
Without coordination, every worker/replica enters the lifespan concurrently and
the existence check + create sequence is non-atomic, causing duplicate users and
duplicate set-password links to appear in the logs.

``bootstrap_omniadmins`` acquires a session-level try-lock before doing any
provisioning work:

    SELECT pg_try_advisory_lock(:key)

If the lock is NOT obtained (another worker holds it), the function exits
immediately.  The one worker that obtains the lock runs the full provisioning
loop and releases the lock in a ``finally`` block.  Because advisory locks are
session-scoped they are released automatically when the DB connection is closed.

The key ``_ADVISORY_LOCK_KEY`` (``0x6f6d6e6961646d31 == 7958800063258050865``)
is a hard-coded stable 64-bit integer derived from the ASCII bytes of
``omniadm1`` — no hash(), no random, no env var.  It must never change once
deployed (changing it would let two workers hold "different" locks simultaneously).

Partial-failure recovery (idempotent on stranded accounts)
----------------------------------------------------------
A crash between ``admin_create_user`` (which commits) and
``issue_set_password_token`` (a second commit) leaves a user row with no
outstanding set-password link.  On next restart the old "skip if user exists"
logic would silently skip that user forever.

The new per-omniadmin logic checks:

1. Does the user exist?                   → No  → create + issue token (original path).
2. User exists + ``is_verified=True``     → already set up; skip.
3. User exists + unexpired ``reset_token`` → a valid link is already outstanding; skip.
4. User exists + NOT verified + no valid link → re-issue token (crash-recovery path).

Idempotency
-----------
Safe to re-run on every restart — completed accounts are never disturbed and
accounts with a live outstanding link receive no duplicate.
"""

import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.auth.credential_service import CredentialService, UserAlreadyExistsError
from utils.config import Config, get_omniadmins
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Whether the bootstrap runs at all.  Set AUTH_BOOTSTRAP_OMNIADMINS=false to
# disable (e.g. in CI or when omniadmins are provisioned by another mechanism).
_BOOTSTRAP_ENABLED: bool = Config.get_bool_env_var("AUTH_BOOTSTRAP_OMNIADMINS", default=True)

# Reuse the same expiry constant as the admin router so operators see a
# consistent window regardless of which code path issued the token.
_TOKEN_MAX_AGE_HOURS: int = Config.get_int_env_var(
    "LOCAL_SET_PASSWORD_TOKEN_MAX_AGE_HOURS", default=48
)

# Frontend URL used to build the full set-password link delivered in the log.
_FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ---------------------------------------------------------------------------
# Advisory lock key
# ---------------------------------------------------------------------------
#
# A stable 64-bit integer used as the PostgreSQL session-level advisory lock
# key.  Derived from the ASCII bytes of "omniadm1" — do NOT change after
# first deployment.  Changing it would allow two workers to hold "different"
# locks and defeat the mutual-exclusion guarantee.
#
# Computation (for auditors):
#   int.from_bytes(b"omniadm1", "big") == 0x6f6d6e6961646d31 == 7958800063258050865
_ADVISORY_LOCK_KEY: int = 0x6F6D6E6961646D31  # "omniadm1" as big-endian int64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def bootstrap_omniadmins(db: Session) -> None:
    """Provision omniadmin accounts on first boot in LOCAL auth mode.

    Acquires a PostgreSQL session-level advisory lock before doing any work so
    that only ONE worker / replica runs the bootstrap loop even when several
    start simultaneously.  Workers that cannot obtain the lock exit immediately.

    For each email in ``AICT_OMNIADMINS``:
    - If NO ``User`` row exists → create a LOCAL auth user via
      ``CredentialService.admin_create_user``, issue a one-time set-password
      token, and log a WARNING containing the full set-password URL.
    - If the user EXISTS and has completed set-password (``is_verified=True``) →
      skip silently.
    - If the user EXISTS and has a valid (unexpired) outstanding link → skip
      (the link is still usable; no need to issue a new one).
    - If the user EXISTS but setup is incomplete AND there is no valid link
      (crash-recovery case) → re-issue a set-password token and log the link.

    This function is a no-op when ``AUTH_BOOTSTRAP_OMNIADMINS=false``.

    Individual per-email failures are caught and logged so that one bad entry
    cannot abort the entire application startup.

    Args:
        db: A synchronous SQLAlchemy session from the startup lifespan context.
            Commits are performed inside ``admin_create_user`` and
            ``issue_set_password_token``; no additional commit is required here.
    """
    if not _BOOTSTRAP_ENABLED:
        logger.debug("omniadmin_bootstrap: AUTH_BOOTSTRAP_OMNIADMINS=false, skipping")
        return

    omniadmins = get_omniadmins()
    if not omniadmins:
        logger.debug("omniadmin_bootstrap: AICT_OMNIADMINS is empty, nothing to bootstrap")
        return

    # Acquire a PostgreSQL session-level advisory try-lock.  If another worker
    # or k8s replica already holds it, exit immediately — they will complete the
    # bootstrap.  The lock is released in the finally block (or automatically
    # when the session/connection is closed).
    try:
        locked: bool = db.execute(
            text("SELECT pg_try_advisory_lock(:k)"),
            {"k": _ADVISORY_LOCK_KEY},
        ).scalar()
    except Exception as lock_exc:
        logger.error(
            "omniadmin_bootstrap: could not acquire advisory lock — %s; skipping bootstrap",
            lock_exc,
        )
        return

    if not locked:
        logger.debug(
            "omniadmin_bootstrap: advisory lock held by another worker/replica; "
            "bootstrap skipped (lock key=0x%x)",
            _ADVISORY_LOCK_KEY,
        )
        return

    logger.info(
        "omniadmin_bootstrap: acquired advisory lock (key=0x%x); checking %d omniadmin email(s)",
        _ADVISORY_LOCK_KEY,
        len(omniadmins),
    )

    try:
        for email in omniadmins:
            # get_omniadmins() already normalises; belt-and-suspenders guard.
            email = email.strip().lower()
            if not email:
                continue
            await _provision_single_omniadmin(db, email)
    finally:
        try:
            db.execute(
                text("SELECT pg_advisory_unlock(:k)"),
                {"k": _ADVISORY_LOCK_KEY},
            )
        except Exception as unlock_exc:
            # Non-fatal: the lock will be released when the connection is closed.
            logger.warning(
                "omniadmin_bootstrap: advisory unlock failed (non-fatal) — %s",
                unlock_exc,
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _provision_single_omniadmin(db: Session, email: str) -> None:
    """Provision or recover a single omniadmin account.

    Decision tree:
    1. No user row → create + issue set-password token + log link.
    2. User exists, ``is_verified=True`` → account is fully set up; skip.
    3. User exists, valid unexpired ``reset_token`` exists → a link is already
       outstanding; skip (avoid flooding the logs with duplicate links).
    4. User exists, NOT verified, no valid link → crash-recovery path: re-issue
       a set-password token and log the link so the operator can complete setup.

    Partial-failure recovery (case 4) handles the scenario where a previous
    startup committed ``admin_create_user`` but crashed before
    ``issue_set_password_token`` completed, leaving the account stranded with no
    usable link.

    Args:
        db: Active database session.
        email: Normalised (lowercase, stripped) omniadmin email address.
    """
    from repositories.user_credential_repository import UserCredentialRepository
    from repositories.user_repository import UserRepository

    user_repo = UserRepository(db)
    existing = user_repo.get_by_email(email)

    if existing is None:
        # --- Path 1: new account ---
        try:
            name = email.split("@", 1)[0].capitalize()
            user = await CredentialService.admin_create_user(db, email=email, name=name)
        except UserAlreadyExistsError:
            # Race between the look-up above and the create (shouldn't happen
            # while the advisory lock is held, but kept as a secondary guard).
            logger.info(
                "omniadmin_bootstrap: concurrent creation detected for %s, skipping token issuance",
                email,
            )
            return
        except Exception as exc:
            logger.error(
                "omniadmin_bootstrap: failed to create user for %s — %s",
                email,
                exc,
                exc_info=True,
            )
            return

        _issue_and_log(db, user.user_id, email)
        return

    # --- Paths 2 / 3 / 4: user row already exists ---
    cred_repo = UserCredentialRepository(db)
    cred = cred_repo.get_by_user_id(existing.user_id)

    if cred is not None and cred.is_verified:
        # Path 2: account fully configured — nothing to do.
        logger.info(
            "omniadmin_bootstrap: %s (user_id=%s) is already verified; skipping",
            email,
            existing.user_id,
        )
        return

    # Check whether a currently-valid reset/set-password link is already stored.
    now_utc = datetime.now(timezone.utc)
    has_valid_link = _has_valid_outstanding_link(cred, now_utc)

    if has_valid_link:
        # Path 3: a link is already outstanding and still usable.
        logger.info(
            "omniadmin_bootstrap: %s (user_id=%s) has a valid outstanding set-password link; "
            "skipping (link has not been used yet)",
            email,
            existing.user_id,
        )
        return

    # Path 4: crash-recovery — user exists but setup is incomplete and the
    # previous token is either missing, expired, or was consumed.
    logger.warning(
        "omniadmin_bootstrap: %s (user_id=%s) exists but setup is incomplete "
        "and no valid link is outstanding; re-issuing set-password token (crash-recovery)",
        email,
        existing.user_id,
    )
    _issue_and_log(db, existing.user_id, email)


def _has_valid_outstanding_link(cred, now_utc: datetime) -> bool:
    """Return True when a non-expired set-password token is stored on the credential.

    Handles both timezone-aware and timezone-naive ``reset_token_expiry`` values
    (legacy rows may be naive UTC) by coercing naive datetimes to UTC before
    comparison — mirroring the UTC-coercion pattern used in
    ``CredentialService.authenticate``.

    Args:
        cred: ``UserCredential`` ORM instance, or ``None``.
        now_utc: Current UTC-aware datetime for comparison.

    Returns:
        ``True`` when ``cred.reset_token`` is non-empty AND
        ``cred.reset_token_expiry`` is in the future.
    """
    if cred is None:
        return False
    if not cred.reset_token:
        return False
    expiry = cred.reset_token_expiry
    if expiry is None:
        return False
    # Coerce naive datetime to UTC-aware.
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry > now_utc


def _issue_and_log(db: Session, user_id: int, email: str) -> None:
    """Issue a set-password token and log the WARNING link line.

    Extracted so both the new-account path and the crash-recovery path share
    identical token issuance and log formatting.

    Args:
        db: Active database session.
        user_id: Numeric PK of the target user.
        email: Normalised email address (for log output only).
    """
    try:
        token = CredentialService.issue_set_password_token(db, user_id)
    except Exception as exc:
        logger.error(
            "omniadmin_bootstrap: failed to issue set-password token for %s (user_id=%s) — %s",
            email,
            user_id,
            exc,
            exc_info=True,
        )
        return

    expires_at = (
        datetime.now(timezone.utc) + timedelta(hours=_TOKEN_MAX_AGE_HOURS)
    ).isoformat()

    set_password_url = f"{_FRONTEND_URL.rstrip('/')}/set-password?token={token}"

    # --- Intentional WARNING-level log of the set-password URL ---
    # This is the designated out-of-band delivery channel for the first-boot
    # admin token.  The URL is single-use and time-limited.  Only an operator
    # with access to container/process logs can read it, which is equivalent
    # to having root access to the deployment.  See module docstring for full
    # security justification.
    logger.warning(
        "omniadmin_bootstrap: NEW omniadmin account provisioned.\n"
        "  Email   : %s\n"
        "  User ID : %s\n"
        "  Expires : %s\n"
        "  Link    : %s\n"
        "Open the link above to set the initial password for this account. "
        "This link will not appear again once used or after the account exists on restart.",
        email,
        user_id,
        expires_at,
        set_password_url,
    )
