"""Development user seeding utility.

Creates users in the database so they can log in while the platform runs in
development authentication mode (``AICT_LOGIN=FAKE`` or ``AICT_LOGIN=LOCAL``).

In FAKE mode, pre-seeded users log in without a password.
In LOCAL mode, users must have a password.  This script can set passwords via
the ``email:Name:password`` CSV format or via the ``AICT_DEV_SEED_PASSWORD``
env var.

CSV formats accepted:
    "email:Name"              — user only, no password (FAKE mode)
    "email:Name:password"     — user + password (LOCAL mode)
    "email::password"         — email + password, name derived from email local-part

The user set is resolved with the following precedence:
    1. ``--users "email:Name[:pw],email2:Name2[:pw]"`` CLI argument
    2. ``AICT_DEV_SEED_USERS`` environment variable (same CSV format)
    3. the built-in ``DEV_USERS`` defaults

Password fallback (LOCAL mode):
    ``AICT_DEV_SEED_PASSWORD`` is applied ONLY to users explicitly provided via
    ``--users`` or ``AICT_DEV_SEED_USERS``.  It is intentionally NOT applied to
    the built-in ``DEV_USERS`` defaults (``admin@example.com`` etc.) — doing so
    would silently give well-known example accounts a shared password from an
    env var, which is a security anti-pattern in production deployments that
    accidentally have the env var set.

    In LOCAL mode, a user without any password source is seeded without
    credentials (created but unable to log in until an admin issues a
    set-password link).

Omniadmin recovery path:
    If the omniadmin account (set via ``AICT_OMNIADMINS``) is accidentally
    locked out or has no credential, run this script in LOCAL mode with an
    explicit password, e.g.::

        AICT_DEV_SEED_USERS="admin@acme.com:Admin:NewPass123!" \\
        python -m utils.seed_dev_users --yes

    Omniadmins are exempt from the exponential lockout in
    ``CredentialService.authenticate`` (they can never be locked out of their
    own instance by brute-force).  If their account is deactivated, use the
    admin panel or the database directly to flip ``is_active=True``.

Safety:
    The script refuses to run unless ``AICT_LOGIN`` is ``FAKE`` or ``LOCAL``,
    to avoid creating users in an OIDC deployment.  Use ``--force`` to
    override that guard deliberately.

Designed to be safe and non-interactive inside containers. Typical usage::

    # Inside a running Docker deployment (no TTY required):
    docker compose exec -T backend python -m utils.seed_dev_users --yes

    # LOCAL mode with explicit passwords:
    docker compose exec -T backend \\
      python -m utils.seed_dev_users --yes \\
      --users "admin@acme.com:Admin:S3cr3tPass!,dev@acme.com:Dev:AnotherPass!"

    # LOCAL mode with a shared fallback password (applied only to --users entries):
    AICT_DEV_SEED_PASSWORD=S3cr3tPass! \\
      python -m utils.seed_dev_users --yes \\
      --users "admin@acme.com:Admin,dev@acme.com:Dev"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.orm import Session
from db.database import SessionLocal
from services.user_service import UserService
from utils.logger import get_logger

logger = get_logger(__name__)

# Default dev users to create when no CLI/env list is provided
DEV_USERS = [
    {
        "email": "admin@example.com",
        "name": "Admin User",
        "description": "Admin/test user for development",
        "password": None,
    },
    {
        "email": "user1@example.com",
        "name": "Test User 1",
        "description": "Regular test user 1",
        "password": None,
    },
    {
        "email": "user2@example.com",
        "name": "Test User 2",
        "description": "Regular test user 2",
        "password": None,
    },
    {
        "email": "dev@example.com",
        "name": "Developer",
        "description": "Developer test account",
        "password": None,
    },
]

# Env var holding a declarative, comma-separated user list
# Formats: "email:Name,email2:Name2" or "email:Name:password,email2:Name2:pw"
SEED_USERS_ENV = "AICT_DEV_SEED_USERS"

# Env var holding a fallback password applied to all users without an explicit
# per-user password in LOCAL mode.
SEED_PASSWORD_ENV = "AICT_DEV_SEED_PASSWORD"

# Auth modes under which seeding dev users is meaningful
_SEEDABLE_MODES = ("FAKE", "LOCAL")


def _parse_users_spec(spec: str) -> list:
    """Parse a CSV user specification string into seed user dicts.

    Supports two formats:
    - ``email:Name``          — user without password (FAKE mode)
    - ``email:Name:password`` — user with password (LOCAL mode)
    - ``email::password``     — name defaults to email local-part

    The name part is optional; when omitted, the local part of the email is
    used as a fallback display name.  Blank entries are ignored.

    Args:
        spec: Comma-separated user specification.

    Returns:
        List of user dicts with ``email``, ``name``, ``description``, and
        ``password`` (``None`` when not supplied) keys.
    """
    users = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        parts = chunk.split(":", 2)
        # Normalise email to canonical lowercase so seeded emails match the
        # login normalisation applied by LoginRequest (strip + lower).
        email = parts[0].strip().lower()
        name = parts[1].strip() if len(parts) > 1 else ""
        password: str | None = parts[2].strip() if len(parts) > 2 else None

        if not email:
            continue
        if not name:
            name = email.split("@", 1)[0]

        # Treat empty-string password as no password
        if password == "":
            password = None

        users.append({
            "email": email,
            "name": name,
            "description": "Seeded dev user",
            "password": password,
        })

    return users


def resolve_users(cli_spec: str = None) -> list:
    """Resolve the user list from CLI arg, env var, or built-in defaults.

    Args:
        cli_spec: Value of the ``--users`` argument, if provided.

    Returns:
        List of user dicts to seed.
    """
    if cli_spec:
        return _parse_users_spec(cli_spec)

    env_spec = os.getenv(SEED_USERS_ENV, "").strip()
    if env_spec:
        return _parse_users_spec(env_spec)

    return list(DEV_USERS)


def _is_explicit_user_spec(cli_spec: str = None) -> bool:
    """Return True when users came from an explicit source (CLI arg or env var).

    Used to decide whether ``AICT_DEV_SEED_PASSWORD`` should be applied as a
    fallback password.  The built-in ``DEV_USERS`` defaults must NOT receive the
    fallback to avoid silently giving well-known example accounts a shared password.

    Args:
        cli_spec: Value of the ``--users`` argument, if provided.

    Returns:
        ``True`` when ``cli_spec`` is set or ``AICT_DEV_SEED_USERS`` is non-empty.
    """
    if cli_spec:
        return True
    return bool(os.getenv(SEED_USERS_ENV, "").strip())


def current_login_mode() -> str:
    """Return the configured login mode (``AICT_LOGIN``), uppercased."""
    return os.getenv("AICT_LOGIN", "OIDC").strip().upper()


def is_seedable_mode() -> bool:
    """Whether the current login mode allows seeding dev users safely."""
    return current_login_mode() in _SEEDABLE_MODES


async def _set_credential(db: Session, user_id: int, password: str) -> None:
    """Idempotently set a credential for ``user_id``.

    Uses ``admin_set_password`` when a credential row already exists and
    ``admin_create_user``/``admin_set_password`` is not appropriate for an
    existing user.  The approach here is:

    1. If no ``UserCredential`` row exists → create one (via the credential
       repo directly) and then set the password.
    2. If a row exists → call ``admin_set_password`` to update it.

    This keeps the seeder idempotent: re-running it re-sets the password to
    the requested value without creating duplicate rows.

    Args:
        db: Active database session.
        user_id: Numeric PK of the target user.
        password: Plaintext password string.
    """
    from repositories.user_credential_repository import UserCredentialRepository
    from services.auth.credential_service import CredentialService, hash_password

    cred_repo = UserCredentialRepository(db)
    cred = cred_repo.get_by_user_id(user_id)

    if cred is None:
        # No credential row yet — create a placeholder then set the real password.
        import secrets as _secrets
        placeholder_hash = await hash_password(_secrets.token_urlsafe(32))
        cred_repo.create(user_id=user_id, hashed_password=placeholder_hash)
        db.flush()

    # admin_set_password updates the hash, resets lockout, revokes sessions.
    await CredentialService.admin_set_password(db, user_id=user_id, new_password=password)


async def seed_dev_users_async(
    db: Session,
    users_data: list = None,
    apply_password_fallback: bool = False,
) -> dict:
    """Seed development users into the database (async variant).

    Idempotent: existing users (matched by email) are left untouched in their
    user row; passwords are only updated when a new value is supplied.

    In LOCAL mode, if a user entry has a ``password`` key the credential is
    created/updated.  In FAKE mode, the ``password`` key is ignored and no
    credential row is written.

    Args:
        db: Database session.
        users_data: List of user dicts with ``email``, ``name``,
                    ``description``, and optional ``password`` keys.
                    If ``None``, uses ``DEV_USERS`` default list.
        apply_password_fallback: When ``True``, ``AICT_DEV_SEED_PASSWORD`` is
            applied to users that have no explicit password.  Should only be
            ``True`` when ``users_data`` came from ``--users`` / ``AICT_DEV_SEED_USERS``
            (i.e., explicitly-provided users), never when using the built-in
            ``DEV_USERS`` defaults.

    Returns:
        Dict with ``created``, ``existing`` user lists, and ``total`` count.
    """
    if users_data is None:
        users_data = DEV_USERS

    login_mode = current_login_mode()
    fallback_password: str | None = (
        os.getenv(SEED_PASSWORD_ENV, "").strip() or None
    ) if apply_password_fallback else None

    created_users = []
    updated_users = []

    for user_data in users_data:
        email = user_data["email"]
        name = user_data["name"]
        explicit_password: str | None = user_data.get("password")

        # Determine effective password for LOCAL mode.
        effective_password: str | None = explicit_password or (
            fallback_password if login_mode == "LOCAL" else None
        )

        # Ensure user row exists.
        existing_user = UserService.get_user_by_email(db, email)
        if existing_user:
            user = existing_user
            logger.info(
                "User already exists: %s (user_id: %s)",
                email,
                existing_user.user_id,
            )
            updated_users.append(user)
        else:
            user, created = UserService.get_or_create_user(db=db, email=email, name=name)
            if created:
                logger.info(
                    "Created dev user: %s (user_id: %s) - %s",
                    email,
                    user.user_id,
                    user_data.get("description", ""),
                )
                created_users.append(user)
            else:
                logger.info("User already exists: %s", email)
                updated_users.append(user)

        # Set/update credential in LOCAL mode when a password is available.
        if login_mode == "LOCAL" and effective_password:
            try:
                await _set_credential(db, user.user_id, effective_password)
                logger.info("Credential set for user: %s (user_id: %s)", email, user.user_id)
            except Exception as exc:
                logger.error(
                    "Failed to set credential for user %s: %s",
                    email,
                    exc,
                    exc_info=True,
                )
        elif login_mode == "LOCAL" and not effective_password:
            logger.info(
                "User %s seeded without password (no password source in LOCAL mode). "
                "Admin must issue a set-password link before this user can log in.",
                email,
            )

    return {
        "created": created_users,
        "existing": updated_users,
        "total": len(created_users) + len(updated_users),
    }


def seed_dev_users(
    db: Session,
    users_data: list = None,
    apply_password_fallback: bool = False,
) -> dict:
    """Synchronous wrapper around ``seed_dev_users_async``.

    Runs the async seeder on the current event loop (or creates one if none
    is running).  Caller is responsible for committing the session.

    Args:
        db: Database session.
        users_data: See ``seed_dev_users_async``.
        apply_password_fallback: See ``seed_dev_users_async``.

    Returns:
        See ``seed_dev_users_async``.
    """
    return asyncio.run(seed_dev_users_async(db, users_data, apply_password_fallback=apply_password_fallback))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seed_dev_users",
        description="Seed development users for FAKE/LOCAL login mode.",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Run non-interactively (skip confirmation prompts). Required when "
             "invoked without a TTY, e.g. 'docker compose exec -T'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Seed even if AICT_LOGIN is not FAKE/LOCAL. Use with care.",
    )
    parser.add_argument(
        "--users",
        metavar="SPEC",
        default=None,
        help="Comma-separated users to create, e.g. "
             "'admin@acme.com:Admin:Pass123!,dev@acme.com:Dev'. Overrides "
             f"{SEED_USERS_ENV} and the built-in defaults. "
             "In LOCAL mode a third ':password' field is supported.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Print the resolved user list and exit without writing anything.",
    )
    return parser


def _print_users(users: list) -> None:
    for user in users:
        description = user.get("description", "")
        has_pw = " [+password]" if user.get("password") else ""
        suffix = f" - {description}" if description else ""
        print(f"  - {user['email']:30s} {user['name']}{has_pw}{suffix}")


def main():
    """CLI entry point for user seeding."""
    args = _build_arg_parser().parse_args()

    users = resolve_users(args.users)
    is_explicit = _is_explicit_user_spec(args.users)
    login_mode = current_login_mode()
    fallback_password = os.getenv(SEED_PASSWORD_ENV, "").strip() or None

    print("\n" + "=" * 70)
    print("  Development User Seeding Utility")
    print("=" * 70 + "\n")

    if not users:
        print("No users to seed (empty list resolved). Nothing to do.\n")
        return

    if args.list_only:
        print(f"Login mode: {login_mode}")
        if fallback_password and login_mode == "LOCAL":
            print(f"Fallback password: {SEED_PASSWORD_ENV} is set")
        print(f"Resolved users ({len(users)}):\n")
        _print_users(users)
        print()
        return

    # Safety guard: only seed users in dev/local modes.
    if not is_seedable_mode() and not args.force:
        message = (
            f"AICT_LOGIN={login_mode} is not a development mode. "
            "Seeding dev users is only intended for FAKE/LOCAL. "
            "Re-run with --force to override."
        )
        if args.yes:
            print(f"ERROR: {message}\n")
            sys.exit(2)
        print(f"WARNING: {message}\n")
        if input("Continue anyway? (y/N): ").strip().lower() != "y":
            print("\nAborted.\n")
            return

    print(f"Login mode: {login_mode}")
    if login_mode == "LOCAL" and fallback_password and is_explicit:
        print(f"Fallback password: {SEED_PASSWORD_ENV} is set (applied to explicitly-provided users without a per-user password)")
    elif login_mode == "LOCAL" and fallback_password and not is_explicit:
        print(f"Note: {SEED_PASSWORD_ENV} is set but will NOT be applied to the built-in default users.")
    print(f"This will create/update the following users ({len(users)}):\n")
    _print_users(users)
    print("\n" + "-" * 70)

    if not args.yes:
        if input("\nProceed with seeding? (y/N): ").strip().lower() != "y":
            print("\nAborted.\n")
            return

    print("\nSeeding users...\n")

    db = SessionLocal()
    try:
        result = asyncio.run(seed_dev_users_async(db, users, apply_password_fallback=is_explicit))
        db.commit()

        print("\n" + "=" * 70)
        print("  Seeding Complete!")
        print("=" * 70)
        print(f"\n  Created:  {len(result['created'])} new users")
        print(f"  Existing: {len(result['existing'])} users already in database")
        print(f"  Total:    {result['total']} users ready for {login_mode} mode\n")

        if result["created"]:
            print("  Newly created users:")
            for user in result["created"]:
                print(f"    - {user.email} (ID: {user.user_id})")

        if login_mode == "LOCAL":
            print(
                "\n  In LOCAL mode, users with no password source will need an "
                "admin to issue a set-password link via the admin panel or:\n"
                "  POST /internal/admin/users/{user_id}/reset-link\n"
            )
        else:
            print("\n  These emails can now log in while AICT_LOGIN=FAKE.\n")

    except Exception as exc:
        db.rollback()
        logger.error("Error seeding users: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}\n")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
