"""JWT token utilities for LOCAL auth mode (email+password SaaS login).

This module is the single point of truth for minting and decoding LOCAL-issuer
access tokens.  It intentionally uses a different issuer/audience pair from the
DEV mode (``ia-core-tools-dev``) so that a dev-mode token or an OIDC token is
**rejected** by ``decode_access_token`` at the issuer/audience check level.

Constants are loaded once at import time from environment variables; the module
has no FastAPI dependencies so it can be exercised in unit tests without starting
the application.

Public API: ``mint_access_token``, ``decode_access_token``.
The ``generate_local_auth_token`` shim is intentionally NOT present here;
it remains in ``utils.dev_auth`` until step_008 completes the decoder cutover
and dev_auth.py is deleted.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from utils.logger import get_logger
from utils.secret_key import get_secret_key

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALGORITHM = "HS256"

_ISSUER = "mattin-local-auth"
_AUDIENCE = "mattin-internal"

_ACCESS_TTL_MINUTES: int = int(os.getenv("LOCAL_ACCESS_TTL_MINUTES", "15"))
_LEEWAY_SECONDS: int = int(os.getenv("LOCAL_TOKEN_LEEWAY_SECONDS", "30"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mint_access_token(user_id: int, email: str, name: str | None) -> tuple[str, datetime]:
    """Mint a signed LOCAL access token for a user.

    Creates an HS256 JWT signed with the application secret key.  The token
    carries the minimum claims required to identify the user and validate the
    token's origin; no sensitive data is included.

    Args:
        user_id: Numeric primary key of the User record.
        email: User's verified email address.
        name: User's display name; falls back to email when None.

    Returns:
        A two-tuple of ``(token_string, expires_at)`` where ``expires_at`` is a
        timezone-aware UTC datetime.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=_ACCESS_TTL_MINUTES)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "name": name or email,
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "iat": now,
        "exp": expires_at,
        "jti": uuid.uuid4().hex,
    }

    token: str = jwt.encode(payload, get_secret_key(), algorithm=_ALGORITHM)
    return token, expires_at


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and fully validate a LOCAL access token.

    Validates the signature, expiry (with configurable clock-skew leeway),
    issuer, and audience.  Tokens issued by the DEV login endpoint
    (``ia-core-tools-dev`` issuer) or by an OIDC provider will be rejected
    because they carry a different issuer/audience.

    Args:
        token: The raw JWT string.

    Returns:
        The decoded claims dict.

    Raises:
        jwt.ExpiredSignatureError: When the token's ``exp`` has passed.
        jwt.InvalidTokenError: For any other validation failure (wrong issuer,
            wrong audience, bad signature, malformed token, etc.).
    """
    return jwt.decode(
        token,
        get_secret_key(),
        algorithms=[_ALGORITHM],
        issuer=_ISSUER,
        audience=_AUDIENCE,
        leeway=timedelta(seconds=_LEEWAY_SECONDS),
    )


