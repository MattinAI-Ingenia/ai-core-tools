"""HTTP-only cookie helpers for LOCAL auth mode session transport.

Centralises all cookie attribute constants and the set/clear helpers so that
every call site uses the same secure defaults.  Only imported by the LOCAL auth
code paths; OIDC and FAKE modes do not use cookies.

Cookie design:
- ``access_token``  — httpOnly, SameSite=Lax, Path=/            (JS cannot read it)
- ``refresh_token`` — httpOnly, SameSite=Lax, Path=/internal/auth (narrow path)
- ``csrf_token``    — NOT httpOnly, SameSite=Lax, Path=/         (JS must read it)

The ``Secure`` flag is enabled by default (``AUTH_COOKIE_SECURE=true``) and may
be disabled via env var only for local HTTP development (never in production).
"""

import secrets
from typing import Optional

from fastapi import Response

from utils.config import Config
from utils.local_auth_tokens import ACCESS_TTL_MINUTES as _TOKEN_ACCESS_TTL_MINUTES
from utils.logger import get_logger
from services.auth.refresh_service import REFRESH_TTL_DAYS as _SERVICE_REFRESH_TTL_DAYS

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cookie configuration constants
# ---------------------------------------------------------------------------

#: Set ``AUTH_COOKIE_SECURE=false`` (or ``0``/``no``/``off``) to opt-out of
#: the Secure flag for local HTTP development only.  Must never be false in
#: production deployments.  Uses Config.get_bool_env_var so that "0", "false",
#: "no", and "off" all correctly disable the flag (unlike a raw != "false"
#: comparison which would treat "0" as True).
_COOKIE_SECURE: bool = Config.get_bool_env_var("AUTH_COOKIE_SECURE", default=True)

#: Access-token TTL in minutes — imported directly from local_auth_tokens so
#: cookie max-age and JWT exp share a single source of truth.
_ACCESS_TTL_MINUTES: int = _TOKEN_ACCESS_TTL_MINUTES

#: Refresh-token TTL in days — imported directly from refresh_service so
#: cookie max-age and token exp share a single source of truth.
_REFRESH_TTL_DAYS: int = _SERVICE_REFRESH_TTL_DAYS

# Cookie name constants
COOKIE_ACCESS_TOKEN = "access_token"
COOKIE_REFRESH_TOKEN = "refresh_token"
COOKIE_CSRF_TOKEN = "csrf_token"

# Cookie path constants
_PATH_ROOT = "/"
_PATH_AUTH = "/internal/auth"

# Max-age values derived from TTL constants
_ACCESS_MAX_AGE: int = _ACCESS_TTL_MINUTES * 60
_REFRESH_MAX_AGE: int = _REFRESH_TTL_DAYS * 24 * 60 * 60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_csrf_token() -> str:
    """Generate a cryptographically random CSRF token.

    Returns:
        A URL-safe base64 token string with 256 bits of entropy.
    """
    return secrets.token_urlsafe(32)


def set_session_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: Optional[str] = None,
) -> None:
    """Attach all three session cookies to an outgoing response.

    Sets ``access_token`` and ``refresh_token`` as httpOnly cookies (not
    readable by JavaScript) and ``csrf_token`` as a readable cookie so the
    frontend can extract it and echo it back via the ``X-CSRF-Token`` header
    on mutating requests.

    Token values are never logged.

    Args:
        response: The FastAPI ``Response`` (or ``JSONResponse``) to attach
            cookies to.
        access_token: The signed LOCAL access JWT.
        refresh_token: The opaque refresh token wire string.
        csrf_token: Optional CSRF token string.  If ``None`` a new token is
            generated with ``generate_csrf_token()``.
    """
    if csrf_token is None:
        csrf_token = generate_csrf_token()

    response.set_cookie(
        key=COOKIE_ACCESS_TOKEN,
        value=access_token,
        max_age=_ACCESS_MAX_AGE,
        path=_PATH_ROOT,
        secure=_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    response.set_cookie(
        key=COOKIE_REFRESH_TOKEN,
        value=refresh_token,
        max_age=_REFRESH_MAX_AGE,
        path=_PATH_AUTH,
        secure=_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    # csrf_token must NOT be httpOnly so the frontend JS can read it.
    response.set_cookie(
        key=COOKIE_CSRF_TOKEN,
        value=csrf_token,
        max_age=_REFRESH_MAX_AGE,  # CSRF cookie lives as long as the session
        path=_PATH_ROOT,
        secure=_COOKIE_SECURE,
        httponly=False,
        samesite="lax",
    )
    logger.debug(
        "Session cookies set (secure=%s, access_max_age=%ds, refresh_max_age=%ds)",
        _COOKIE_SECURE,
        _ACCESS_MAX_AGE,
        _REFRESH_MAX_AGE,
    )


def clear_session_cookies(response: Response) -> None:
    """Remove all three session cookies from the client.

    Path attributes must match exactly those used in ``set_session_cookies``
    or the browser will not delete the cookies.

    Args:
        response: The FastAPI ``Response`` to attach the delete directives to.
    """
    response.delete_cookie(
        key=COOKIE_ACCESS_TOKEN,
        path=_PATH_ROOT,
        secure=_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        key=COOKIE_REFRESH_TOKEN,
        path=_PATH_AUTH,
        secure=_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        key=COOKIE_CSRF_TOKEN,
        path=_PATH_ROOT,
        secure=_COOKIE_SECURE,
        httponly=False,
        samesite="lax",
    )
    logger.debug("Session cookies cleared")
