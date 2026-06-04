"""CSRF double-submit cookie protection for the internal API.

This module exposes a single FastAPI dependency, ``enforce_csrf``, that
implements the double-submit cookie pattern:

1. On every mutating request (POST / PUT / PATCH / DELETE) the browser sends
   the ``csrf_token`` cookie automatically.
2. The JavaScript frontend also reads that cookie (it is NOT httpOnly) and
   echoes its value in the ``X-CSRF-Token`` request header.
3. This dependency compares both values with a constant-time compare and
   rejects the request with HTTP 403 on any mismatch.

Scope and bypass rules
----------------------
The dependency is attached ONLY to the internal router (``/internal``).
The public (``/public/v1``) and MCP (``/mcp/v1``) routers use API-key auth
and are never touched here.

The check is also a deliberate **no-op** when the request is not
cookie-authenticated:

- No ``access_token`` cookie is present AND
- The request carries an ``Authorization`` header (Bearer token) or an
  ``X-API-KEY`` header.

This preserves full backward-compatibility for:
- Swagger UI using Bearer tokens directly.
- Any backend-to-backend caller using ``Authorization: Bearer``.
- Public API callers that somehow route through ``/internal`` tooling paths.

Only requests that carry the ``access_token`` cookie — i.e. browser sessions
created by the LOCAL auth login endpoint — are subject to CSRF enforcement.
"""

import hmac
from typing import Optional

from fastapi import Cookie, Header, HTTPException, Request, status

from utils.logger import get_logger

logger = get_logger(__name__)

# HTTP methods that mutate server state and therefore require CSRF protection.
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Pre-session auth-entry endpoints that must NOT require a CSRF token even when
# the browser still holds a stale ``access_token`` cookie from a previous
# session. These are credential- or one-time-token-gated and (re)establish a
# session by overwriting the cookies, so a matching CSRF token from a prior
# session is both unavailable to a legitimate client and meaningless to verify.
# Cross-site abuse is already blocked by the cookies' ``SameSite=Lax`` flag plus
# the mandatory credential/token in the request body. Path is matched by suffix
# so it is robust to the ``/internal`` mount prefix.
_CSRF_EXEMPT_SUFFIXES = ("/auth/login", "/auth/set-password")

# Generic error message — never reveal whether the cookie or header was absent
# vs. mismatched, to avoid leaking information to an attacker.
_CSRF_ERROR_MESSAGE = "CSRF validation failed"


async def enforce_csrf(
    request: Request,
    access_token_cookie: Optional[str] = Cookie(default=None, alias="access_token"),
    csrf_cookie: Optional[str] = Cookie(default=None, alias="csrf_token"),
    x_csrf_token: Optional[str] = Header(default=None, alias="X-CSRF-Token"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
) -> None:
    """FastAPI dependency that enforces CSRF double-submit for cookie sessions.

    Intended to be used as a router-level dependency on the internal router
    only.  Should never be attached to the public or MCP routers.

    Args:
        request: Incoming FastAPI request (used to read the HTTP method).
        access_token_cookie: Value of the ``access_token`` httpOnly cookie.
        csrf_cookie: Value of the ``csrf_token`` readable cookie.
        x_csrf_token: Value of the ``X-CSRF-Token`` request header.
        authorization: Value of the ``Authorization`` request header.
        x_api_key: Value of the ``X-API-KEY`` request header.

    Raises:
        HTTPException 403: When a cookie-authenticated mutating request lacks
            a valid ``X-CSRF-Token`` header that matches the ``csrf_token``
            cookie.
    """
    if request.method not in _MUTATING_METHODS:
        return  # Safe methods (GET, HEAD, OPTIONS) need no CSRF check.

    # Pre-session auth-entry endpoints are exempt regardless of cookie state.
    if request.url.path.endswith(_CSRF_EXEMPT_SUFFIXES):
        return

    # If there is no access_token cookie the request is not a cookie session
    # (it uses Bearer or API-key auth).  Skip CSRF enforcement entirely.
    if not access_token_cookie:
        return

    # The request IS cookie-authenticated.  The CSRF token must be present in
    # both the cookie and the header and they must match.
    if not csrf_cookie or not x_csrf_token:
        logger.warning(
            "CSRF check failed: missing token — method=%s path=%s "
            "has_cookie=%s has_header=%s",
            request.method,
            request.url.path,
            bool(csrf_cookie),
            bool(x_csrf_token),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_CSRF_ERROR_MESSAGE,
        )

    # Constant-time comparison to prevent timing attacks.
    tokens_match = hmac.compare_digest(
        csrf_cookie.encode("utf-8"),
        x_csrf_token.encode("utf-8"),
    )
    if not tokens_match:
        logger.warning(
            "CSRF check failed: token mismatch — method=%s path=%s",
            request.method,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_CSRF_ERROR_MESSAGE,
        )

    logger.debug("CSRF check passed — method=%s path=%s", request.method, request.url.path)
