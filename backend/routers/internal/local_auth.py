"""LOCAL auth HTTP endpoints.

Exposes the email+password authentication surface for ``AICT_LOGIN=LOCAL`` mode.
These routes are registered only when ``AuthConfig.LOGIN_MODE == 'LOCAL'``.

Route list:
- POST /internal/auth/login          — password login → httpOnly cookies
- POST /internal/auth/refresh        — rotate refresh token → new cookies
- POST /internal/auth/logout         — revoke current device session
- POST /internal/auth/logout-all     — revoke all sessions for the user
- POST /internal/auth/set-password   — consume set-password token (first-use / reset)
- POST /internal/auth/change-password — authenticated self-service password change

CSRF behaviour:
  All endpoints are mounted inside ``internal_router`` which carries the
  ``enforce_csrf`` dependency.  The login endpoint has no ``access_token``
  cookie when it is called (the user is not yet logged in), so
  ``enforce_csrf`` hits the ``if not access_token_cookie: return`` no-op path —
  no false-positive 403 is raised and no CSRF header is required on login.
  All subsequent cookie-bearing endpoints (refresh, logout, logout-all,
  change-password) do carry the ``access_token`` cookie and therefore require
  the ``X-CSRF-Token`` header to match the ``csrf_token`` cookie.
  The set-password endpoint is invoked from a one-time link before the user
  holds a session cookie, so it is also effectively exempt by the same no-op.

Domain exception → HTTP status mapping (enforced in every handler):
  CredentialError        → 401
  AccountLockedError     → 429
  TokenError             → 400 (bad/used) | 401 (expired, per caller context)
  InactiveAccountError   → 403  (subclass of CredentialError; checked first)
  RefreshTokenError      → 401
  UserAlreadyExistsError → 409
"""

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from lks_idprovider import AuthContext
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from db.database import get_db
from routers.internal.auth_utils import get_current_user_oauth
from schemas.local_auth_schemas import ChangePasswordRequest, LoginRequest, SetPasswordRequest
from services.auth.credential_service import (
    AccountLockedError,
    CredentialError,
    InactiveAccountError,
    PasswordPolicyError,
    TokenError,
    CredentialService,
)
from services.auth.login_throttle import (
    ERR_TOO_MANY_REQUESTS,
    enforce_login_throttle,
    extract_client_ip,
    hit_account_throttle,
    reset_account_throttle,
)
from services.auth.refresh_service import RefreshService, RefreshTokenError
from utils.auth_cookies import (
    COOKIE_REFRESH_TOKEN,
    clear_session_cookies,
    set_session_cookies,
)
from utils.config import is_omniadmin
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Error message constants
# ---------------------------------------------------------------------------

_ERR_INVALID_CREDENTIALS = "Invalid email or password."
_ERR_ACCOUNT_LOCKED = "Account is temporarily locked. Please try again later."
_ERR_ACCOUNT_INACTIVE = "This account has been deactivated."
_ERR_TOKEN_INVALID = "The link is invalid or has already been used."
_ERR_TOKEN_EXPIRED = "The link has expired. Please request a new one."
_ERR_REFRESH_INVALID = "Session is invalid. Please log in again."
_ERR_CHANGE_PASSWORD_WRONG = "Current password is incorrect."
_ERR_PASSWORD_POLICY = "Password does not meet requirements."
_ERR_SESSION_START = "Unable to start session. Please try again."
# ERR_TOO_MANY_REQUESTS is imported from login_throttle to keep one canonical string.

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class LoginResponse(BaseModel):
    """Minimal user information returned on successful login.

    Access and refresh tokens are delivered exclusively via httpOnly cookies
    (AC-7); they are NEVER included in this response body.
    """

    model_config = ConfigDict(from_attributes=True)

    user_id: int
    email: str
    name: Optional[str] = None
    is_omniadmin: bool


class SetPasswordResponse(BaseModel):
    """Response for the set-password flow — no session is auto-created."""

    message: str


class ChangePasswordResponse(BaseModel):
    """Response for the change-password flow."""

    message: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["local-auth"])


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="LOCAL auth login",
    description="Authenticate with email and password. Session cookies are set on success.",
    dependencies=[Depends(enforce_login_throttle)],
)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> LoginResponse:
    """Authenticate via email+password and issue session cookies.

    On success sets three cookies:
    - ``access_token``  — httpOnly, short-lived JWT.
    - ``refresh_token`` — httpOnly, long-lived opaque token.
    - ``csrf_token``    — readable by JS (not httpOnly), echoed back as ``X-CSRF-Token``.

    No token value appears in the response body (AC-7).

    CSRF note: This endpoint has no ``access_token`` cookie on entry, so the
    ``enforce_csrf`` dependency installed on the internal router is a no-op
    here — no ``X-CSRF-Token`` header is required for the login call itself.

    Throttle: the per-IP limit is enforced via ``Depends(enforce_login_throttle)``.
    The per-account limit is enforced here (before ``authenticate``) because the
    account key derives from the parsed request body.

    Args:
        body: ``LoginRequest`` with email and password.
        request: FastAPI request (used to extract User-Agent and client IP).
        response: FastAPI response (cookies are attached here).
        db: Synchronous database session.

    Returns:
        ``LoginResponse`` with minimal user info (no tokens in body).

    Raises:
        HTTPException 401: Invalid credentials.
        HTTPException 429: Account temporarily locked or throttle exceeded.
        HTTPException 403: Account deactivated.
    """
    user_agent: Optional[str] = request.headers.get("user-agent")
    client_ip: Optional[str] = extract_client_ip(request)

    # Per-account throttle — keyed on the submitted email.  Applied before
    # authenticate() so a flood of wrong-password attempts for a known address
    # is stopped at the route layer (the DB-backed lockout in authenticate()
    # is a separate, complementary control).
    # ``body.email`` is already normalised (strip+lower) by the LoginRequest
    # schema validator, so the throttle key matches the DB lookup key exactly.
    account_state = hit_account_throttle(str(body.email))
    if not account_state.allowed:
        logger.warning(
            "auth:throttle_account_exceeded email=%s ip=%s",
            body.email,
            client_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=ERR_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(account_state.retry_after_seconds)},
        )

    try:
        user = await CredentialService.authenticate(
            db, email=str(body.email), password=body.password.get_secret_value()
        )
    except InactiveAccountError:
        # Must be checked before the generic CredentialError since it is a subclass.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_ERR_ACCOUNT_INACTIVE,
        )
    except AccountLockedError as exc:
        logger.warning("auth:login_locked email=%s ip=%s", body.email, client_ip)
        # Compute Retry-After from the lockout expiry carried on the exception.
        # Guard against None (should not occur) and naive datetimes (stored
        # without tzinfo by SQLite in tests; the service normalises to UTC-aware
        # before raising, so this is a belt-and-suspenders fallback only).
        locked_until = exc.locked_until
        if locked_until is not None and locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        if locked_until is not None:
            retry_after = max(0, int((locked_until - datetime.now(timezone.utc)).total_seconds()))
        else:
            retry_after = 60  # conservative fallback when timestamp unavailable
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=_ERR_ACCOUNT_LOCKED,
            headers={"Retry-After": str(retry_after)},
        )
    except CredentialError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_ERR_INVALID_CREDENTIALS,
        )

    # Successful login: clear the per-account throttle so this user is not
    # penalised for their own earlier typos in the same minute window.
    reset_account_throttle(str(body.email))

    try:
        access_token, refresh_token_plain, _ = RefreshService.issue_session(
            db,
            user,
            user_agent=user_agent,
            ip=client_ip,
        )
        set_session_cookies(response, access_token, refresh_token_plain)
    except Exception as exc:
        logger.error("auth:http_login_session_error user_id=%s — %s", user.user_id, type(exc).__name__)
        clear_session_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_ERR_SESSION_START,
        )

    logger.info(
        "auth:http_login_success user_id=%s ip=%s",
        user.user_id,
        client_ip,
    )

    return LoginResponse(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        is_omniadmin=is_omniadmin(user.email),
    )


# ---------------------------------------------------------------------------
# POST /refresh
# ---------------------------------------------------------------------------


@router.post(
    "/refresh",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Rotate refresh token",
    description=(
        "Exchange the current refresh-token cookie for a fresh token pair. "
        "Requires X-CSRF-Token header (double-submit CSRF protection is active "
        "because the access_token cookie is present on this call)."
    ),
    dependencies=[Depends(enforce_login_throttle)],
)
async def refresh(
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    refresh_token_cookie: Optional[str] = Cookie(default=None, alias=COOKIE_REFRESH_TOKEN),
) -> LoginResponse:
    """Rotate refresh token and issue new session cookies.

    The per-IP throttle is enforced via ``Depends(enforce_login_throttle)``
    to prevent bulk refresh-token probing from a single IP.

    Args:
        request: FastAPI request for User-Agent / IP extraction.
        response: FastAPI response to attach new cookies.
        db: Synchronous database session.
        refresh_token_cookie: Value of the ``refresh_token`` httpOnly cookie.

    Returns:
        ``LoginResponse`` with minimal user info.

    Raises:
        HTTPException 401: Token absent, invalid, expired, rotated, or reuse detected.
        HTTPException 429: Per-IP throttle exceeded.
    """
    if not refresh_token_cookie:
        clear_session_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_ERR_REFRESH_INVALID,
        )

    user_agent: Optional[str] = request.headers.get("user-agent")
    client_ip: Optional[str] = extract_client_ip(request)

    try:
        access_token, new_refresh_plain, _ = RefreshService.rotate(
            db,
            refresh_token_cookie,
            user_agent=user_agent,
            ip=client_ip,
        )
    except RefreshTokenError as exc:
        logger.warning("auth:refresh_failed reason=%s ip=%s", exc, client_ip)
        clear_session_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_ERR_REFRESH_INVALID,
        )

    set_session_cookies(response, access_token, new_refresh_plain)
    logger.debug("auth:token_rotated ip=%s", client_ip)

    # We need a user to build the response — parse email from the new access token.
    # The rotate() call already validates the user; load via the auth dependency
    # would require an extra DB round-trip.  Instead, decode the access token we
    # just minted (it is guaranteed fresh and valid).
    from utils.local_auth_tokens import decode_access_token
    import jwt

    try:
        payload = decode_access_token(access_token)
    except jwt.InvalidTokenError:
        # Extremely unlikely: we just minted this token, but guard anyway.
        logger.error("auth:refresh_decode_failure — newly minted token failed decode")
        clear_session_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_ERR_REFRESH_INVALID,
        )

    user_email: str = payload.get("email", "")
    user_id: int = int(payload.get("sub", 0))
    user_name: Optional[str] = payload.get("name")

    return LoginResponse(
        user_id=user_id,
        email=user_email,
        name=user_name,
        is_omniadmin=is_omniadmin(user_email),
    )


# ---------------------------------------------------------------------------
# POST /logout
# ---------------------------------------------------------------------------


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Logout (current device)",
    description=(
        "Revoke the current device's refresh token and clear session cookies. "
        "Requires X-CSRF-Token header when the access_token cookie is present."
    ),
)
async def logout(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    refresh_token_cookie: Optional[str] = Cookie(default=None, alias=COOKIE_REFRESH_TOKEN),
) -> dict:
    """Revoke the current session and clear all auth cookies.

    Idempotent: if the cookie is absent or the token already revoked the
    response is still 200 with cookies cleared (prevents enumeration).

    Args:
        response: FastAPI response to clear cookies.
        db: Synchronous database session.
        refresh_token_cookie: Value of the ``refresh_token`` httpOnly cookie.

    Returns:
        JSON ``{"message": "Logged out successfully"}``.
    """
    if refresh_token_cookie:
        RefreshService.revoke_current(db, refresh_token_cookie)

    clear_session_cookies(response)
    logger.info("auth:http_logout")
    return {"message": "Logged out successfully."}


# ---------------------------------------------------------------------------
# POST /logout-all
# ---------------------------------------------------------------------------


@router.post(
    "/logout-all",
    status_code=status.HTTP_200_OK,
    summary="Logout all devices",
    description=(
        "Revoke all refresh tokens for the authenticated user (all devices). "
        "Requires a valid session (access_token cookie or Bearer) and X-CSRF-Token header."
    ),
)
async def logout_all(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
) -> dict:
    """Revoke all sessions for the authenticated user and clear cookies.

    Args:
        response: FastAPI response to clear cookies.
        db: Synchronous database session.
        auth_context: Resolved authentication context (current user).

    Returns:
        JSON ``{"message": "All sessions revoked."}``.
    """
    user_id = int(auth_context.identity.id)
    RefreshService.revoke_all(db, user_id)
    clear_session_cookies(response)
    logger.info("auth:http_logout_all user_id=%s", user_id)
    return {"message": "All sessions revoked."}


# ---------------------------------------------------------------------------
# POST /set-password
# ---------------------------------------------------------------------------


@router.post(
    "/set-password",
    response_model=SetPasswordResponse,
    status_code=status.HTTP_200_OK,
    summary="Set password via one-time token",
    description=(
        "Consume a set-password token (first-time setup or admin-triggered reset) "
        "and apply the new password. No session is created — the user must log in "
        "after completing this flow. "
        "CSRF no-op: no access_token cookie is present before the user logs in."
    ),
    dependencies=[Depends(enforce_login_throttle)],
)
async def set_password(
    body: SetPasswordRequest,
    db: Annotated[Session, Depends(get_db)],
) -> SetPasswordResponse:
    """Consume a one-time set-password token and apply the new password.

    Marks the user as verified (they followed the admin-issued link) and
    resets any lockout state.  No session cookies are issued here —
    the user must log in explicitly afterwards.

    Args:
        body: ``SetPasswordRequest`` with ``token`` and ``new_password``.
        db: Synchronous database session.

    Returns:
        ``SetPasswordResponse`` with a success message.

    Raises:
        HTTPException 400: Token invalid or already used, or password policy violation.
        HTTPException 401: Token expired.
    """
    try:
        await CredentialService.consume_set_password_token(
            db, body.token, body.new_password.get_secret_value()
        )
    except PasswordPolicyError:
        # Password matches email local-part: policy rejection, not auth failure.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_ERR_PASSWORD_POLICY,
        )
    except TokenError as exc:
        msg = str(exc)
        # Distinguish expired from invalid/used tokens.
        if "expired" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=_ERR_TOKEN_EXPIRED,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_ERR_TOKEN_INVALID,
        )
    except CredentialError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_ERR_TOKEN_INVALID,
        )

    return SetPasswordResponse(message="Password set successfully. You can now log in.")


# ---------------------------------------------------------------------------
# POST /change-password
# ---------------------------------------------------------------------------


@router.post(
    "/change-password",
    response_model=ChangePasswordResponse,
    status_code=status.HTTP_200_OK,
    summary="Change own password",
    description=(
        "Authenticated self-service password change. "
        "Verifies the current password, applies the new one, and revokes all "
        "refresh sessions — the user must re-authenticate on all devices. "
        "Requires X-CSRF-Token header."
    ),
)
async def change_password(
    body: ChangePasswordRequest,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
) -> ChangePasswordResponse:
    """Change the authenticated user's password.

    On success all refresh tokens for the user are revoked; the caller's own
    session is also invalidated — the frontend should redirect to the login page.

    Args:
        body: ``ChangePasswordRequest`` with ``current_password`` and ``new_password``.
        db: Synchronous database session.
        auth_context: Resolved authentication context (current user).

    Returns:
        ``ChangePasswordResponse`` with a success message.

    Raises:
        HTTPException 400: Password policy violation (new password matches email local-part).
        HTTPException 401: Current password is incorrect.
    """
    from repositories.user_repository import UserRepository

    user_id = int(auth_context.identity.id)
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_ERR_INVALID_CREDENTIALS,
        )

    try:
        await CredentialService.change_password(
            db,
            user,
            current_password=body.current_password.get_secret_value(),
            new_password=body.new_password.get_secret_value(),
        )
    except PasswordPolicyError:
        # New password matches email local-part: policy rejection.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_ERR_PASSWORD_POLICY,
        )
    except CredentialError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_ERR_CHANGE_PASSWORD_WRONG,
        )

    logger.info("auth:http_change_password_success user_id=%s", user_id)
    return ChangePasswordResponse(
        message="Password changed successfully. Please log in again on all devices."
    )
