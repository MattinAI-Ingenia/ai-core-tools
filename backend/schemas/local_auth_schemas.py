"""Pydantic v2 schemas for LOCAL auth mode endpoints.

All password fields are typed as ``SecretStr`` so that:
- Pydantic never includes them in ``model_dump()`` output by default.
- The Python repr never echoes the plain value.
- They are never accidentally serialised to JSON logs.

Password policy (NIST SP 800-63B aligned):
- Minimum length configurable via ``LOCAL_PASSWORD_MIN_LENGTH`` env var (default 12).
- Rejects passwords found in the common-password denylist loaded at import time.
- No mandatory composition rules (no "must contain uppercase, digit, symbol").
- Rejects passwords whose normalised form exactly matches the user's email local-part
  where the local-part can be injected (``SetPasswordRequest``, ``ChangePasswordRequest``).

Because the user's email is not always available at schema-validation time (e.g.
``AdminSetPasswordRequest``), the email-match check is an optional validator helper
``check_password_not_email`` that callers invoke explicitly when the email is known.
"""

from pydantic import BaseModel, EmailStr, SecretStr, field_validator
from utils.config import Config


# ---------------------------------------------------------------------------
# Email normalisation helper
# ---------------------------------------------------------------------------


def _normalise_email_value(v: object) -> object:
    """Strip whitespace and lowercase an email value.

    Used as a shared ``mode="before"`` field validator on all LOCAL auth
    schemas that store or look up an email address.  LOCAL emails are stored
    canonical-lowercase so they remain consistent with the normalisation
    applied at login time — preventing a stored ``Bob@Acme.com`` from becoming
    permanently unloggable when the login path lowercases to ``bob@acme.com``.

    Args:
        v: Raw email value (may be any type; non-strings are passed through so
            Pydantic's ``EmailStr`` validator can raise the appropriate error).

    Returns:
        Normalised string (strip + lower), or the original value unchanged.
    """
    return v.strip().lower() if isinstance(v, str) else v

# ---------------------------------------------------------------------------
# Common-password denylist
# ---------------------------------------------------------------------------
# Top ~200 passwords from the Have I Been Pwned / SecLists corpus.
# Using an inline frozenset avoids file-path dependencies in containers and
# makes the import entirely self-contained.  The set contains only lowercase
# normalised forms; comparison is case-insensitive.

_COMMON_PASSWORDS: frozenset[str] = frozenset({
    "password", "password1", "password123", "password1234", "password12345",
    "123456", "1234567", "12345678", "123456789", "1234567890",
    "111111", "11111111", "000000", "0000000000",
    "1q2w3e4r", "1q2w3e", "1q2w3e4r5t", "qwerty", "qwerty123", "qwertyuiop",
    "asdfghjkl", "zxcvbnm", "abcdefgh",
    "letmein", "welcome", "welcome1", "welcome123",
    "monkey", "monkey1", "dragon", "master", "sunshine", "princess",
    "shadow", "iloveyou", "trustno1", "superman", "batman",
    "football", "baseball", "soccer", "hockey", "basketball",
    "abc123", "abc1234", "abcd1234",
    "admin", "admin123", "administrator",
    "login", "login123", "passw0rd", "pass123", "pass1234",
    "test", "test123", "testing", "testing123",
    "hello", "hello123", "hello1234",
    "secret", "secret1", "secret123",
    "changeme", "change_me", "changeme123",
    "michael", "ashley", "jessica", "charlie", "andrew", "daniel",
    "joshua", "david", "james", "robert", "thomas",
    "nicole", "jessica", "hunter", "jennifer", "jordan",
    "111222333", "123123", "321321", "654321", "987654321",
    "!@#$%^&*", "qazwsx", "1qaz2wsx",
    "computer", "internet", "windows", "linux", "ubuntu",
    "samsung", "iphone", "android", "google", "amazon",
    "passpass", "passwordpassword",
    "mustang", "ferrari", "porsche",
    "lovely", "pretty", "beautiful", "gorgeous",
    "flower", "chocolate", "butterfly", "diamond",
    "yankees", "manchester", "chelsea", "arsenal", "liverpool",
    "madison", "jessica", "brandon", "taylor", "morgan",
    "summer", "winter", "spring", "autumn",
    "maggie", "bailey", "buster", "goldie",
    "pa$$w0rd", "p@ssword", "p@ssw0rd", "p@$$w0rd",
    "abc", "12345", "123", "1234", "54321",
    "1111", "11111", "111111111", "1111111111",
    "0987654321", "9876543210",
    "google123", "facebook", "twitter",
    "baseball1", "basketball1", "football1",
    "password2", "password01", "password11",
    "aaaaaa", "bbbbbb", "cccccc", "zzzzzzz",
    "qqqqqq", "pppppp",
    "abcabc", "123abc", "abc123abc",
    "matrix", "starwars", "pokemon", "minecraft",
    "naruto", "bleach", "dragon123",
    "love123", "love1234", "loveyou",
    "monkey123", "dragon123", "shadow123",
    "super123", "super1234",
    "blue123", "red123", "green123",
    "house123", "home123", "homepass",
    "work123", "office123",
    "sky123", "earth123",
    "user", "user123", "user1234", "users",
    "guest", "guest123",
    "root", "root123", "toor",
    "temp", "temp123", "temporary",
    "default", "demo", "demo123",
    "service", "support", "helpdesk",
    "netpass", "netword", "network",
    "company", "company1",
    "apple123", "orange123",
    "bear123", "tiger123", "lion123",
    "pass", "pass1", "pass12",
    "secure", "secure123", "security",
    "access", "access123",
    "system", "system123",
    "manager", "manager123",
    "account", "account123",
    "member", "member123",
    "developer", "develop",
    "michael1", "daniel1", "matthew1",
    "abcdefg", "aaaaaaaa",
    "mypassword", "mypass", "mypassword1",
    "newpassword", "newpass",
    "oldpassword", "oldpass",
    "wrongpassword",
    "passme", "passme123",
    "love", "loveme", "loveme123",
    "cats", "dogs", "birds",
    "fish", "horse", "rabbit",
    "minecraft1", "roblox", "fortnite",
    "gaming", "gamer", "gamer123",
    "hacker", "hacking",
})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MIN_LENGTH: int = Config.get_int_env_var("LOCAL_PASSWORD_MIN_LENGTH", default=12)


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def _validate_password_strength(plain: str) -> str:
    """Apply password policy and return the plain value if it passes.

    Args:
        plain: The candidate password string (already extracted from SecretStr
            by the field validator before this helper is called).

    Returns:
        The unmodified ``plain`` value when policy is satisfied.

    Raises:
        ValueError: Describing why the password was rejected (message is safe
            to surface to the end user — contains no internal detail).
    """
    if len(plain) < _MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {_MIN_LENGTH} characters long."
        )
    if len(plain.encode("utf-8")) > 72:
        raise ValueError("Password must be at most 72 bytes long.")
    if plain.lower() in _COMMON_PASSWORDS:
        raise ValueError(
            "This password is too common. Please choose a more unique password."
        )
    return plain


def check_password_not_email(plain: str, email: str) -> None:
    """Raise PasswordPolicyError when the password matches the user's email local-part.

    Call this explicitly in service layer when the email is available (e.g.
    inside ``consume_set_password_token`` and ``change_password``).

    Note: This function raises ``PasswordPolicyError`` (not ``CredentialError``)
    so that HTTP handlers can map it to HTTP 400 (policy rejection) independently
    from authentication failures (HTTP 401).  Schema-level validators that need a
    ``ValueError`` must perform the same check inline.

    Args:
        plain: The candidate password (plain text, already policy-validated).
        email: The user's email address.

    Raises:
        PasswordPolicyError: When the password (case-insensitive) matches the
            local-part of the email address.
    """
    # Import here to avoid a circular import: schemas → services → schemas.
    # This helper is only called from service-layer code, so the deferred import
    # is safe and the circular reference never materialises at module load time.
    from services.auth.credential_service import PasswordPolicyError  # noqa: PLC0415

    local_part = email.split("@")[0].lower()
    if local_part and plain.lower() == local_part:
        raise PasswordPolicyError(
            "Password must not match your email address."
        )


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SetPasswordRequest(BaseModel):
    """Request body for the set-password-via-token flow (admin-issued link).

    Used by both first-time password setup and admin-triggered resets.
    """

    token: str
    new_password: SecretStr

    @field_validator("token", mode="before")
    @classmethod
    def _strip_token(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("new_password", mode="before")
    @classmethod
    def _validate_new_password(cls, v: object) -> object:
        plain = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        _validate_password_strength(plain)
        return v


class ChangePasswordRequest(BaseModel):
    """Request body for an authenticated user changing their own password."""

    current_password: SecretStr
    new_password: SecretStr

    @field_validator("new_password", mode="before")
    @classmethod
    def _validate_new_password(cls, v: object) -> object:
        plain = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        _validate_password_strength(plain)
        return v


class LoginRequest(BaseModel):
    """Request body for LOCAL auth email+password login."""

    email: EmailStr
    password: SecretStr

    @field_validator("email", mode="before")
    @classmethod
    def _normalise_email(cls, v: object) -> object:
        """Strip whitespace and lowercase so throttle, lockout, and DB keys align."""
        return _normalise_email_value(v)


class AdminCreateUserRequest(BaseModel):
    """Request body for an admin creating a new LOCAL auth user account.

    No password field — the admin calls ``issue_set_password_token`` separately
    to generate a first-time setup link.

    Email normalisation: LOCAL emails are stored canonical-lowercase (strip +
    lower) to stay consistent with the normalisation applied at login time.
    An email stored with capital letters would be permanently unloggable because
    ``LoginRequest`` lowercases before the DB lookup.
    """

    email: EmailStr
    name: str

    @field_validator("email", mode="before")
    @classmethod
    def _normalise_email(cls, v: object) -> object:
        """Strip whitespace and lowercase the email before storage."""
        return _normalise_email_value(v)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class AdminSetPasswordRequest(BaseModel):
    """Request body for an admin forcibly setting a user's password directly.

    Used for emergency password resets from the admin panel when the email
    token flow is not appropriate.
    """

    new_password: SecretStr

    @field_validator("new_password", mode="before")
    @classmethod
    def _validate_new_password(cls, v: object) -> object:
        plain = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        _validate_password_strength(plain)
        return v
