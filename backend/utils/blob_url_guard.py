"""Anti-SSRF guard for Azure Blob Storage account URLs.

Every ``account_url`` accepted from a client is attacker-influenced input: it
picks the outbound destination of a server-initiated HTTP request. This
module enforces, in order, that the URL (1) uses ``https``, (2) is not an IP
literal (checked before the allowlist so it always surfaces its own specific
code, and so it remains an independent defense-in-depth layer if the
allowlist is ever misconfigured too loosely), (3) targets a host in an
operator-controlled allowlist, and (4) — when ``resolve=True`` — resolves
exclusively to globally routable addresses, blocking cloud metadata endpoints
(``169.254.169.254``) and other private/loopback/link-local ranges.
"""
import ipaddress
import socket
from urllib.parse import urlsplit

from utils.config import Config

_VALID_CODES = (
    "SCHEME_NOT_ALLOWED",
    "HOST_NOT_ALLOWED",
    "IP_LITERAL_NOT_ALLOWED",
    "PRIVATE_ADDRESS_NOT_ALLOWED",
    "DNS_RESOLUTION_FAILED",
    "MALFORMED_URL",
)


class BlobUrlRejected(Exception):
    """Raised when an ``account_url`` fails an anti-SSRF check.

    ``code`` is a short machine-readable identifier a caller can map to an
    HTTP status (typically 422). ``message`` is safe to return to the client.
    """

    VALID_CODES = _VALID_CODES

    def __init__(self, code: str, message: str):
        if code not in self.VALID_CODES:
            raise ValueError(f"Invalid BlobUrlRejected code: {code!r}")
        self.code = code
        self.message = message
        super().__init__(message)


def normalize_account_url(raw: str) -> str:
    """Parse and canonicalize an Azure Blob Storage account URL.

    Args:
        raw: The raw ``account_url`` supplied by the caller.

    Returns:
        A canonical ``https://host[:port]`` string — no path, query or
        fragment (those belong to ``container``/``prefix``, not the account
        URL), host lowercased and IDNA-encoded.

    Raises:
        BlobUrlRejected: ``MALFORMED_URL`` if ``raw`` is not a parseable
            absolute URL with both a scheme and a host; ``SCHEME_NOT_ALLOWED``
            if the scheme isn't exactly ``https``.
    """
    if not raw or not isinstance(raw, str):
        raise BlobUrlRejected("MALFORMED_URL", "account_url must be a non-empty string")

    try:
        parsed = urlsplit(raw.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise BlobUrlRejected("MALFORMED_URL", f"account_url could not be parsed: {exc}") from None

    if not parsed.scheme or not hostname:
        raise BlobUrlRejected("MALFORMED_URL", "account_url must be an absolute URL with a scheme and a host")

    if parsed.scheme.lower() != "https":
        raise BlobUrlRejected(
            "SCHEME_NOT_ALLOWED", f"Scheme '{parsed.scheme}' is not allowed; only 'https' is permitted"
        )

    try:
        idna_host = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise BlobUrlRejected("MALFORMED_URL", f"account_url host could not be encoded: {exc}") from None

    # IPv6 literals pass through IDNA encoding unchanged (still contain ':') and must be
    # re-bracketed, or re-parsing the rebuilt URL below (and again in assert_allowed_account_url)
    # silently truncates or nulls out the host instead of raising a clean BlobUrlRejected.
    host_repr = f"[{idna_host}]" if ":" in idna_host else idna_host
    return f"https://{host_repr}{f':{port}' if port else ''}"


def _allowed_host_suffixes() -> tuple[str, ...]:
    raw = Config.get_env_var("BLOB_ALLOWED_HOST_SUFFIXES", Config.DEFAULTS["BLOB_ALLOWED_HOST_SUFFIXES"])
    return tuple(suffix.strip().lower() for suffix in raw.split(",") if suffix.strip())


def _host_matches_suffix(host: str, suffix: str) -> bool:
    """Return True if ``host`` equals ``suffix`` or is a proper subdomain of it.

    Normalizes a leading dot off ``suffix`` (operators naturally write
    ``mycorp.com`` rather than ``.mycorp.com``) and then requires a
    label-boundary match, so ``evilmycorp.com`` never matches ``mycorp.com``.
    """
    s = suffix.lstrip(".")
    return host == s or host.endswith("." + s)


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _assert_resolves_to_public_addresses(host: str) -> None:
    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise BlobUrlRejected("DNS_RESOLUTION_FAILED", f"Could not resolve host '{host}': {exc}") from None

    resolved_addresses = {info[4][0] for info in addr_infos}
    for address in resolved_addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise BlobUrlRejected(
                "DNS_RESOLUTION_FAILED", f"Host '{host}' resolved to an unparsable address"
            ) from None
        if not ip.is_global:
            raise BlobUrlRejected(
                "PRIVATE_ADDRESS_NOT_ALLOWED", f"Host '{host}' resolves to a non-public address"
            )


def assert_allowed_account_url(raw: str, *, resolve: bool = True) -> str:
    """Validate ``raw`` against the anti-SSRF allowlist and return its normalized form.

    Args:
        raw: The raw ``account_url`` supplied by the caller.
        resolve: When True (default), also resolve the host via DNS and reject
            it unless every resolved address is globally routable. Callers
            that already validated the URL earlier in the same request may
            pass ``resolve=False`` to skip a redundant DNS lookup, but the SDK
            call site must always validate with ``resolve=True`` immediately
            before use (defense in depth, AD-4).

    Returns:
        The normalized ``account_url`` (see ``normalize_account_url``).

    Raises:
        BlobUrlRejected: If any check fails. See the class docstring for codes.
    """
    normalized = normalize_account_url(raw)
    host = urlsplit(normalized).hostname
    if not host:
        raise BlobUrlRejected("MALFORMED_URL", "account_url could not be re-parsed after normalization")

    # IP-literal check runs before the suffix check (not after, as a plain reading of "in
    # order" might suggest) so it fires with its own specific code rather than an IP always
    # being rejected "by accident" via HOST_NOT_ALLOWED (an IP string essentially never ends
    # with an allowed suffix like ".blob.core.windows.net"). This also keeps it a real,
    # independent defense-in-depth layer if BLOB_ALLOWED_HOST_SUFFIXES is ever misconfigured
    # loosely enough (e.g. an accidental blank entry) to match an IP-shaped host.
    if _is_ip_literal(host):
        raise BlobUrlRejected(
            "IP_LITERAL_NOT_ALLOWED", f"Host '{host}' is an IP literal; only DNS hostnames are allowed"
        )

    if not any(_host_matches_suffix(host, suffix) for suffix in _allowed_host_suffixes()):
        raise BlobUrlRejected("HOST_NOT_ALLOWED", f"Host '{host}' is not in the allowed host suffix list")

    if resolve:
        _assert_resolves_to_public_addresses(host)

    return normalized
