"""Thin, synchronous wrapper around the Azure Blob Storage SDK.

This is the **only** module in the codebase allowed to import
``azure.storage.blob``. Everything here is synchronous by design — callers
(the ingestion service) run these functions in a background thread, never on
the asyncio event loop.

Security notes:
    - Every outbound call goes through ``build_container_client``, which
      re-validates ``cfg.account_url`` against the anti-SSRF allowlist
      (``utils.blob_url_guard``) even if the caller already validated it once.
    - No ``logger.*`` call in this module ever interpolates ``cfg.sas_token``
      or a raw exception. Every ``BlobConnectionError.message`` is produced
      through ``sanitize_azure_error`` before it is raised, logged, or
      returned to a client.
"""
import os
import re
import socket
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional
from urllib.parse import quote

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.storage.blob import ContainerClient

from utils.blob_url_guard import assert_allowed_account_url
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_RETRY_TOTAL = 2
_MAX_SANITIZED_LENGTH = 500

_QUERY_STRING_RE = re.compile(r'(https?://[^\s"\'<>]+?)\?[^\s"\'<>]*', re.IGNORECASE)
_SHARED_ACCESS_SIGNATURE_RE = re.compile(r"SharedAccessSignature=[^&\s\"'<>]*", re.IGNORECASE)
_SIG_PARAM_RE = re.compile(r"\bsig=[^&\s\"'<>]*", re.IGNORECASE)
_SE_PARAM_RE = re.compile(r"\bse=[^&\s\"'<>]*", re.IGNORECASE)


@dataclass(frozen=True)
class BlobSourceConfig:
    """Request-scoped description of an Azure Blob Storage source. Never persisted."""

    account_url: str
    container: str
    prefix: Optional[str]
    auth_mode: str  # "ANONYMOUS" | "SAS_TOKEN"
    sas_token: Optional[str]


@dataclass(frozen=True)
class RemoteBlob:
    """A single blob returned by ``list_blobs``."""

    name: str
    url: str
    size_bytes: Optional[int]
    content_type: Optional[str]
    etag: Optional[str]
    last_modified: Optional[datetime]


class BlobConnectionError(Exception):
    """Raised for any Azure SDK / connectivity failure in this module.

    ``code`` is a short machine-readable identifier a caller maps to an HTTP
    status. ``message`` must already be sanitized (see ``sanitize_azure_error``)
    — it is safe to log or return to a client as-is.
    """

    VALID_CODES = (
        "CONTAINER_NOT_FOUND",
        "AUTH_FAILED",
        "LISTING_FORBIDDEN",
        "ACCOUNT_UNREACHABLE",
        "TIMEOUT",
        "INVALID_CONFIG",
        "DOWNLOAD_FAILED",
        "FILE_TOO_LARGE",
    )

    def __init__(self, code: str, message: str):
        if code not in self.VALID_CODES:
            raise ValueError(f"Invalid BlobConnectionError code: {code!r}")
        self.code = code
        self.message = message
        super().__init__(message)


def sanitize_azure_error(exc_or_msg, secret: Optional[str] = None) -> str:
    """Strip query strings and known SAS parameters from an error message.

    Every ``BlobConnectionError.message`` must be produced through this
    function before it is raised, logged, or returned. It is what keeps a
    request-scoped ``sas_token`` from ever reaching a log line or an API
    response (NFR-3/AC-9).

    Args:
        exc_or_msg: The exception (its ``str()``) or raw message to sanitize.
        secret: The SAS token in play for this call, if any. Redacted from the
            text verbatim (with and without a leading ``?``) in addition to
            the pattern-based redactions below.

    Returns:
        A sanitized string, capped at 500 characters.
    """
    text = str(exc_or_msg)
    text = _QUERY_STRING_RE.sub(r"\1", text)
    text = _SHARED_ACCESS_SIGNATURE_RE.sub("SharedAccessSignature=***", text)
    text = _SIG_PARAM_RE.sub("sig=***", text)
    text = _SE_PARAM_RE.sub("se=***", text)

    if secret:
        text = text.replace(secret, "***")
        stripped = secret.lstrip("?")
        if stripped and stripped != secret:
            text = text.replace(stripped, "***")

    return text[:_MAX_SANITIZED_LENGTH]


def classify_error(exc: Exception) -> str:
    """Map an Azure SDK / networking exception to a ``BlobConnectionError`` code.

    Order matters: SDK exception types are checked first (most specific),
    then HTTP status code as a fallback, then heuristics on the exception's
    class name/text for timeouts and connectivity — azure-core wraps both DNS
    failures and socket timeouts inside ``ServiceRequestError`` /
    ``ServiceResponseError`` without a dedicated timeout exception class.

    Args:
        exc: The exception raised by the Azure SDK (or the transport below it).

    Returns:
        One of ``BlobConnectionError.VALID_CODES`` (excluding ``INVALID_CONFIG``
        and ``FILE_TOO_LARGE``, which are raised directly by this module, not
        classified from an SDK exception).
    """
    if isinstance(exc, ClientAuthenticationError):
        return "AUTH_FAILED"
    if isinstance(exc, ResourceNotFoundError):
        return "CONTAINER_NOT_FOUND"

    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return "AUTH_FAILED"
    if status_code == 403:
        return "LISTING_FORBIDDEN"
    if status_code == 404:
        return "CONTAINER_NOT_FOUND"

    exc_type_name = type(exc).__name__.lower()
    exc_text = str(exc).lower()
    if isinstance(exc, socket.timeout) or "timeout" in exc_type_name or "timed out" in exc_text:
        return "TIMEOUT"
    if isinstance(exc, (ServiceRequestError, ServiceResponseError, ConnectionError)):
        return "ACCOUNT_UNREACHABLE"

    return "DOWNLOAD_FAILED"


def build_container_client(cfg: BlobSourceConfig) -> ContainerClient:
    """Build a ``ContainerClient`` for ``cfg``, re-validated against the anti-SSRF allowlist.

    Args:
        cfg: The blob source configuration.

    Returns:
        A configured ``ContainerClient``. No network call is made by this
        function itself.

    Raises:
        BlobUrlRejected: If ``cfg.account_url`` fails the anti-SSRF check.
        BlobConnectionError: ``INVALID_CONFIG`` if ``auth_mode`` is unsupported,
            ``SAS_TOKEN`` is selected without a token, or the SDK itself
            rejects the configuration (e.g. an empty ``container``).
    """
    account_url = assert_allowed_account_url(cfg.account_url)

    if cfg.auth_mode == "ANONYMOUS":
        credential = None
    elif cfg.auth_mode == "SAS_TOKEN":
        if not cfg.sas_token:
            raise BlobConnectionError(
                "INVALID_CONFIG", sanitize_azure_error("SAS_TOKEN auth mode requires a non-empty sas_token", None)
            )
        credential = cfg.sas_token[1:] if cfg.sas_token.startswith("?") else cfg.sas_token
    else:
        raise BlobConnectionError(
            "INVALID_CONFIG", sanitize_azure_error(f"Unsupported auth_mode: {cfg.auth_mode}", cfg.sas_token)
        )

    timeout_seconds = Config.get_int_env_var(
        "BLOB_INGEST_DOWNLOAD_TIMEOUT_SECONDS", int(Config.DEFAULTS["BLOB_INGEST_DOWNLOAD_TIMEOUT_SECONDS"])
    )

    try:
        return ContainerClient(
            account_url=account_url,
            container_name=cfg.container,
            credential=credential,
            retry_total=_DEFAULT_RETRY_TOTAL,
            connection_timeout=timeout_seconds,
            read_timeout=timeout_seconds,
            # The blob REST surface used here (list/download) never needs a redirect, and the
            # SDK's default RedirectPolicy would follow one without re-checking it against our
            # anti-SSRF allowlist — it only binds the first hop.
            permit_redirects=False,
        )
    except ValueError as exc:
        raise BlobConnectionError("INVALID_CONFIG", sanitize_azure_error(exc, cfg.sas_token)) from None


def list_blobs(cfg: BlobSourceConfig, extensions: set[str], *, page_size: int, max_items: int) -> Iterator[RemoteBlob]:
    """List blobs under ``cfg.prefix`` whose extension is in ``extensions``.

    Eagerly builds the container client (and therefore runs the anti-SSRF and
    auth-mode validation) before returning, so a caller gets ``BlobUrlRejected``
    / ``BlobConnectionError`` at call time — not from inside a ``for`` loop the
    first time the returned iterator is advanced. The actual page fetching
    happens lazily in the returned generator.

    Args:
        cfg: The blob source configuration.
        extensions: Lowercase extensions (including the leading dot, e.g.
            ``.pdf``) to keep. Empty set means no filtering. Blobs with any
            other extension are skipped without being yielded.
        page_size: Number of blobs requested per page from the service.
        max_items: Stop yielding once this many matching blobs have been produced.

    Returns:
        An iterator of ``RemoteBlob``, in listing order. Pages are fetched
        lazily via the SDK's auto-paging iterator — a caller that stops
        iterating early never triggers the remaining pages.

    Raises:
        BlobUrlRejected: If ``cfg.account_url`` fails the anti-SSRF check.
        BlobConnectionError: ``INVALID_CONFIG`` if ``auth_mode``/``container``
            is invalid, or any other Azure SDK failure, sanitized and classified.
    """
    container_client = build_container_client(cfg)
    # Recomputed (cheaply — resolve=False skips the DNS lookup already done by
    # build_container_client) rather than reading container_client.url: the SDK's
    # ContainerClient.url re-appends the raw SAS token for SAS_TOKEN auth, which would
    # otherwise leak the secret into RemoteBlob.url / Resource.extra_metadata (AC-9).
    account_url = assert_allowed_account_url(cfg.account_url, resolve=False)
    return _iter_blobs(container_client, account_url, cfg, extensions, page_size=page_size, max_items=max_items)


def _iter_blobs(
    container_client: ContainerClient,
    account_url: str,
    cfg: BlobSourceConfig,
    extensions: set[str],
    *,
    page_size: int,
    max_items: int,
) -> Iterator[RemoteBlob]:
    try:
        pages = container_client.list_blobs(name_starts_with=cfg.prefix, results_per_page=page_size).by_page()
        page_iterator = iter(pages)
        yielded = 0
        while yielded < max_items:
            try:
                page = next(page_iterator)
            except StopIteration:
                return
            for blob in page:
                if yielded >= max_items:
                    return
                extension = os.path.splitext(blob.name)[1].lower()
                if extensions and extension not in extensions:
                    continue
                content_settings = getattr(blob, "content_settings", None)
                content_type = getattr(content_settings, "content_type", None) if content_settings else None
                yield RemoteBlob(
                    name=blob.name,
                    url=f"{account_url}/{quote(cfg.container, safe='')}/{quote(blob.name, safe='/')}",
                    size_bytes=blob.size,
                    content_type=content_type,
                    etag=blob.etag,
                    last_modified=blob.last_modified,
                )
                yielded += 1
    except (AzureError, OSError) as exc:
        raise BlobConnectionError(classify_error(exc), sanitize_azure_error(exc, cfg.sas_token)) from None


def _cleanup_partial(dest_path: str) -> None:
    try:
        if os.path.exists(dest_path):
            os.remove(dest_path)
    except OSError:
        logger.warning(f"Could not remove partial download at {dest_path}")


def download_to_path(cfg: BlobSourceConfig, blob_name: str, dest_path: str, *, max_bytes: Optional[int] = None) -> int:
    """Stream a blob to ``dest_path``. Never loads the blob fully into memory.

    Args:
        cfg: The blob source configuration.
        blob_name: Full path of the blob within the container.
        dest_path: Local filesystem path to write to.
        max_bytes: If set, the blob's reported size is checked before any
            content is downloaded (a blob larger than this raises
            ``FILE_TOO_LARGE`` without transferring any bytes), and the actual
            bytes written to disk are checked again after the download
            completes — a second, defense-in-depth check that also catches a
            ``None``/understated reported size.

    Returns:
        The number of bytes actually written to ``dest_path``.

    Raises:
        BlobUrlRejected: If ``cfg.account_url`` fails the anti-SSRF check.
        BlobConnectionError: On any Azure SDK failure or when the blob exceeds
            ``max_bytes``. The partial file, if any, is removed before raising.
    """
    container_client = build_container_client(cfg)
    try:
        blob_client = container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob()

        size = downloader.properties.size
        if max_bytes is not None and size is not None and size > max_bytes:
            raise BlobConnectionError(
                "FILE_TOO_LARGE",
                sanitize_azure_error(
                    f"Blob '{blob_name}' ({size} bytes) exceeds the {max_bytes}-byte limit", cfg.sas_token
                ),
            )

        with open(dest_path, "wb") as dest_file:
            downloader.readinto(dest_file)

        actual_size = os.path.getsize(dest_path)
        if max_bytes is not None and actual_size > max_bytes:
            _cleanup_partial(dest_path)
            raise BlobConnectionError(
                "FILE_TOO_LARGE",
                sanitize_azure_error(
                    f"Blob '{blob_name}' ({actual_size} bytes) exceeds the {max_bytes}-byte limit", cfg.sas_token
                ),
            )
        return actual_size
    except BlobConnectionError:
        _cleanup_partial(dest_path)
        raise
    except AzureError as exc:
        _cleanup_partial(dest_path)
        raise BlobConnectionError(classify_error(exc), sanitize_azure_error(exc, cfg.sas_token)) from None
    except OSError as exc:
        _cleanup_partial(dest_path)
        raise BlobConnectionError(classify_error(exc), sanitize_azure_error(exc, cfg.sas_token)) from None
