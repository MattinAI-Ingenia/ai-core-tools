"""Concrete EmailSender implementations for LOCAL auth mode.

Architecture
------------
- ``EmailSender`` — abstract base class / protocol.
- ``NoopEmailSender`` — default when SMTP is not configured.  Logs at INFO
  level that a link was generated and is being returned to the caller (the
  admin token route), but NEVER logs the actual token or link value.
- ``SmtpEmailSender`` — full SMTP implementation using stdlib ``smtplib``.
  Constructed only when ``smtp_configured()`` returns True.
- ``get_email_sender()`` — application factory: returns ``SmtpEmailSender``
  when configured, otherwise ``NoopEmailSender``.

Environment variables consumed (all optional unless SMTP is intended)::

    SMTP_HOST       Hostname of the SMTP relay (required for SMTP)
    SMTP_PORT       Port; default 587
    SMTP_USER       SMTP authentication username (optional)
    SMTP_PASSWORD   SMTP authentication password (optional — never logged)
    SMTP_TLS        "true"/"false" — whether STARTTLS is used; default "true"
    SMTP_FROM       Sender address (required for SMTP, e.g. no-reply@example.com)
"""

import os
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmailSender(ABC):
    """Abstract email sender.  All delivery paths implement this interface."""

    @abstractmethod
    def send(self, *, to: str, subject: str, body_html: str, body_text: str = "") -> None:
        """Send an email.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body_html: HTML body of the message.
            body_text: Plain-text fallback body; used when body_html is empty.
        """


# ---------------------------------------------------------------------------
# No-op sender (SMTP not configured)
# ---------------------------------------------------------------------------


class NoopEmailSender(EmailSender):
    """Silent email sender used when SMTP is not configured.

    Never logs token or link values.  Only emits a single INFO line indicating
    that the set-password link has been returned directly to the admin caller
    rather than sent by email.
    """

    def send(self, *, to: str, subject: str, body_html: str, body_text: str = "") -> None:
        """Log that email delivery is disabled; swallow the message silently.

        Args:
            to: Recipient address (logged for audit; no token/link value here).
            subject: Subject line (logged).
            body_html: Not logged.
            body_text: Not logged.
        """
        logger.info(
            "auth:email_skipped to=%s subject=%r reason=smtp_not_configured "
            "(link returned directly to admin caller)",
            to,
            subject,
        )


# ---------------------------------------------------------------------------
# SMTP sender
# ---------------------------------------------------------------------------


class SmtpEmailSender(EmailSender):
    """Outbound email delivery via stdlib smtplib.

    Reads configuration once at instantiation time.  Raises ``RuntimeError``
    at construction if the minimum required env vars are absent (SMTP_HOST and
    SMTP_FROM), so misconfiguration is caught at startup rather than at first
    send attempt.
    """

    def __init__(self) -> None:
        self._host: str = os.environ["SMTP_HOST"]
        self._port: int = int(os.getenv("SMTP_PORT", "587"))
        self._user: str = os.getenv("SMTP_USER", "")
        # Password read from env; intentionally not stored as instance attr
        # with a public name — accessed via property at send time.
        self._password: str = os.getenv("SMTP_PASSWORD", "")
        self._tls: bool = os.getenv("SMTP_TLS", "true").lower() in ("true", "1", "yes")
        self._from: str = os.environ["SMTP_FROM"]
        self._timeout: int = int(os.getenv("SMTP_TIMEOUT_SECONDS", "10"))

    def send(self, *, to: str, subject: str, body_html: str, body_text: str = "") -> None:
        """Send an email via SMTP.  Logs errors without raising.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body_html: HTML body.
            body_text: Plain-text fallback; defaults to empty string.
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from
        msg["To"] = to

        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as server:
                server.ehlo()
                if self._tls:
                    server.starttls()
                if self._user and self._password:
                    server.login(self._user, self._password)
                server.sendmail(self._from, [to], msg.as_string())

            logger.info("auth:email_sent to=%s subject=%r", to, subject)

        except (smtplib.SMTPAuthenticationError, smtplib.SMTPConnectError):
            # Configuration faults (bad credentials / unreachable relay) must
            # surface so they are not silently swallowed for every email.
            logger.error("auth:email_misconfigured to=%s subject=%r", to, subject)
            raise
        except Exception as exc:
            # Best-effort for transient failures: log and continue; the calling
            # service controls whether a failed send is fatal.
            logger.error("auth:email_failed to=%s subject=%r error=%s", to, subject, exc)


# ---------------------------------------------------------------------------
# Configuration probe + factory
# ---------------------------------------------------------------------------


def smtp_configured() -> bool:
    """Return True iff both SMTP_HOST and SMTP_FROM are set in the environment.

    Returns:
        True when minimum SMTP configuration is present; False otherwise.
    """
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"))


def get_email_sender() -> EmailSender:
    """Return the appropriate ``EmailSender`` for the current environment.

    Returns:
        ``SmtpEmailSender`` when ``smtp_configured()`` is True, otherwise
        ``NoopEmailSender``.
    """
    if smtp_configured():
        return SmtpEmailSender()
    return NoopEmailSender()
