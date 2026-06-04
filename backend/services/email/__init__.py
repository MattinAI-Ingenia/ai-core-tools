"""Email sender abstraction for LOCAL auth mode.

Provides an ``EmailSender`` protocol, a ``NoopEmailSender`` for deployments
without SMTP configured, an ``SmtpEmailSender`` for full outbound email, and
``get_email_sender()`` as the application-wide factory.

Usage::

    sender = get_email_sender()
    sender.send(to="user@example.com", subject="...", body_html="<p>...</p>")
"""

from services.email.sender import (
    EmailSender,
    NoopEmailSender,
    SmtpEmailSender,
    get_email_sender,
    smtp_configured,
)

__all__ = [
    "EmailSender",
    "NoopEmailSender",
    "SmtpEmailSender",
    "get_email_sender",
    "smtp_configured",
]
