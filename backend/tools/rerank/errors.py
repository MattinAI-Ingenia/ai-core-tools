from typing import Optional

KINDS = {"auth", "http", "timeout", "connect", "invalid_response", "unknown_provider", "config", "busy"}


class RerankError(Exception):
    """Base rerank failure. `message` is short and never contains bodies, credentials or the key."""
    kind = "config"
    # False for local, non-service conditions (e.g. our own pool saturated): callers must not record
    # a service health failure for them.
    service_failure = True

    def __init__(self, message: str, kind: Optional[str] = None):
        super().__init__(message)
        self.message = message
        if kind is not None:
            self.kind = kind


class RerankLocalError(RerankError):
    """Local (non-service) failure: misuse, cancellation or an unmapped exception. Never a health failure."""
    kind = "config"
    service_failure = False


class RerankAuthError(RerankError):
    kind = "auth"


class RerankHTTPError(RerankError):
    kind = "http"

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RerankBusyError(RerankError):
    """Local connection pool saturated: says nothing about the reranker's health."""
    kind = "busy"
    service_failure = False


class RerankTimeoutError(RerankError):
    kind = "timeout"


class RerankConnectError(RerankError):
    kind = "connect"


class InvalidRerankResponseError(RerankError):
    kind = "invalid_response"


class UnknownRerankProviderError(RerankError):
    kind = "unknown_provider"
