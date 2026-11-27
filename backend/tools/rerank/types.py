from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class RerankTarget:
    """ORM-free snapshot of a resolved rerank service (safe to use off the DB session)."""
    service_id: Optional[int]
    app_id: Optional[int]
    name: str
    provider: str
    endpoint: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)
    last_error: Optional[str] = None
    # True only for platform (system-scoped) services; fail-closed: anything else gets the SSRF guard.
    system: bool = False

    @classmethod
    def from_service(cls, service: Any, system: Optional[bool] = None) -> "RerankTarget":
        """Snapshot any object exposing the service attributes (ORM row or test stand-in).

        `system` defaults to True only for a real `RerankService` row whose `app_id` is NULL; any other
        object (e.g. a test stand-in) is treated as app-owned unless the caller states otherwise.
        """
        if system is None:
            from models.rerank_service import RerankService
            system = isinstance(service, RerankService) and getattr(service, "app_id", None) is None
        return cls(
            system=bool(system) and getattr(service, "app_id", None) is None,
            service_id=getattr(service, "service_id", None),
            app_id=getattr(service, "app_id", None),
            name=getattr(service, "name", None) or "",
            provider=getattr(service, "provider", None) or "",
            endpoint=getattr(service, "endpoint", None) or "",
            model=getattr(service, "model_name", None) or "",
            api_key=getattr(service, "api_key", None),
            last_error=getattr(service, "last_error", None),
        )


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float
