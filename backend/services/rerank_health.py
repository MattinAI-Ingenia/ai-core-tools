"""Health recorder injected into rerank plans (the retrieval layer never imports services)."""

from models.rerank_service import RerankService
from services.service_health_service import ServiceHealthService
from tools.rerank.types import RerankTarget


class DbRerankHealthRecorder:
    """Writes rerank outcomes onto the ``rerank_service`` row (satisfies the runtime-checkable ``tools.retrieval`` recorder protocol)."""

    def record_success(self, target: RerankTarget) -> None:
        """Clear a recorded failure; free when the target snapshot shows the service healthy."""
        ServiceHealthService.record_success(RerankService, target)

    def record_failure(self, target: RerankTarget, message: str) -> None:
        """Record a short key-free failure message (throttled in SQL)."""
        ServiceHealthService.record_failure_message(RerankService, target.service_id, message)
