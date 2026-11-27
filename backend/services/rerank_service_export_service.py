"""Service for exporting Rerank Services."""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from repositories.rerank_service_repository import RerankServiceRepository
from schemas.export_schemas import ExportRerankServiceSchema, RerankServiceExportFileSchema
from services.base_export_service import BaseExportService

logger = logging.getLogger(__name__)


class RerankServiceExportService(BaseExportService):
    """Service for exporting Rerank Services (the API key is never exported)."""

    def __init__(self, session: Session):
        """Initialize Rerank Service export service.

        Args:
            session: SQLAlchemy database session
        """
        super().__init__(session)
        self.rerank_service_repo = RerankServiceRepository()

    def export_rerank_service(
        self, service_id: int, app_id: int, user_id: Optional[int] = None
    ) -> RerankServiceExportFileSchema:
        """Export a Rerank Service to a JSON structure.

        Args:
            service_id: ID of the rerank service to export
            app_id: App ID (tenant check: the service must belong to this app)
            user_id: User ID for the metadata (optional)

        Returns:
            RerankServiceExportFileSchema: Export file structure

        Raises:
            ValueError: If the service is not found in the app
        """
        service = self.rerank_service_repo.get_by_id_and_app_id(self.session, service_id, app_id)
        if not service:
            raise ValueError(f"Rerank Service with ID {service_id} not found in app {app_id}")

        export_service = ExportRerankServiceSchema(
            name=service.name,
            api_key=None,  # CRITICAL: always strip the API key
            provider=service.provider,
            model_name=service.model_name,
            endpoint=service.endpoint,
            end_of_service_at=service.end_of_service_at,
            deprecation_notice=service.deprecation_notice,
        )
        logger.info("Exported Rerank Service '%s' (ID: %s)", service.name, service_id)
        return RerankServiceExportFileSchema(
            metadata=self.create_metadata(user_id, app_id),
            rerank_service=export_service,
        )
