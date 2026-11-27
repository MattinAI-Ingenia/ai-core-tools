"""Service for importing Rerank Services."""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from core.export_constants import PLACEHOLDER_API_KEY, validate_export_version
from models.rerank_service import RerankProvider, RerankService
from schemas.export_schemas import ExportRerankServiceSchema, RerankServiceExportFileSchema
from schemas.import_schemas import ComponentType, ConflictMode, ImportSummarySchema
from services.rerank_service_service import RerankServiceError, RerankServiceService, _check_key_reuse
from services.service_lifecycle_service import ServiceLifecycleService
from tools.rerank.endpoints import normalize_endpoint
from utils.service_lifecycle import SERVICE_KIND_RERANK

logger = logging.getLogger(__name__)

class RerankServiceConflictError(ValueError):
    """A rerank service with the same name already exists in the app (FAIL conflict mode)."""


class RerankServiceImportService:
    """Service for importing Rerank Services into an app."""

    def __init__(self, session: Session):
        """Initialize Rerank Service import service.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def get_by_name_and_app(self, name: str, app_id: int) -> Optional[RerankService]:
        """Get a rerank service by name within an app (tenant-scoped)."""
        return (
            self.session.query(RerankService)
            .filter(RerankService.name == name, RerankService.app_id == app_id)
            .first()
        )

    @staticmethod
    def checked_endpoint(item: ExportRerankServiceSchema, effective_api_key: Optional[str] = None) -> str:
        """Normalise the endpoint and apply the app-scope SSRF policy.

        Raises:
            ValueError: With a clear message when the provider or endpoint is rejected.
        """
        if item.provider not in {p.value for p in RerankProvider}:
            raise ValueError(f"Rerank Service '{item.name}': unknown provider '{item.provider}'")
        if not item.endpoint:
            raise ValueError(f"Rerank Service '{item.name}': endpoint is required")
        try:
            endpoint = normalize_endpoint(item.endpoint)
            RerankServiceService.validate_endpoint(endpoint, effective_api_key, app_scoped=True)
        except RerankServiceError as exc:
            raise ValueError(f"Rerank Service '{item.name}': {exc.message}") from None
        except ValueError as exc:
            raise ValueError(f"Rerank Service '{item.name}': {exc}") from None
        return endpoint

    def import_rerank_service(
        self,
        export_data: RerankServiceExportFileSchema,
        app_id: int,
        conflict_mode: ConflictMode = ConflictMode.FAIL,
        new_name: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> ImportSummarySchema:
        """Import a Rerank Service.

        Args:
            export_data: Parsed export file
            app_id: Target app ID
            conflict_mode: How to handle name conflicts
            new_name: Optional custom name (rename mode)
            user_id: Acting user (lifecycle audit)

        Returns:
            ImportSummarySchema: Import operation summary

        Raises:
            ValueError: On conflict in FAIL mode, or a rejected provider/endpoint.
        """
        validate_export_version(export_data.metadata.export_version)
        item = export_data.rerank_service

        final_name = item.name
        existing = self.get_by_name_and_app(final_name, app_id)

        if existing:
            if conflict_mode == ConflictMode.FAIL:
                raise RerankServiceConflictError(f"Rerank Service '{final_name}' already exists in app {app_id}")
            if conflict_mode == ConflictMode.RENAME:
                if new_name:
                    final_name = new_name
                else:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    final_name = f"{item.name} (imported {date_str})"
                    counter = 1
                    while self.get_by_name_and_app(final_name, app_id):
                        final_name = f"{item.name} (imported {date_str} {counter})"
                        counter += 1
            elif conflict_mode == ConflictMode.OVERRIDE:
                endpoint = self.checked_endpoint(item, existing.api_key)
                # The stored key must never follow an endpoint to a different origin.
                try:
                    _check_key_reuse(existing, endpoint)
                except RerankServiceError as exc:
                    raise ValueError(f"Rerank Service '{item.name}': {exc.message}") from None
                existing.provider = item.provider
                existing.model_name = item.model_name
                existing.endpoint = endpoint
                ServiceLifecycleService.apply_lifecycle_fields(existing, SERVICE_KIND_RERANK, item, user_id)
                # The stored (user configured) API key is kept.
                self.session.add(existing)
                self.session.commit()
                self.session.refresh(existing)
                logger.info("Overrode Rerank Service '%s' (ID: %s)", existing.name, existing.service_id)
                return ImportSummarySchema(
                    component_type=ComponentType.RERANK_SERVICE,
                    component_id=existing.service_id,
                    component_name=existing.name,
                    mode=conflict_mode,
                    created=False,
                    conflict_detected=True,
                    warnings=["Existing API key preserved"],
                    next_steps=[],
                )

        endpoint = self.checked_endpoint(item, PLACEHOLDER_API_KEY)
        new_service = RerankService(
            app_id=app_id,
            name=final_name,
            api_key=PLACEHOLDER_API_KEY,  # the user must configure the real key
            provider=item.provider,
            model_name=item.model_name,
            endpoint=endpoint,
            is_platform_default=False,
            create_date=datetime.now(),
        )
        ServiceLifecycleService.apply_lifecycle_fields(new_service, SERVICE_KIND_RERANK, item, user_id)
        self.session.add(new_service)
        self.session.commit()
        self.session.refresh(new_service)
        if new_service.end_of_service_at is not None:
            ServiceLifecycleService.log_change(
                new_service, SERVICE_KIND_RERANK, None, new_service.end_of_service_at, user_id
            )
        logger.info("Imported Rerank Service '%s' (ID: %s)", final_name, new_service.service_id)

        return ImportSummarySchema(
            component_type=ComponentType.RERANK_SERVICE,
            component_id=new_service.service_id,
            component_name=new_service.name,
            mode=conflict_mode,
            created=True,
            conflict_detected=existing is not None,
            warnings=[
                "API key set to placeholder 'CHANGE_ME' — update it before using this service"
            ],
            next_steps=["Configure API key for the imported Rerank Service"],
        )
