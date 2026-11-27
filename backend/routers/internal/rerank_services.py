import json
import re
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from lks_idprovider import AuthContext
from sqlalchemy.orm import Session

from db.database import get_db
from routers.controls.role_authorization import AppRole, require_min_role
from schemas.export_schemas import RerankServiceExportFileSchema
from schemas.import_schemas import ConflictMode, ImportResponseSchema
from schemas.rerank_service_schemas import (
    CreateUpdateRerankServiceSchema,
    RerankServiceDetailSchema,
    RerankServiceListItemSchema,
    RerankServiceOptionsSchema,
)
from services.rerank_service_service import (
    RERANK_SERVICE_NOT_FOUND_MSG,
    RerankServiceError,
    RerankServiceService,
)
from schemas.service_lifecycle_schemas import ServiceUsageSchema
from services.rerank_service_export_service import RerankServiceExportService
from services.rerank_service_import_service import RerankServiceConflictError, RerankServiceImportService
from services.service_lifecycle_service import ServiceLifecycleService
from utils.service_lifecycle import SERVICE_KIND_RERANK
from utils.logger import get_logger

from .auth_utils import get_current_user_oauth

logger = get_logger(__name__)

rerank_services_router = APIRouter()


def _http_error(exc: RerankServiceError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail())


@rerank_services_router.get(
    "/", summary="List rerank services", tags=["Rerank Services"], response_model=List[RerankServiceListItemSchema]
)
async def list_rerank_services(
    app_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """List the app's rerank services plus the read-only platform services."""
    return RerankServiceService.get_rerank_services_list(db, app_id)


# ==================== STATIC ROUTES (before {service_id}) ====================


@rerank_services_router.get(
    "/options", summary="Rerank service selector options", tags=["Rerank Services"],
    response_model=RerankServiceOptionsSchema,
)
async def get_rerank_service_options(
    app_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """Options for selectors plus app, platform and effective defaults."""
    return RerankServiceService.options(db, app_id)


@rerank_services_router.post(
    "/test-connection", summary="Test rerank service connection with config", tags=["Rerank Services"]
)
async def test_rerank_service_connection_with_config(
    app_id: int,
    config: CreateUpdateRerankServiceSchema,
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("administrator"))],
    service_id: Annotated[
        Optional[int], Query(description="Edit-mode: recover the stored API key when the request sends a masked one")
    ] = None,
):
    """Send a minimal rerank request to the configured endpoint. Nothing is persisted."""
    try:
        return await RerankServiceService.test_connection_with_config(
            db, config, app_scoped=True, app_id=app_id, service_id=service_id
        )
    except RerankServiceError as exc:
        raise _http_error(exc)
    except Exception as exc:
        logger.error("Error testing rerank service connection: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error testing rerank service")


@rerank_services_router.post(
    "/import", summary="Import Rerank Service", tags=["Rerank Services", "Export/Import"],
    response_model=ImportResponseSchema, status_code=status.HTTP_201_CREATED,
)
async def import_rerank_service(
    app_id: int,
    file: Annotated[UploadFile, File(...)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("administrator"))],
    conflict_mode: Annotated[ConflictMode, Query()] = ConflictMode.FAIL,
    new_name: Annotated[Optional[str], Query()] = None,
):
    """Import a Rerank Service from a JSON file (the API key becomes a CHANGE_ME placeholder)."""
    try:
        content = await file.read()
        export_data = RerankServiceExportFileSchema(**json.loads(content))
        summary = await run_in_threadpool(
            RerankServiceImportService(db).import_rerank_service,
            export_data, app_id, conflict_mode, new_name, int(auth_context.identity.id),
        )
        return ImportResponseSchema(
            success=True,
            message=f"Rerank Service '{summary.component_name}' imported successfully",
            summary=summary,
        )
    except RerankServiceConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Rerank service import error: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Import failed")


# ==================== DYNAMIC ROUTES ====================


@rerank_services_router.post(
    "/{service_id}/export", summary="Export Rerank Service", tags=["Rerank Services", "Export/Import"],
    status_code=status.HTTP_200_OK,
)
async def export_rerank_service(
    app_id: int,
    service_id: int,
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """Export a Rerank Service configuration to a JSON file (without the API key)."""
    try:
        export_data = RerankServiceExportService(db).export_rerank_service(
            service_id, app_id, int(auth_context.identity.id)
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    except Exception as exc:
        logger.error("Rerank service export error: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Export failed")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", export_data.rerank_service.name)
    filename = f"{safe_name}_rerank_service.json"
    return JSONResponse(
        content=export_data.model_dump(mode="json"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@rerank_services_router.get(
    "/{service_id}/used-by", summary="List what depends on a rerank service", tags=["Rerank Services"],
    response_model=ServiceUsageSchema,
)
async def get_rerank_service_used_by(
    app_id: int,
    service_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """Agents, scheduled tasks and app defaults of this app using the rerank service."""
    usage = await run_in_threadpool(ServiceLifecycleService.get_usage, db, SERVICE_KIND_RERANK, service_id, app_id)
    if usage is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=RERANK_SERVICE_NOT_FOUND_MSG)
    return usage


@rerank_services_router.get(
    "/{service_id}", summary="Get rerank service details", tags=["Rerank Services"],
    response_model=RerankServiceDetailSchema,
)
async def get_rerank_service(
    app_id: int,
    service_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """Details of an app-owned rerank service (masked key); 0 returns an empty template."""
    detail = RerankServiceService.get_rerank_service_detail(db, app_id, service_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=RERANK_SERVICE_NOT_FOUND_MSG)
    return detail


@rerank_services_router.post(
    "/{service_id}", summary="Create or update rerank service", tags=["Rerank Services"],
    response_model=RerankServiceDetailSchema,
)
async def create_or_update_rerank_service(
    app_id: int,
    service_id: int,
    service_data: CreateUpdateRerankServiceSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("administrator"))],
):
    """Create (service_id 0) or update an app-owned rerank service."""
    try:
        service = await run_in_threadpool(
            RerankServiceService.create_or_update_rerank_service,
            db, app_id, service_id, service_data, int(auth_context.identity.id),
        )
    except RerankServiceError as exc:
        raise _http_error(exc)
    if service is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=RERANK_SERVICE_NOT_FOUND_MSG)
    return RerankServiceService.get_rerank_service_detail(db, app_id, service.service_id)


@rerank_services_router.delete(
    "/{service_id}", summary="Delete rerank service", tags=["Rerank Services"], status_code=status.HTTP_204_NO_CONTENT
)
async def delete_rerank_service(
    app_id: int,
    service_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("administrator"))],
):
    """Delete an app-owned rerank service."""
    try:
        deleted = await run_in_threadpool(RerankServiceService.delete_rerank_service, db, app_id, service_id)
    except RerankServiceError as exc:
        raise _http_error(exc)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=RERANK_SERVICE_NOT_FOUND_MSG)
