from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
from typing import List, Optional, Annotated
import json
import logging
from lks_idprovider import AuthContext
from sqlalchemy.orm import Session
import os
import tempfile

# Import services
from services.repository_service import RepositoryService
from services.resource_service import ResourceService
from services.media_service import MediaService
from services.repository_export_service import RepositoryExportService
from services.repository_import_service import RepositoryImportService
from services.silo_service import SiloService

from schemas.repository_schemas import RepositoryListItemSchema, RepositoryDetailSchema, CreateUpdateRepositorySchema, CreateRepositorySchema, UpdateRepositorySchema, RepositorySearchSchema
from schemas.media_schemas import MediaResponse, MediaUploadResponse
from schemas.import_schemas import ConflictMode, ImportResponseSchema
from schemas.export_schemas import RepositoryExportFileSchema
from routers.internal.auth_utils import get_current_user_oauth
from routers.controls import enforce_file_size_limit
from routers.controls.role_authorization import require_min_role, AppRole
from repositories.media_repository import MediaRepository
from utils.error_handlers import ValidationError
from schemas.silo_schemas import CostEstimationResponseSchema
from services.import_job_service import ImportJobService, ConflictError
from services.import_job_download import download_one_row
from services.import_job_confirm import confirm_rows, estimate_rows
from services.import_job_discard import discard_rows
from schemas.import_job_schemas import (
    CsvPreviewResponseSchema, ImportJobResponseSchema, ConfirmDiscardRowsSchema, ConfirmRowsResponseSchema,
)

# Import database dependency
from db.database import get_db

# Set up logging
logger = logging.getLogger(__name__)

repositories_router = APIRouter()

# Debug log when router is loaded
logger.info("Repositories router loaded successfully")


def _validate_repository_app_ownership(repository_id: int, app_id: int, db) -> None:
    """Raise HTTP 404 if the repository does not exist or does not belong to app_id."""
    from models.repository import Repository as RepositoryModel
    repo = RepositoryService.get_repository(repository_id, db)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found",
        )
    if repo.app_id != app_id:
        logger.warning(
            f"Access violation: repository {repository_id} (app {repo.app_id}) accessed from app {app_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Repository does not belong to this app",
        )


# ==================== REPOSITORY MANAGEMENT ====================

@repositories_router.post(
    "/import",
    summary="Import Repository",
    tags=["Repositories", "Export/Import"],
    response_model=ImportResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def import_repository(
    app_id: int,
    file: Annotated[UploadFile, File(...)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("administrator"))],
    conflict_mode: Annotated[ConflictMode, Query()] = ConflictMode.FAIL,
    new_name: Annotated[Optional[str], Query()] = None,
    selected_embedding_service_id: Annotated[Optional[int], Query()] = None,
):
    """Import Repository from JSON file.

    Note: Imports repository STRUCTURE only (no files).
    Upload documents after import.
    """
    try:
        content = await file.read()
        file_data = json.loads(content)
        export_data = RepositoryExportFileSchema(**file_data)

        import_service = RepositoryImportService(db)
        import_service.validate_import(export_data, app_id)

        summary = import_service.import_repository(
            export_data,
            app_id,
            conflict_mode,
            new_name,
            selected_embedding_service_id=(
                selected_embedding_service_id
            ),
        )

        return ImportResponseSchema(
            success=True,
            message=(
                f"Repository '{summary.component_name}' "
                f"imported successfully"
            ),
            summary=summary,
        )
    except HTTPException:
        raise
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status.HTTP_409_CONFLICT, str(e)
            )
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, str(e)
        )
    except Exception as e:
        logger.error(
            f"Import error: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Import failed",
        )


@repositories_router.get("/", 
                         summary="List repositories",
                         tags=["Repositories"],
                         response_model=List[RepositoryListItemSchema])
async def list_repositories(
    app_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """
    List all repositories for a specific app.
    """
    user_id = int(auth_context.identity.id)
    
    logger.info(f"List repositories called for app_id: {app_id}, user_id: {user_id}")
    
    # Use RepositoryService for business logic
    return RepositoryService.get_repositories_list(app_id, db)


@repositories_router.get("/{repository_id}",
                        summary="Get repository details",
                        tags=["Repositories"],
                        response_model=RepositoryDetailSchema)
async def get_repository(
    app_id: int,
    repository_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """
    Get detailed information about a specific repository including its resources.
    """
    
    # Use RepositoryService for business logic
    return RepositoryService.get_repository_detail(app_id, repository_id, db)


@repositories_router.post("/",
                         summary="Create repository",
                         tags=["Repositories"],
                         response_model=RepositoryDetailSchema,
                         status_code=status.HTTP_201_CREATED)
async def create_repository(
    app_id: int,
    repo_data: CreateRepositorySchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    """
    Create a new repository.
    """
    try:
        repo = RepositoryService.create_repository_router(app_id, repo_data, db)
        return RepositoryService.get_repository_detail(app_id, repo.repository_id, db)
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating repository: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@repositories_router.put("/{repository_id}",
                         summary="Update repository",
                         tags=["Repositories"],
                         response_model=RepositoryDetailSchema)
async def update_repository(
    app_id: int,
    repository_id: int,
    repo_data: UpdateRepositorySchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    """
    Update an existing repository. Note: vector_db_type cannot be changed after creation.
    """
    _validate_repository_app_ownership(repository_id, app_id, db)
    try:
        repo = RepositoryService.update_repository_router(app_id, repository_id, repo_data, db)
        return RepositoryService.get_repository_detail(app_id, repo.repository_id, db)
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating repository: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@repositories_router.delete("/{repository_id}",
                           summary="Delete repository",
                           tags=["Repositories"])
async def delete_repository(
    app_id: int,
    repository_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    """
    Delete a repository and all its resources.
    """
    
    # Use RepositoryService for business logic
    RepositoryService.delete_repository_router(repository_id, db)
    
    return {"message": "Repository deleted successfully"}


@repositories_router.post(
    "/{repository_id}/export",
    summary="Export Repository",
    tags=["Repositories", "Export/Import"],
    status_code=status.HTTP_200_OK,
)
async def export_repository(
    app_id: int,
    repository_id: int,
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
    include_dependencies: Annotated[bool, Query(description="Bundle silo and its dependencies")] = True,
):
    """Export Repository configuration to JSON file.

    Note: Exports repository STRUCTURE only (no files).
    Files must be re-uploaded after import.
    """
    try:
        export_service = RepositoryExportService(db)
        export_data = export_service.export_repository(
            repository_id,
            app_id,
            getattr(auth_context, "user_id", None),
            include_dependencies,
        )

        filename = (
            f"{export_data.repository.name.replace(' ', '_')}"
            f"_repository.json"
        )

        return JSONResponse(
            content=export_data.model_dump(mode="json"),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{filename}"'
                )
            },
        )
    except ValueError as e:
        logger.warning(f"Export failed: {str(e)}")
        if "not found" in str(e):
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, str(e)
            )
        else:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, str(e)
            )
    except Exception as e:
        logger.error(
            f"Export error: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Export failed",
        )


# ==================== RESOURCE MANAGEMENT ====================

@repositories_router.post("/{repository_id}/resources",
                         summary="Upload resources",
                         tags=["Resources"])
async def upload_resources(
    app_id: int,
    repository_id: int,
    files: Annotated[List[UploadFile], File(...)],
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
    folder_id: Annotated[Optional[int], Form()] = None,
):
    """
    Upload multiple resources to a repository.
    Optionally specify a folder_id to upload files to a specific folder.
    """
    user_id = int(auth_context.identity.id)
    
    logger.info(f"Upload resources endpoint called - app_id: {app_id}, repository_id: {repository_id}, files_count: {len(files)}, folder_id: {folder_id} (type: {type(folder_id)}), user_id: {user_id}")
    
    # Use ResourceService to handle the business logic
    result = ResourceService.upload_resources_to_repository(
        app_id=app_id,
        repository_id=repository_id,
        files=files,
        db=db,
        folder_id=folder_id
    )
    
    return result


@repositories_router.post("/{repository_id}/resources/estimate",
                         summary="Estimate indexing cost for uploaded files",
                         tags=["Resources"],
                         response_model=CostEstimationResponseSchema)
async def estimate_upload_resources(
    app_id: int,
    repository_id: int,
    files: Annotated[List[UploadFile], File(...)],
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
    folder_id: Annotated[Optional[int], Form()] = None,
):
    """
    Server-side dry-run: receive multipart files, extract text server-side,
    and return a LightRAG indexing cost estimation without persisting files.
    """
    user_id = int(auth_context.identity.id)
    logger.info(f"Estimate resources endpoint called - app_id: {app_id}, repository_id: {repository_id}, files_count: {len(files)}, folder_id: {folder_id}, user_id: {user_id}")

    # Validate repository ownership
    _validate_repository_app_ownership(repository_id, app_id, db)
    repo = RepositoryService.get_repository(repository_id, db)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    silo_id = getattr(repo, 'silo_id', None)
    if not silo_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Repository has no silo configured")

    tmp_paths = []
    try:
        extracted_documents = []
        for upload in files:
            filename = upload.filename
            file_ext = os.path.splitext(filename)[1].lower()
            content = await upload.read()
            # write to temp file for existing loader utilities
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tf:
                tf.write(content)
                temp_path = tf.name
            tmp_paths.append(temp_path)

            base_metadata = {
                "repository_id": repository_id,
                "name": filename,
                "file_type": file_ext,
                "folder_id": folder_id,
            }

            # Estimation is LightRAG-only; feed whole pages/files so the chunk
            # count reflects LightRAG's own token-based chunking (not the
            # generic pre-split path used by PGVector/Qdrant).
            docs = SiloService.extract_documents_from_file(temp_path, file_ext, base_metadata, split=False)
            for doc in docs:
                extracted_documents.append({
                    "content": getattr(doc, "page_content", ""),
                    "metadata": getattr(doc, "metadata", {}),
                })

        # Call existing estimation logic
        result = SiloService.estimate_indexing_cost(silo_id, extracted_documents, db)
        return CostEstimationResponseSchema(**result)

    except ValueError as e:
        logger.error(f"Error estimating indexing cost: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except LookupError as e:
        logger.error(f"Silo lookup error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in estimate resources: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass


# ==================== CSV IMPORT ====================

@repositories_router.post("/{repository_id}/csv-imports/preview",
                         summary="Preview CSV headers for import",
                         tags=["CSV Import"],
                         response_model=CsvPreviewResponseSchema)
async def preview_csv_import(
    app_id: int,
    repository_id: int,
    file: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    headers = ImportJobService.preview_headers(file.file)
    return CsvPreviewResponseSchema(headers=headers)


@repositories_router.post("/{repository_id}/csv-imports",
                         summary="Start a CSV → PDF import",
                         tags=["CSV Import"],
                         response_model=ImportJobResponseSchema,
                         status_code=status.HTTP_202_ACCEPTED)
async def create_csv_import(
    app_id: int,
    repository_id: int,
    file: Annotated[UploadFile, File(...)],
    link_column: Annotated[str, Form(...)],
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    try:
        job = ImportJobService.create_job(repository_id, file.file, link_column, db, source_filename=file.filename)
    except ConflictError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={"message": "An import is already in progress for this repository", "job_id": e.job_id},
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return _import_job_response(job.id, db)


@repositories_router.get("/{repository_id}/csv-imports/active",
                        summary="Get the active CSV import job, if any",
                        tags=["CSV Import"],
                        response_model=Optional[ImportJobResponseSchema])
async def get_active_csv_import(
    app_id: int,
    repository_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    job = ImportJobService.get_active_job(repository_id, db)
    if not job:
        return None
    return _import_job_response(job.id, db)


@repositories_router.get("/{repository_id}/csv-imports/{import_job_id}",
                        summary="Get a CSV import job with its rows",
                        tags=["CSV Import"],
                        response_model=ImportJobResponseSchema)
async def get_csv_import(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    result = _import_job_response(import_job_id, db)
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Import job not found")
    return result


@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/rows/{row_id}/retry",
                         summary="Retry a failed CSV import row",
                         tags=["CSV Import"],
                         response_model=ImportJobResponseSchema)
async def retry_csv_import_row(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    row_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    await download_one_row(row_id)
    result = _import_job_response(import_job_id, db)
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Import job not found")
    return result


@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/estimate",
                         summary="Estimate LightRAG indexing cost for selected CSV import rows",
                         tags=["CSV Import"],
                         response_model=CostEstimationResponseSchema)
async def estimate_csv_import_rows(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    body: ConfirmDiscardRowsSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    try:
        result = estimate_rows(import_job_id, body.row_ids, db)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return CostEstimationResponseSchema(**result)


@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/confirm",
                         summary="Confirm selected CSV import rows into Resources",
                         tags=["CSV Import"],
                         response_model=ConfirmRowsResponseSchema)
async def confirm_csv_import_rows(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    body: ConfirmDiscardRowsSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    result = confirm_rows(import_job_id, body.row_ids, db)
    return ConfirmRowsResponseSchema(**result)


@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/discard",
                         summary="Discard selected CSV import rows",
                         tags=["CSV Import"])
async def discard_csv_import_rows(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    body: ConfirmDiscardRowsSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    _validate_repository_app_ownership(repository_id, app_id, db)
    discard_rows(import_job_id, body.row_ids, db)
    return {"success": True}


def _import_job_response(import_job_id: int, db: Session) -> Optional[ImportJobResponseSchema]:
    data = ImportJobService.get_job_with_counts(import_job_id, db)
    if not data:
        return None
    job = data['job']
    return ImportJobResponseSchema(
        id=job.id, repository_id=job.repository_id, status=job.status.value,
        source_filename=job.source_filename, link_column=job.link_column,
        created_at=job.created_at, last_activity_at=job.last_activity_at,
        rows=data['rows'], counts=data['counts'],
    )


@repositories_router.post("/{repository_id}/resources/{resource_id}/move",
                         summary="Move resource to different folder",
                         tags=["Resources"])
async def move_resource(
    app_id: int,
    repository_id: int,
    resource_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
    new_folder_id: Annotated[Optional[int], Form()] = None,
):
    """
    Move a resource to a different folder within the same repository.
    """
    user_id = int(auth_context.identity.id)
    
    logger.info(f"Move resource endpoint called - app_id: {app_id}, repository_id: {repository_id}, resource_id: {resource_id}, new_folder_id: {new_folder_id}, user_id: {user_id}")
    
    # Use ResourceService to handle the business logic
    result = ResourceService.move_resource_to_folder(
        resource_id=resource_id,
        repository_id=repository_id,
        new_folder_id=new_folder_id,
        db=db
    )
    
    return result


@repositories_router.delete("/{repository_id}/resources/{resource_id}",
                           summary="Delete resource",
                           tags=["Resources"])
async def delete_resource(
    app_id: int,
    repository_id: int,
    resource_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    """
    Delete a specific resource from a repository.
    """
    user_id = int(auth_context.identity.id)
    
    logger.info(f"Delete resource endpoint called - app_id: {app_id}, repository_id: {repository_id}, resource_id: {resource_id}, user_id: {user_id}")
    
    # Use ResourceService to handle the business logic
    result = ResourceService.delete_resource_from_repository(
        app_id=app_id,
        repository_id=repository_id,
        resource_id=resource_id,
        db=db
    )
    
    return result


@repositories_router.delete("/{repository_id}/resources",
                           summary="Delete all resources in a repository",
                           tags=["Resources"])
async def delete_all_resources(
    app_id: int,
    repository_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    """
    Delete every resource (file) in a repository. The repository itself is kept.
    """
    user_id = int(auth_context.identity.id)

    logger.info(f"Delete all resources endpoint called - app_id: {app_id}, repository_id: {repository_id}, user_id: {user_id}")

    _validate_repository_app_ownership(repository_id, app_id, db)

    result = ResourceService.delete_all_resources_from_repository(
        repository_id=repository_id,
        db=db
    )

    return result


@repositories_router.get("/{repository_id}/resources/{resource_id}/download",
                        summary="Download resource",
                        tags=["Resources"])
async def download_resource(
    app_id: int,
    repository_id: int,
    resource_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """
    Download a specific resource from a repository.
    """
    user_id = int(auth_context.identity.id)
    
    logger.info(f"Download resource endpoint called - app_id: {app_id}, repository_id: {repository_id}, resource_id: {resource_id}, user_id: {user_id}")
    
    # Use ResourceService to handle the business logic
    file_path, filename = ResourceService.download_resource_from_repository(
        app_id=app_id,
        repository_id=repository_id,
        resource_id=resource_id,
        user_id=user_id,
        db=db
    )
    
    # Return file for download
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


# ==================== MEDIA MANAGEMENT ====================
@repositories_router.post("/{repository_id}/media", response_model=MediaUploadResponse, responses={500: {"description": "Internal server error"}})
async def upload_media(
    app_id: int,
    repository_id: int,
    background_tasks: BackgroundTasks,
    files: Annotated[List[UploadFile], File(...)],
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
    folder_id: Annotated[Optional[int], Form()] = None,
    forced_language: Annotated[Optional[str], Form()] = None,
    chunk_min_duration: Annotated[Optional[int], Form()] = None,
    chunk_max_duration: Annotated[Optional[int], Form()] = None,
    chunk_overlap: Annotated[Optional[int], Form()] = None,
):
    """
    Upload video/audio files for transcription and indexing.
    AI services (transcription, video analysis) are configured at repository level.

    Supported formats:
    - Video: mp4, mov, avi, mkv, webm, flv, wmv, mpeg, mpg
    - Audio: mp3, wav, m4a, aac, ogg, flac, wma

    Configuration:
    - forced_language: Force transcription language (e.g., 'es', 'en', 'fr'). Leave empty for auto-detect.
    - chunk_min_duration: Minimum chunk duration in seconds (default: 30)
    - chunk_max_duration: Maximum chunk duration in seconds (default: 120)
    - chunk_overlap: Overlap between chunks in seconds (default: 0, recommended: 5-10)
    """
    user_id = auth_context.identity.id
    logger.info(f"Upload media - app_id: {app_id}, repository_id: {repository_id}, user_id: {user_id}, files: {len(files)}")
    
    try:
        created_media, failed_files = await MediaService.upload_media_files(
            repository_id=repository_id,
            files=files,
            folder_id=folder_id,
            db=db,
            background_tasks=background_tasks,
            user_context=auth_context,
            forced_language=forced_language,
            chunk_min_duration=chunk_min_duration,
            chunk_max_duration=chunk_max_duration,
            chunk_overlap=chunk_overlap,
        )

        return MediaUploadResponse(
            message=f"Uploaded {len(created_media)} media file(s)",
            created_media=[MediaResponse(**m.__dict__) for m in created_media],
            failed_files=failed_files
        )
    except Exception as e:
        logger.error(f"Error uploading media: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@repositories_router.post("/{repository_id}/media/youtube", response_model=MediaResponse, responses={400: {"description": "Validation error"}, 500: {"description": "Internal server error"}})
async def add_youtube_video(
    app_id: int,
    background_tasks: BackgroundTasks,
    repository_id: int,
    url: Annotated[str, Form(...)],
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
    folder_id: Annotated[Optional[int], Form()] = None,
    forced_language: Annotated[Optional[str], Form()] = None,
    chunk_min_duration: Annotated[Optional[int], Form()] = None,
    chunk_max_duration: Annotated[Optional[int], Form()] = None,
    chunk_overlap: Annotated[Optional[int], Form()] = None,
):
    """
    Add YouTube video for transcription and indexing.
    AI services (transcription, video analysis) are configured at repository level.

    The video will be:
    1. Downloaded from YouTube
    2. Audio extracted and normalized
    3. Transcribed using Whisper
    4. Chunked into segments
    5. Indexed for RAG queries
    6. (If multimodal) Video analyzed with Video-LLM and visual descriptions merged

    Configuration:
    - forced_language: Force transcription language (e.g., 'es', 'en', 'fr'). Leave empty for auto-detect.
    - chunk_min_duration: Minimum chunk duration in seconds (default: 30)
    - chunk_max_duration: Maximum chunk duration in seconds (default: 120)
    - chunk_overlap: Overlap between chunks in seconds (default: 0, recommended: 5-10)
    """
    user_id = auth_context.identity.id
    logger.info(f"Add YouTube video - app_id: {app_id}, repository_id: {repository_id}, user_id: {user_id}, url: {url}")

    try:
        media = await MediaService.create_media_from_youtube(
            url=url,
            repository_id=repository_id,
            folder_id=folder_id,
            db=db,
            background_tasks=background_tasks,
            forced_language=forced_language,
            chunk_min_duration=chunk_min_duration,
            chunk_max_duration=chunk_max_duration,
            chunk_overlap=chunk_overlap,
        )

        return MediaResponse(**media.__dict__)
    except ValueError as e:
        # Handle validation errors (invalid URL, duplicate)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding YouTube video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@repositories_router.get("/{repository_id}/media", response_model=List[MediaResponse])
async def list_media(
    app_id: int,
    repository_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
    folder_id: Annotated[Optional[int], Query()] = None,
):
    """List all media in repository"""
    media_list = MediaService.list_media(
        repository_id=repository_id,
        folder_id=folder_id,
        db=db,
    )
    return [MediaResponse(**{k: v for k, v in m.__dict__.items() if not k.startswith('_')}) for m in media_list]

@repositories_router.post("/{repository_id}/media/{media_id}/move",
                         summary="Move media to different folder",
                         tags=["Media"])
async def move_media(
    app_id: int,
    repository_id: int,
    media_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
    new_folder_id: Annotated[Optional[int], Form()] = None,
):
    """
    Move a resource to a different folder within the same repository.
    """
    user_id = int(auth_context.identity.id)

    logger.info(f"Move media endpoint called - app_id: {app_id}, repository_id: {repository_id}, media_id: {media_id}, new_folder_id: {new_folder_id}, user_id: {user_id}")

    # Use MediaService to handle the business logic
    result = MediaService.move_media_to_folder(
        app_id=app_id,
        media_id=media_id,
        repository_id=repository_id,
        new_folder_id=new_folder_id,
        db=db
    )
    
    return result

@repositories_router.get("/{repository_id}/media/{media_id}", response_model=MediaResponse, responses={404: {"description": "Media not found"}})
async def get_media_status(
    app_id: int,
    repository_id: int,
    media_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    media = MediaRepository.get_by_id(media_id, db)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")
    return media

@repositories_router.delete("/{repository_id}/media/{media_id}")
async def delete_media(
    app_id: int,
    repository_id: int,
    media_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    """
    Delete a media file and all derived data (chunks, transcripts, embeddings).
    """
    user_id = int(auth_context.identity.id)

    logger.info(f"Delete media endpoint called - app_id={app_id}, repository_id={repository_id}, media_id={media_id}, user_id={user_id}")

    result = MediaService.delete_media(
        app_id=app_id,
        repository_id=repository_id,
        media_id=media_id,
        db=db
    )

    return result

# ==================== REPOSITORY SEARCH ====================

@repositories_router.post("/{repository_id}/search",
                         summary="Search documents in repository",
                         tags=["Repositories", "Search"])
async def search_repository_documents(
    app_id: int,
    repository_id: int,
    search_query: RepositorySearchSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """
    Search for documents in a repository using semantic search with optional metadata filtering.
    This leverages the repository's associated silo for searching.
    """
    user_id = int(auth_context.identity.id)
    
    logger.info(f"Repository search request - app_id: {app_id}, repository_id: {repository_id}, user_id: {user_id}")
    logger.info(f"Search query: {search_query.query}, limit: {search_query.limit}, filter_metadata: {search_query.filter_metadata}")
    
    try:
        # Use RepositoryService to handle the search
        result = RepositoryService.search_repository_documents_router(
            repository_id=repository_id,
            query=search_query.query,
            filter_metadata=search_query.filter_metadata,
            limit=search_query.limit or 10,
            db=db
        )
        
        logger.info(f"Repository search completed - found {len(result.get('results', []))} results")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching repository {repository_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching repository: {str(e)}"
        )


@repositories_router.post(
    "/{repository_id}/resume-indexing",
    summary="Resume an interrupted ingestion",
    tags=["Resources"],
)
async def resume_indexing(
    app_id: int,
    repository_id: int,
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    """Re-index every resource of this repository that is not ``ready``.

    For picking up after a backend restart or an LLM outage. Pages already
    extracted are skipped by LightRAG, so this costs only the remaining work.

    Response:
    - ``session_id``: id of the started run, or null when nothing was pending
    - ``resumed``: how many resources were queued
    """
    _validate_repository_app_ownership(repository_id, app_id, db)

    session_id, resumed = ResourceService.resume_indexing(db, repository_id)
    return {"session_id": session_id, "resumed": resumed}


@repositories_router.post(
    "/{repository_id}/stop-indexing",
    summary="Pause or cancel the running ingestion",
    tags=["Resources"],
)
async def stop_indexing(
    app_id: int,
    repository_id: int,
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
    mode: str = "pause",
):
    """Ask the running ingestion of this repository to stop.

    ``mode=pause`` (default) or ``mode=cancel``. **Neither deletes anything**:
    pages already indexed stay indexed either way, and both leave the remaining
    resources re-indexable via ``resume-indexing``. The difference is intent —
    ``pause`` keeps them counted as resumable so the UI offers to continue,
    ``cancel`` does not.

    Not immediate: the resource being indexed is left to finish, and only then
    does the run end.

    Response:
    - ``mode``: the mode applied
    - ``stopped``: how many resources this run will no longer index
    - ``was_running``: whether a run was actually alive when asked
    """
    _validate_repository_app_ownership(repository_id, app_id, db)

    if mode not in ("pause", "cancel"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="mode must be 'pause' or 'cancel'",
        )
    return ResourceService.request_ingestion_stop(db, repository_id, mode=mode)


@repositories_router.get(
    "/{repository_id}/ingestion-status",
    summary="Check if repository silo has an active ingestion",
    tags=["Resources"],
)
async def get_ingestion_status(
    app_id: int,
    repository_id: int,
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """Return whether this repository is currently indexing.

    Response:
    - ``is_indexing``: true only while a run is **alive** (the silo's advisory
      lock is held) — not merely while rows are left unfinished
    - ``active_session_id``: id of the running batch (or null)
    - ``resumable``: resources an interrupted run left behind, 0 while alive
    """
    _validate_repository_app_ownership(repository_id, app_id, db)

    liveness = ResourceService.get_ingestion_liveness(db, repository_id)
    progress = (
        ResourceService.get_indexing_progress(db, repository_id)
        if liveness["is_indexing"] else None
    )
    return {
        "is_indexing": liveness["is_indexing"],
        "active_session_id": progress["session_id"] if progress else None,
        "resumable": liveness["resumable"],
    }


@repositories_router.get(
    "/{repository_id}/ingestion-progress/{session_id}",
    summary="Stream ingestion progress via SSE",
    tags=["Resources"],
)
async def stream_ingestion_progress(
    app_id: int,
    repository_id: int,
    session_id: str,
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    db: Annotated[Session, Depends(get_db)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    """Stream file ingestion progress via Server-Sent Events.

    Connect immediately after uploading files using the ``session_id``
    returned in the upload response.

    Progress is read from the ``Resource`` rows, so any uvicorn worker can serve
    this stream.  ``session_id`` is therefore only an identifier for the client's
    stream — the payload always describes the repository's active batch.

    Events:
    - ``message``: JSON progress payload (emitted on every change)
    - ``complete``: no resource of this repository is pending/indexing any more
    - ``error``: something went wrong (data contains error message)
    """
    async def event_generator():
        max_wait = 3600
        elapsed = 0.0

        def _read():
            # End the read transaction each tick: this generator lives for as
            # long as the indexing run, and an open transaction would sit as
            # "idle in transaction", blocking DDL on Resource.
            db.rollback()
            # Gate on liveness, not on leftover rows: a killed run leaves them
            # 'pending' forever and the stream would never end.
            if not ResourceService.get_ingestion_liveness(db, repository_id)["is_indexing"]:
                return None
            return ResourceService.get_indexing_progress(db, repository_id)

        try:
            # Emit initial state immediately on connect — no 0.5 s wait.
            # This flushes the response buffer through any intermediate proxy,
            # causing isConnected to flip to true AND the progress bar to render
            # actual data right away instead of staying at "Connecting…".
            initial = _read()
            if initial is None:
                yield "event: complete\ndata: {}\n\n"
                return
            yield f"data: {json.dumps(initial)}\n\n"

            while elapsed < max_wait:
                await asyncio.sleep(0.5)
                elapsed += 0.5

                progress = _read()
                if progress is None:
                    yield "event: complete\ndata: {}\n\n"
                    break

                # Emit every tick, not only on chunk changes: the elapsed clock
                # and ETA move even while a page is still being extracted.
                yield f"data: {json.dumps(progress)}\n\n"

        except Exception as e:
            logger.error("SSE error for session %s: %s", session_id, e)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )