from models.resource import Resource
from repositories.resource_repository import ResourceRepository
from services.folder_service import FolderService
from typing import List, Tuple, Optional
import os
from services.silo_service import SiloService
from utils.logger import get_logger
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, UploadFile

REPO_BASE_FOLDER = os.path.abspath(os.getenv('REPO_BASE_FOLDER'))
logger = get_logger(__name__)

class ResourceService:

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}


    @staticmethod
    def get_resources_by_repo_id(repository_id: int, db: Session) -> List[Resource]:
        """
        Get all resources by repository ID
        
        Args:
            repository_id: Repository ID
            db: Database session
            
        Returns:
            List of Resource instances
        """
        return ResourceRepository.get_by_repository_id(db, repository_id)
    
    @staticmethod
    def get_resource(resource_id: int, db: Session) -> Optional[Resource]:
        """
        Get a resource by its ID
        
        Args:
            resource_id: The ID of the resource
            db: Database session
            
        Returns:
            The Resource instance or None if not found
        """
        return ResourceRepository.get_by_id(db, resource_id)
    
    @staticmethod
    def move_resource_to_folder(resource_id: int, repository_id: int, new_folder_id: Optional[int], db: Session) -> dict:
        """
        Move a resource to a different folder within the same repository.
        Updates file system location, database record, and re-indexes in vector DB.
        
        Args:
            resource_id: ID of the resource to move
            repository_id: Repository ID (for validation)
            new_folder_id: New folder ID (None for root)
            db: Database session
            
        Returns:
            dict: Result with success status and updated resource info
        """
        try:
            # Get the resource
            resource = ResourceService.get_resource(resource_id, db)
            if not resource:
                raise ValueError(f"Resource {resource_id} not found")
            
            # Validate repository ownership
            if resource.repository_id != repository_id:
                raise ValueError(f"Resource {resource_id} does not belong to repository {repository_id}")
            
            # Validate new folder if provided
            if new_folder_id is not None:
                if not FolderService.validate_folder_access(new_folder_id, repository_id, db):
                    raise ValueError(f"Folder {new_folder_id} does not belong to repository {repository_id}")
            
            # Get old and new paths
            old_path = ResourceService.get_resource_file_path(resource_id, db)
            if not old_path:
                raise ValueError(f"Could not determine current path for resource {resource_id}")
            
            # Build new path
            repository_path = os.path.join(REPO_BASE_FOLDER, str(repository_id))
            if new_folder_id:
                folder_path = FolderService.get_folder_path(new_folder_id, db)
                new_path = os.path.join(repository_path, folder_path, resource.uri)
            else:
                new_path = os.path.join(repository_path, resource.uri)
            
            # Create target directory if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Move the file in the file system
            import shutil
            shutil.move(old_path, new_path)
            logger.info(f"Moved file from {old_path} to {new_path}")
            
            # Update database record
            resource.folder_id = new_folder_id
            db.add(resource)
            db.commit()  # Commit the database changes first
            logger.info(f"Updated resource {resource_id} folder_id to {new_folder_id}")
            
            # Update metadata in vector database without re-indexing content
            from services.silo_service import SiloService
            SiloService.update_resource_metadata(resource, db)
            logger.info(f"Updated metadata for resource {resource_id} with new folder information")
            
            return {
                "success": True,
                "message": "Resource moved successfully",
                "resource_id": resource_id,
                "new_folder_id": new_folder_id,
                "new_path": new_path
            }
            
        except Exception as e:
            logger.error(f"Error moving resource {resource_id}: {str(e)}")
            raise ValueError(f"Failed to move resource: {str(e)}")

    @staticmethod
    def delete_resource(resource_id: int, db: Session) -> bool:
        """
        Delete a resource completely (file, database record, and silo indexing)
        
        Args:
            resource_id: The ID of the resource to delete
            db: Database session
            
        Returns:
            True if deletion was successful, False if resource not found
        """
        resource = ResourceRepository.get_by_id(db, resource_id)
        if not resource:
            logger.warning(f"Resource {resource_id} not found for deletion")
            return False
        
        try:
            # Delete from silo first
            SiloService.delete_resource(resource)
            logger.info(f"Resource {resource_id} deleted from silo")
            
            # Delete file from disk
            file_path = os.path.join(REPO_BASE_FOLDER, str(resource.repository_id), resource.uri)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File {file_path} deleted from disk")
            
            # Delete from database
            ResourceRepository.delete(db, resource)
            ResourceRepository.commit(db)
            logger.info(f"Resource {resource_id} deleted from database")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting resource {resource_id}: {str(e)}")
            ResourceRepository.rollback(db)
            return False
    
    @staticmethod
    def get_resource_file_path(resource_id: int, db: Session) -> Optional[str]:
        """
        Get the file path for a resource
        
        Args:
            resource_id: The ID of the resource
            db: Database session
            
        Returns:
            The full file path or None if resource not found
        """
        resource = ResourceService.get_resource(resource_id, db)
        if not resource:
            return None
        
        # Build path including folder structure
        if resource.folder_id:
            folder_path = FolderService.get_folder_path(resource.folder_id, db)
            return os.path.join(REPO_BASE_FOLDER, str(resource.repository_id), folder_path, resource.uri)
        else:
            # Resource is at root level
            return os.path.join(REPO_BASE_FOLDER, str(resource.repository_id), resource.uri)
    
    @staticmethod
    def create_multiple_resources(files: List, repository_id: int, db: Session, custom_names: dict = None, folder_id: Optional[int] = None, extra_metadata: dict = None) -> Tuple[List[Resource], List[dict]]:
        """
        Create multiple resources from uploaded files
        
        Args:
            files: List of uploaded files
            repository_id: The ID of the repository
            db: Database session
            custom_names: Dictionary mapping file indices to custom names (without extensions)
            folder_id: Optional folder ID to upload files to
            
        Returns:
            Tuple containing a list of created Resource instances and a list of failed files
            
        Raises:
            ValueError: If any file is invalid or missing
        """
        if not files:
            raise ValueError("No files provided")
        
        if custom_names is None:
            custom_names = {}
        if extra_metadata is None:
            extra_metadata = {}

        # Validate folder_id if provided
        if folder_id is not None:
            logger.info(f"Validating folder access: folder_id={folder_id}, repository_id={repository_id}")
            if not FolderService.validate_folder_access(folder_id, repository_id, db):
                logger.error(f"Folder {folder_id} does not belong to repository {repository_id}")
                raise ValueError(f"Folder {folder_id} does not belong to repository {repository_id}")
            logger.info(f"Folder access validated successfully")
        
        # Build the target path
        repository_path = os.path.join(REPO_BASE_FOLDER, str(repository_id))
        if folder_id:
            folder_path = FolderService.get_folder_path(folder_id, db)
            target_path = os.path.join(repository_path, folder_path)
            logger.info(f"Building path for folder: repository_path={repository_path}, folder_path={folder_path}, target_path={target_path}")
        else:
            target_path = repository_path
            logger.info(f"Building path for root: target_path={target_path}")
        
        os.makedirs(target_path, exist_ok=True)

        created_resources = []
        failed_files = []
        
        for index, file in enumerate(files):
            custom_name = custom_names.get(index)
            result = ResourceService._process_single_file(
                file, repository_id, target_path, custom_name, folder_id, db,
                extra_metadata=extra_metadata.get(index),
            )
            if isinstance(result, Resource):
                created_resources.append(result)
                logger.info(f"Resource {result.name} prepared for indexing")
            else:
                failed_files.append(result)

        session_id = None
        if created_resources:
            try:
                ResourceRepository.commit(db)
                logger.info(f"Successfully saved {len(created_resources)} resources to database")
                
                # Fetch silo_id from repository
                repo = ResourceRepository.get_repository_by_id(db, repository_id)
                silo_id = repo.silo_id if repo else 0
                
                session_id = ResourceService._index_resources_background(created_resources, silo_id)
            except Exception as e:
                logger.error(f"Error committing resources to database: {str(e)}")
                ResourceRepository.rollback(db)
                ResourceService._cleanup_files(created_resources, repository_path)
                raise

        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

        return created_resources, failed_files, session_id

    @staticmethod
    def _process_single_file(file, repository_id: int, target_path: str, custom_name: str = None, folder_id: Optional[int] = None, db: Session = None, extra_metadata: dict = None):
        """
        Process a single file upload
        
        Args:
            file: The uploaded file
            repository_id: The ID of the repository
            target_path: The path to save the file
            custom_name: Custom name for the resource (without extension)
            folder_id: Optional folder ID
            db: Database session to use
            
        Returns:
            Resource instance if successful, error dict if failed
        """
        if not file or not hasattr(file, 'filename') or file.filename == '':
            return {'filename': 'Unknown', 'error': 'Empty filename'}
            
            
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ResourceService.SUPPORTED_EXTENSIONS:
            supported = ', '.join(ResourceService.SUPPORTED_EXTENSIONS)
            return {
                'filename': file.filename,
                'error': f"Unsupported file type: {file_extension}. Supported: {supported}"
            }
        
        # Use custom name if provided, otherwise use original filename without extension
        if custom_name and custom_name.strip():
            name = custom_name.strip()
            # Ensure the saved filename includes the custom name with original extension
            save_filename = f"{name}{file_extension}"
        else:
            name = os.path.splitext(file.filename)[0]
            save_filename = file.filename
        
        try:
            file_path = os.path.join(target_path, save_filename)

            # Read file content first (needed both for saving and for reuse check)
            if hasattr(file, 'save'):
                file.save(file_path)
                content = None  # already saved
            else:
                if hasattr(file, 'file'):
                    file.file.seek(0)
                    content = file.file.read()
                else:
                    content = file.read()

            # Re-upload edge case: if a resource with the same URI already exists
            # in an error state, reset it instead of creating a duplicate record.
            # This prevents orphaned DB rows and LightRAG conflicts from stale data.
            existing = (
                db.query(Resource)
                .filter(
                    Resource.uri == save_filename,
                    Resource.repository_id == repository_id,
                    Resource.folder_id == folder_id,
                    Resource.status == 'error',
                )
                .first()
            )
            if existing:
                existing.status = 'pending'
                existing.name = name
                existing.type = file_extension
                db.commit()
                resource = existing
                logger.info(
                    "Re-upload: reset existing error resource %s → pending (resource_id=%s)",
                    save_filename, existing.resource_id,
                )
            else:
                resource = Resource(
                    name=name,
                    uri=save_filename,
                    repository_id=repository_id,
                    folder_id=folder_id,
                    type=file_extension,
                    status='pending',    # Will be updated to 'indexing', then 'ready' or 'error' by background thread
                    extra_metadata=extra_metadata,
                )
                ResourceRepository.create(db, resource)

            # Write file to disk (overwrite if re-upload)
            if content is not None:
                with open(file_path, 'wb') as f:
                    f.write(content)

            return resource

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            return {'filename': file.filename, 'error': str(e)}

    @staticmethod
    def _index_resources_background(resources: List[Resource], silo_id: int = 0) -> str:
        """Launch indexing in a background thread and return a session_id.

        Progress is tracked by real chunk count, not file count.
        The client connects to the SSE endpoint with this session_id to
        monitor the ingestion progress bar in real time.
        """
        import uuid
        import threading
        from services.ingestion_progress_tracker import IngestionProgressManager

        session_id = str(uuid.uuid4())

        # Snapshot resource IDs and names so we don't hold ORM refs across threads
        resource_snapshots = [
            (r.resource_id, getattr(r, 'uri', str(r.resource_id)))
            for r in resources
        ]

        def _run():
            import asyncio
            from models.resource import Resource as ResourceModel
            from db.database import SessionLocal as _SessionLocal

            # Background threads have no event loop; LightRAG/Neo4j async code
            # requires one.  Create a dedicated loop for this thread.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create the session immediately (before Phase 1) so the SSE client
            # can connect right away instead of waiting for chunk counting to finish.
            IngestionProgressManager.create_session(
                session_id=session_id,
                silo_id=silo_id,
                total_chunks=len(resource_snapshots),  # placeholder; updated after Phase 1
            )

            # Phase 1: extract docs per resource to get the real chunk count.
            # Extraction is CPU-only (no LLM calls), so this adds only seconds.
            chunk_counts = {}
            for resource_id, _ in resource_snapshots:
                _db = _SessionLocal()
                resource_obj = _db.query(ResourceModel).filter_by(resource_id=resource_id).first()
                _db.close()
                if resource_obj:
                    chunk_counts[resource_id] = SiloService.count_resource_chunks(resource_obj)

            total_chunks = sum(chunk_counts.values()) or len(resource_snapshots)
            # Update with real chunk count so progress math is correct for Phase 2.
            IngestionProgressManager.update_total_chunks(session_id, total_chunks)

            # Phase 2: index each resource, reporting progress by chunk count.
            cumulative = 0
            failed = 0
            try:
                for resource_id, resource_name in resource_snapshots:
                    resource_chunks = chunk_counts.get(resource_id, 1)
                    IngestionProgressManager.update_progress(
                        session_id=session_id,
                        processed=cumulative,
                        failed=failed,
                        chunk_name=resource_name,
                    )
                    try:
                        _db = _SessionLocal()
                        resource_obj = _db.query(ResourceModel).filter_by(resource_id=resource_id).first()
                        _db.close()
                        if resource_obj:
                            # Transition resource to 'indexing' before processing starts
                            _db_pre = _SessionLocal()
                            try:
                                r_pre = _db_pre.query(ResourceModel).filter_by(resource_id=resource_id).first()
                                if r_pre:
                                    r_pre.status = 'indexing'
                                    _db_pre.commit()
                            finally:
                                _db_pre.close()

                            offset = cumulative
                            def _on_progress(n, _offset=offset, _failed=failed):
                                IngestionProgressManager.update_progress(
                                    session_id=session_id,
                                    processed=_offset + n,
                                    failed=_failed,
                                )
                            SiloService.index_resource(resource_obj, progress_callback=_on_progress)
                        # Mark resource as ready in DB
                        _db2 = _SessionLocal()
                        try:
                            r = _db2.query(ResourceModel).filter_by(resource_id=resource_id).first()
                            if r:
                                r.status = 'ready'
                                _db2.commit()
                        finally:
                            _db2.close()
                        cumulative += resource_chunks
                    except Exception as e:
                        logger.error(f"Failed to index resource {resource_id}: {e}")
                        _db3 = _SessionLocal()
                        try:
                            r = _db3.query(ResourceModel).filter_by(resource_id=resource_id).first()
                            if r:
                                r.status = 'error'
                                _db3.commit()
                        finally:
                            _db3.close()
                        failed += resource_chunks
                        cumulative += resource_chunks

                    IngestionProgressManager.update_progress(
                        session_id=session_id,
                        processed=cumulative,
                        failed=failed,
                        chunk_name=resource_name,
                    )
            finally:
                loop.close()

            IngestionProgressManager.complete_session(session_id)
            logger.info(
                "Background indexing done: %s/%s chunks, %s failed",
                cumulative - failed, total_chunks, failed,
            )

        thread = threading.Thread(target=_run, daemon=True, name=f"index-{session_id[:8]}")
        thread.start()
        return session_id

    @staticmethod
    def _cleanup_files(resources: List[Resource], repository_path: str):
        for resource in resources:
            file_path = os.path.join(repository_path, resource.uri)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    # ==================== ROUTER SERVICE METHODS ====================
    
    @staticmethod
    def upload_resources_to_repository(
        app_id: int,
        repository_id: int,
        files: List[UploadFile],
        db: Session,
        folder_id: Optional[int] = None
    ) -> dict:
        """
        Upload multiple resources to a repository - business logic from router
        
        Args:
            app_id: Application ID
            repository_id: Repository ID
            files: List of uploaded files
            db: Database session
            folder_id: Optional folder ID to upload files to
            
        Returns:
            Dictionary with upload results
            
        Raises:
            HTTPException: If validation fails or repository not found
        """
        logger.info(f"Upload resources service called - app_id: {app_id}, repository_id: {repository_id}, files_count: {len(files)}")
        
        if not files:
            logger.warning("No files provided in upload request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )


        # Validate repository exists
        repo = ResourceRepository.get_repository_by_id(db, repository_id)
        if not repo:
            logger.error(f"Repository {repository_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Repository not found"
            )

        # Guard: reject concurrent ingestion on the same silo — LightRAG asyncio
        # locks are bound to a single event loop and cannot be shared across the
        # independent background threads each upload batch spawns.
        silo_id = repo.silo_id if repo.silo_id else 0
        if silo_id:
            from services.ingestion_progress_tracker import IngestionProgressManager
            if IngestionProgressManager.has_active_session_for_silo(silo_id):
                logger.warning(
                    f"Rejected concurrent upload for silo {silo_id} "
                    f"(repository {repository_id}): ingestion already in progress"
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="This silo already has an active ingestion in progress. "
                           "Please wait for it to complete before uploading more files.",
                )

        logger.info(f"Repository {repository_id} found, processing {len(files)} files")

        # Process files using create_multiple_resources method
        created_resources, failed_files, session_id = ResourceService.create_multiple_resources(
            files, repository_id, db, folder_id=folder_id
        )

        logger.info(f"Upload completed - {len(created_resources)} resources created, {len(failed_files)} failed")

        return {
            "message": f"Successfully uploaded {len(created_resources)} files to repository {repository_id}",
            "session_id": session_id,  # client uses this to monitor ingestion progress via SSE
            "created_resources": [
                {
                    "resource_id": r.resource_id,
                    "uri": r.uri,
                    "repository_id": r.repository_id,
                    "create_date": r.create_date,
                    "size": None,
                    "content_type": r.type or "unknown"
                } for r in created_resources
            ],
            "failed_files": failed_files
        }
    
    @staticmethod
    def delete_resource_from_repository(
        app_id: int,
        repository_id: int,
        resource_id: int,
        db: Session
    ) -> dict:
        """
        Delete a specific resource from a repository - business logic from router
        
        Args:
            app_id: Application ID
            repository_id: Repository ID
            resource_id: Resource ID to delete
            db: Database session
            
        Returns:
            Dictionary with deletion result
            
        Raises:
            HTTPException: If resource not found or could not be deleted
        """
        logger.info(f"Delete resource service called - app_id: {app_id}, repository_id: {repository_id}, resource_id: {resource_id}")

        # LightRAG has no real per-file delete (it would need to unwind the doc's
        # entities/relations from the knowledge graph). Block it up-front so the
        # file isn't removed from DB/disk while its graph data lingers — deleting
        # would be a lie. Whole-silo delete is the supported way to purge.
        resource = ResourceRepository.get_by_id(db, resource_id)
        silo = getattr(getattr(resource, "repository", None), "silo", None)
        if silo is not None and (silo.vector_db_type or "").upper() == "LIGHTRAG":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="LightRAG silos do not support deleting individual files; "
                       "delete the whole silo to remove its data.",
            )

        # Use delete_resource method with database session
        success = ResourceService.delete_resource(resource_id, db)
        if not success:
            logger.error(f"Resource {resource_id} not found or could not be deleted")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found or could not be deleted"
            )
        
        logger.info(f"Resource {resource_id} deleted successfully")
        return {"message": "Resource deleted successfully"}

    @staticmethod
    def delete_all_resources_from_repository(repository_id: int, db: Session) -> dict:
        """
        Delete every resource in a repository - business logic from router

        Raises:
            HTTPException: If the repository's silo is LightRAG (no per-file delete support)
        """
        repo = ResourceRepository.get_repository_by_id(db, repository_id)
        silo = getattr(repo, "silo", None)
        if silo is not None and (silo.vector_db_type or "").upper() == "LIGHTRAG":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="LightRAG silos do not support deleting individual files; "
                       "delete the whole silo to remove its data.",
            )

        resources = ResourceRepository.get_by_repository_id(db, repository_id)
        deleted_count = 0
        for resource in resources:
            if ResourceService.delete_resource(resource.resource_id, db):
                deleted_count += 1
            else:
                logger.error(f"Resource {resource.resource_id} could not be deleted in bulk delete")

        logger.info(f"Deleted {deleted_count}/{len(resources)} resources from repository {repository_id}")
        return {"deleted_count": deleted_count, "failed_count": len(resources) - deleted_count}

    @staticmethod
    def download_resource_from_repository(
        app_id: int,
        repository_id: int,
        resource_id: int,
        user_id: str,
        db: Session
    ) -> tuple:
        """
        Download a specific resource from a repository - business logic from router
        
        Args:
            app_id: Application ID
            repository_id: Repository ID
            resource_id: Resource ID to download
            user_id: User ID for logging
            db: Database session
            
        Returns:
            Tuple containing (file_path, filename) for FileResponse
            
        Raises:
            HTTPException: If resource not found or file doesn't exist
        """
        logger.info(f"Download request - app_id: {app_id}, repository_id: {repository_id}, resource_id: {resource_id}, user_id: {user_id}")
        
        # Get resource using method with database session
        resource = ResourceService.get_resource(resource_id, db)
        if not resource:
            logger.error(f"Resource {resource_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found"
            )
        
        logger.info(f"Resource found: {resource.name}, uri: {resource.uri}, repository_id: {resource.repository_id}")
        
        # Get file path using method with database session
        file_path = ResourceService.get_resource_file_path(resource_id, db)
        logger.info(f"File path: {file_path}")
        
        if not file_path:
            logger.error(f"No file path returned for resource {resource_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File path not found"
            )
        
        if not os.path.exists(file_path):
            logger.error(f"File does not exist at path: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found on disk"
            )
        
        logger.info(f"File exists, returning file info for: {file_path}")
        
        return file_path, resource.uri 