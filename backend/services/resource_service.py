from models.resource import Resource
from repositories.resource_repository import ResourceRepository
from services.folder_service import FolderService
from typing import List, Tuple, Optional
from datetime import datetime
import os
from services.silo_service import SiloService
from utils.logger import get_logger
from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, UploadFile

REPO_BASE_FOLDER = os.path.abspath(os.getenv('REPO_BASE_FOLDER'))
logger = get_logger(__name__)

def _is_transient_llm_failure(exc: BaseException) -> bool:
    """Whether *exc* means "the LLM endpoint is unreachable", not "this file is bad".

    A dead endpoint fails every remaining resource in milliseconds, so a batch of
    1000 files would march to 'error' without a single token being extracted.
    Detecting it lets the batch stop and be resumed later instead.

    Walks the ``__cause__``/``__context__`` chain: LangChain wraps provider
    errors, so the connection error is rarely the outermost exception.

    ``TimeoutError`` counts too, and it is the likely one after a laptop
    suspend: LightRAG caps every role LLM call at ``LLM_TIMEOUT`` (240 s by
    default) and re-raises the expiry as the builtin ``TimeoutError``
    (``lightrag/utils.py:215``), not as the provider's own timeout class.
    """
    from openai import APIConnectionError, APITimeoutError  # noqa: WPS433

    transient = (APIConnectionError, APITimeoutError, TimeoutError)
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, transient):
            return True
        current = current.__cause__ or current.__context__
    return False


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

        # Claim the silo before writing anything: LightRAG cannot have two
        # ingestion runs on the same silo (its asyncio locks are per-event-loop).
        # The lock is a PostgreSQL advisory lock, so it is honoured across all
        # uvicorn workers — the previous in-memory guard was invisible to the
        # other three. Held until the background thread finishes.
        from services import silo_indexing_lock

        _repo = ResourceRepository.get_repository_by_id(db, repository_id)
        silo_id = _repo.silo_id if _repo and _repo.silo_id else 0
        lock_conn = silo_indexing_lock.acquire(silo_id) if silo_id else None
        if silo_id and lock_conn is None:
            logger.warning(
                "Rejected concurrent ingestion for silo %s (repository %s)",
                silo_id, repository_id,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This silo already has an active ingestion in progress. "
                       "Please wait for it to complete before uploading more files.",
            )

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
        if not created_resources:
            # Nothing to index: the lock would otherwise be held until the
            # process dies, blocking the silo forever.
            silo_indexing_lock.release(lock_conn, silo_id)
            lock_conn = None
        if created_resources:
            try:
                ResourceRepository.commit(db)
                logger.info(f"Successfully saved {len(created_resources)} resources to database")

                session_id = ResourceService._index_resources_background(
                    created_resources, silo_id, lock_conn=lock_conn,
                )
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
    def _index_resources_background(
        resources: List[Resource], silo_id: int = 0, lock_conn=None
    ) -> str:
        """Launch indexing in a background thread and return a session_id.

        Progress is tracked by real chunk count, not file count.
        The client connects to the SSE endpoint with this session_id to
        monitor the ingestion progress bar in real time.

        ``lock_conn`` holds the silo's advisory lock (see
        :mod:`services.silo_indexing_lock`). It is released when the thread
        finishes — including on failure, or the silo would stay blocked until
        the worker process dies.
        """
        import uuid
        import threading

        session_id = str(uuid.uuid4())

        # Snapshot resource IDs and names so we don't hold ORM refs across threads
        resource_snapshots = [
            (r.resource_id, getattr(r, 'uri', str(r.resource_id)))
            for r in resources
        ]

        # Stamp the batch synchronously, before the thread starts: the client
        # connects to the SSE stream as soon as the upload response returns, and
        # that stream reads progress from these rows.  Stamping inside the thread
        # would leave a window where the batch looks finished.
        batch_started_at = datetime.now()
        from db.database import SessionLocal as _StampSession
        from models.resource import Resource as _StampResource
        _stamp_db = _StampSession()
        try:
            # status: 'pending' too — a resource queued from a previous run's
            # 'error' would otherwise count toward failed_chunks (which counts
            # by current status, see get_indexing_progress) before this run
            # ever got to retry it. The processing loop below sets 'indexing'
            # the moment it actually starts on each resource.
            _stamp_db.query(_StampResource).filter(
                _StampResource.resource_id.in_([rid for rid, _ in resource_snapshots])
            ).update(
                {
                    'progress_started_at': batch_started_at,
                    'progress_done': 0,
                    'progress_total': None,
                    'status': 'pending',
                },
                synchronize_session=False,
            )
            _stamp_db.commit()
        finally:
            _stamp_db.close()

        def _run():
            import asyncio
            from models.resource import Resource as ResourceModel
            from db.database import SessionLocal as _SessionLocal

            # Background threads have no event loop; LightRAG/Neo4j async code
            # requires one.  Create a dedicated loop for this thread.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Phase 1: extract docs per resource to get the real chunk count.
            # Extraction is CPU-only (no LLM calls), so this adds only seconds.
            # The counts are also written to the rows so the progress bar can be
            # served from the DB by any uvicorn worker (the in-memory session
            # above only exists in this process).
            chunk_counts = {}
            for resource_id, _ in resource_snapshots:
                _db = _SessionLocal()
                try:
                    resource_obj = _db.query(ResourceModel).filter_by(resource_id=resource_id).first()
                    if resource_obj:
                        count = SiloService.count_resource_chunks(resource_obj)
                        chunk_counts[resource_id] = count
                        resource_obj.progress_total = count or None
                        _db.commit()
                finally:
                    _db.close()

            total_chunks = sum(chunk_counts.values()) or len(resource_snapshots)

            from models.silo import Silo as SiloModel
            _db_silo = _SessionLocal()
            try:
                _silo_obj = _db_silo.query(SiloModel).filter_by(silo_id=silo_id).first()
                _is_lightrag_silo = bool(_silo_obj and _silo_obj.vector_db_type == 'LIGHTRAG')
            finally:
                _db_silo.close()

            # Phase 2: index each resource, reporting progress by chunk count.
            cumulative = 0
            failed = 0
            try:
                if _is_lightrag_silo:
                    # LightRAG path: enqueue every resource first (fast, no LLM
                    # calls), then drain the whole batch in ONE
                    # apipeline_process_enqueue_documents() run. Doing one
                    # enqueue+process cycle per resource (the branch below) caps
                    # concurrency at that single resource's own page count —
                    # MAX_PARALLEL_INSERT/MAX_ASYNC_LLM never fill up unless one
                    # PDF alone has that many pages. See
                    # LightRAGStore.process_enqueued_documents's docstring.
                    doc_ids_by_resource = {}
                    enqueued_resource_ids = []
                    for resource_id, resource_name in resource_snapshots:
                        resource_chunks = chunk_counts.get(resource_id, 1)
                        try:
                            _db = _SessionLocal()
                            resource_obj = _db.query(ResourceModel).filter_by(resource_id=resource_id).first()
                            _db.close()
                            if not resource_obj:
                                continue
                            _db_pre = _SessionLocal()
                            try:
                                r_pre = _db_pre.query(ResourceModel).filter_by(resource_id=resource_id).first()
                                if r_pre:
                                    r_pre.status = 'indexing'
                                    _db_pre.commit()
                            finally:
                                _db_pre.close()

                            _, ids_by_res = SiloService.enqueue_resource(resource_obj)
                            if ids_by_res:
                                doc_ids_by_resource.update(ids_by_res)
                                enqueued_resource_ids.append(resource_id)
                            else:
                                # Nothing extracted (empty file, no embedding
                                # service, etc.) — nothing to process, done now.
                                _db2 = _SessionLocal()
                                try:
                                    r = _db2.query(ResourceModel).filter_by(resource_id=resource_id).first()
                                    if r:
                                        r.status = 'ready'
                                        r.progress_done = r.progress_total
                                        _db2.commit()
                                finally:
                                    _db2.close()
                                cumulative += resource_chunks
                        except Exception as e:
                            logger.error(f"Failed to enqueue resource {resource_id}: {e}")
                            _db3 = _SessionLocal()
                            try:
                                r = _db3.query(ResourceModel).filter_by(resource_id=resource_id).first()
                                if r:
                                    r.status = 'error'
                                    r.progress_done = r.progress_total
                                    _db3.commit()
                            finally:
                                _db3.close()
                            failed += resource_chunks
                            cumulative += resource_chunks
                            if _is_transient_llm_failure(e):
                                logger.warning(
                                    "Aborting batch after a transient failure during enqueue; "
                                    "remaining resource(s) left pending",
                                )
                                break

                    if doc_ids_by_resource:
                        def _on_progress_multi(rid, n, total):
                            # Same best-effort DB write as the non-batched
                            # _on_progress below, keyed by resource id instead
                            # of a closed-over single resource.
                            _db_p = _SessionLocal()
                            try:
                                _db_p.query(ResourceModel).filter_by(resource_id=rid).update(
                                    {'progress_done': n, 'progress_total': total}
                                )
                                _db_p.commit()
                            except Exception:
                                _db_p.rollback()
                            finally:
                                _db_p.close()

                        try:
                            SiloService.process_enqueued_batch(
                                silo_id, doc_ids_by_resource, progress_callback=_on_progress_multi,
                            )
                            batch_failed = False
                        except Exception as e:
                            logger.error(f"Failed to process enqueued batch for silo {silo_id}: {e}")
                            batch_failed = True

                        for resource_id in enqueued_resource_ids:
                            resource_chunks = chunk_counts.get(resource_id, 1)
                            _db4 = _SessionLocal()
                            try:
                                r = _db4.query(ResourceModel).filter_by(resource_id=resource_id).first()
                                if r:
                                    # Matches the pre-batch behaviour: a resource
                                    # is only marked 'error' when the LightRAG
                                    # call itself failed outright, not when some
                                    # of its individual pages ended up in
                                    # doc_status='failed' (LightRAG already
                                    # tolerates and logs those per-chunk).
                                    r.status = 'error' if batch_failed else 'ready'
                                    r.progress_done = r.progress_total
                                    _db4.commit()
                            finally:
                                _db4.close()
                            if batch_failed:
                                failed += resource_chunks
                            cumulative += resource_chunks
                else:
                    for resource_id, resource_name in resource_snapshots:
                        resource_chunks = chunk_counts.get(resource_id, 1)
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

                                last_reported = [-1]
                                def _on_progress(n, total=None, _rid=resource_id):
                                    # Progress lives on the row: every uvicorn worker
                                    # can serve it and it survives F5 and restarts.
                                    # The poller ticks every 0.5 s but only writes
                                    # when n moves.
                                    if n == last_reported[0]:
                                        return
                                    last_reported[0] = n
                                    _db_p = _SessionLocal()
                                    try:
                                        _db_p.query(ResourceModel).filter_by(resource_id=_rid).update(
                                            {'progress_done': n, 'progress_total': total}
                                        )
                                        _db_p.commit()
                                    except Exception:
                                        _db_p.rollback()  # progress is best-effort
                                    finally:
                                        _db_p.close()
                                SiloService.index_resource(resource_obj, progress_callback=_on_progress)
                            # Mark resource as ready in DB
                            _db2 = _SessionLocal()
                            try:
                                r = _db2.query(ResourceModel).filter_by(resource_id=resource_id).first()
                                if r:
                                    r.status = 'ready'
                                    # Keep done == total: the batch totals are summed
                                    # across finished files too, so the bar must not
                                    # lose them when a file completes.
                                    r.progress_done = r.progress_total
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
                                    r.progress_done = r.progress_total
                                    _db3.commit()
                            finally:
                                _db3.close()
                            failed += resource_chunks
                            cumulative += resource_chunks
                            if _is_transient_llm_failure(e):
                                # The endpoint is down: every remaining resource would
                                # fail in milliseconds. Stop and leave them 'pending'
                                # so "resume indexing" can pick them up later.
                                logger.warning(
                                    "Aborting batch after a transient LLM failure; "
                                    "%s resource(s) left pending",
                                    len(resource_snapshots) - resource_snapshots.index(
                                        (resource_id, resource_name)
                                    ) - 1,
                                )
                                break

            finally:
                loop.close()

            logger.info(
                "Background indexing done: %s/%s chunks, %s failed",
                cumulative - failed, total_chunks, failed,
            )

        def _run_and_unlock():
            import time
            from services import silo_indexing_lock
            from tools.vector_stores.lightrag_store import reset_lightrag_postgres_pool

            # LightRAG's Postgres client pool is process-wide and bound forever
            # to whichever event loop first created it, but every indexing job
            # runs in its own throwaway loop (see reset_lightrag_postgres_pool's
            # docstring) — so two indexing jobs must never touch it at once, or
            # whichever finishes first pulls the pool out from under the other.
            # This sentinel id (never a real silo_id) serializes ALL indexing
            # jobs against each other process-wide, on top of the per-silo lock
            # below (which only prevents two runs on the *same* silo).
            _LIGHTRAG_POOL_LOCK_ID = -1
            pool_lock_conn = None
            while pool_lock_conn is None:
                pool_lock_conn = silo_indexing_lock.acquire(_LIGHTRAG_POOL_LOCK_ID)
                if pool_lock_conn is None:
                    time.sleep(0.5)
            try:
                reset_lightrag_postgres_pool()
                _run()
            finally:
                silo_indexing_lock.release(pool_lock_conn, _LIGHTRAG_POOL_LOCK_ID)
                silo_indexing_lock.release(lock_conn, silo_id)

        thread = threading.Thread(
            target=_run_and_unlock, daemon=True, name=f"index-{session_id[:8]}"
        )
        thread.start()

        return session_id

    @staticmethod
    def get_ingestion_liveness(db: Session, repository_id: int) -> dict:
        """Whether a run is alive for this repository, and what could be resumed.

        ``is_indexing`` comes from the silo's advisory lock, not from row
        statuses: a batch killed mid-run leaves rows in 'pending'/'indexing'
        forever, so statuses alone would report a dead run as live and the UI
        would sit on a frozen progress bar with no way out.
        """
        from services import silo_indexing_lock

        repo = ResourceRepository.get_repository_by_id(db, repository_id)
        silo_id = repo.silo_id if repo and repo.silo_id else 0
        alive = bool(silo_id) and silo_indexing_lock.is_locked(db.connection(), silo_id)
        unfinished = (
            db.query(func.count(Resource.resource_id))
            .filter(
                Resource.repository_id == repository_id,
                Resource.status.in_(('pending', 'error', 'indexing')),
            )
            .scalar()
        ) or 0
        return {
            "is_indexing": alive,
            # Nothing to resume while a run is alive: it will get to them.
            "resumable": 0 if alive else unfinished,
        }

    @staticmethod
    def resume_indexing(db: Session, repository_id: int) -> Tuple[Optional[str], int]:
        """Re-run indexing for every resource of *repository_id* not yet indexed.

        Resuming needs no checkpoint of our own: LightRAG skips documents already
        in ``PROCESSED`` and resets interrupted ones (``PROCESSING``/``PARSING``/
        ``ANALYZING``/``FAILED``) back to ``PENDING``
        (``lightrag/pipeline.py:1457-1470``), so a file that stopped at 70% picks
        up at 70% instead of re-extracting what is already in the graph.

        Returns ``(session_id, resources_queued)``; the session id is ``None``
        when there was nothing left to index.

        Raises HTTPException(409) if this silo is already being indexed.
        """
        # Ordered by resource_id: without it Postgres returns these in an
        # unspecified order, so which file resuming starts with — and which
        # ones are still unfinished when the user stops again — looked random
        # from one resume to the next instead of steadily working through
        # the same queue.
        pending = (
            db.query(Resource)
            .filter(
                Resource.repository_id == repository_id,
                Resource.status.in_(('pending', 'error', 'indexing')),
            )
            .order_by(Resource.resource_id)
            .all()
        )
        if not pending:
            return None, 0

        repo = ResourceRepository.get_repository_by_id(db, repository_id)
        silo_id = repo.silo_id if repo and repo.silo_id else 0

        from services import silo_indexing_lock

        lock_conn = silo_indexing_lock.acquire(silo_id) if silo_id else None
        if silo_id and lock_conn is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This silo already has an active ingestion in progress. "
                       "Please wait for it to complete before resuming.",
            )

        logger.info(
            "Resuming indexing of repository %s: %s resource(s)",
            repository_id, len(pending),
        )
        session_id = ResourceService._index_resources_background(
            pending, silo_id, lock_conn=lock_conn,
        )
        return session_id, len(pending)

    @staticmethod
    def get_indexing_progress(db: Session, repository_id: int) -> Optional[dict]:
        """Return the live indexing progress of a repository, read from the rows.

        Derived from ``Resource.progress_*`` instead of the in-memory session
        tracker: the indexing thread lives in one uvicorn worker while HTTP
        requests are balanced across all of them, so anything in process memory
        is invisible to roughly half the requests.  The rows are shared, so this
        also survives a backend restart and a browser reload.

        A batch is the set of rows sharing one ``progress_started_at`` stamp; the
        newest batch that still has unfinished rows is the active one.  Finished
        rows of that batch stay in the totals so the percentage never jumps back.

        Returns ``None`` when nothing is indexing.
        """
        active_batch = (
            db.query(func.max(Resource.progress_started_at))
            .filter(
                Resource.repository_id == repository_id,
                Resource.status.in_(('pending', 'indexing')),
            )
            .scalar()
        )
        if active_batch is None:
            return None

        done, total, failed = (
            db.query(
                func.coalesce(func.sum(Resource.progress_done), 0),
                func.coalesce(func.sum(Resource.progress_total), 0),
                func.count(Resource.resource_id).filter(Resource.status == 'error'),
            )
            .filter(
                Resource.repository_id == repository_id,
                Resource.progress_started_at == active_batch,
            )
            .one()
        )
        current = (
            db.query(Resource.uri)
            .filter(
                Resource.repository_id == repository_id,
                Resource.progress_started_at == active_batch,
                Resource.status == 'indexing',
            )
            .limit(1)
            .scalar()
        )

        elapsed = (datetime.now() - active_batch).total_seconds()
        percent = min(100.0, done * 100 / total) if total else 0.0
        # Linear extrapolation from the batch start — same formula the in-memory
        # tracker used.  Optimistic early on: the first pages run before entity
        # merging piles up.
        remaining = (total - done) * elapsed / done if done else None

        return {
            'session_id': active_batch.isoformat(),
            'total_chunks': total,
            'processed_chunks': done,
            'failed_chunks': failed,
            'progress_percent': percent,
            'current_chunk_name': current or '',
            'elapsed_seconds': round(elapsed, 1),
            'estimated_remaining_seconds': round(remaining, 1) if remaining else None,
            'estimated_total_time_seconds': round(elapsed + remaining, 1) if remaining else None,
        }

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

        # Concurrent-ingestion rejection now lives in create_multiple_resources,
        # which takes an advisory lock before writing anything (the previous
        # in-memory check could not see runs started by other uvicorn workers).

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