"""
Temporary Media Service - Processes video/audio files for temporary session-based context.

This service handles the temporary (session-scoped) video processing for the playground,
using the same core processing pipeline as repository media processing.

Processing flow:
1. Delegates to MediaProcessingService.process_media_full() for:
   - Audio extraction from video files
   - Transcription using configured AI services  
   - Chunking of transcripts
2. Indexes chunks into temporary vector collections (temp_session_*)
3. Provides search and cleanup utilities for session collections

Shared infrastructure:
- MediaProcessingService.process_media_full() - core processing (shared with repositories)
- VectorStoreFactory - vector operations
"""

import os
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
from sqlalchemy.orm import Session
from langchain_core.documents import Document

from utils.logger import get_logger
from services.media_processing_service import MediaProcessingService
from tools.vector_store_factory import VectorStoreFactory
from db.database import db as db_obj

logger = get_logger(__name__)

# Prefix for temporary collections
TEMP_COLLECTION_PREFIX = 'temp_session_'


class TemporaryMediaService:
    """
    Service for processing video/audio files into temporary searchable context.
    
    The content is stored in temporary vector collections that are cleaned up
    when the session ends.
    """
    
    def __init__(self, tmp_base_folder: str = None):
        """
        Initialize the service.
        
        Args:
            tmp_base_folder: Base folder for temporary files
        """
        from utils.config import get_app_config
        app_config = get_app_config()
        self._tmp_base_folder = tmp_base_folder or app_config['TMP_BASE_FOLDER']
        self._temp_media_dir = os.path.join(self._tmp_base_folder, "temp_media")
        os.makedirs(self._temp_media_dir, exist_ok=True)
    
    async def process_video_for_session(
        self,
        video_path: str,
        session_id: str,
        embedding_service,
        ai_service_id: int,
        db: Session,
        filename: str = None,
        media_id: str = None,
        forced_language: str = None,
        chunk_min_duration: int = 30,
        chunk_max_duration: int = 120,
        chunk_overlap: int = 5,
        progress_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process a video/audio file and index its content for temporary session use.
        
        Uses MediaProcessingService.process_media_full() for the core processing,
        then indexes chunks in a temporary collection for the session.
        
        Args:
            video_path: Path to the video/audio file
            session_id: Unique session identifier (used for collection naming)
            embedding_service: Embedding service configuration to use
            ai_service_id: AI service ID for transcription (Whisper)
            db: Database session
            filename: Original filename for metadata
            media_id: Optional explicit media ID (for tracking cleanup)
            forced_language: Force transcription language (e.g., 'es', 'en'). None for auto-detect.
            chunk_min_duration: Minimum chunk duration in seconds
            chunk_max_duration: Maximum chunk duration in seconds
            chunk_overlap: Overlap between chunks in seconds
            progress_callback: Optional callback function to report progress (called with status string)
            
        Returns:
            Dict with processing results including collection name and chunk count
        """
        try:
            logger.info(f"Processing video for session {session_id}: {video_path}")
            
            # Use provided media_id or generate unique one
            if not media_id:
                media_id = str(uuid.uuid4())[:8]
            
            # Step 1: Report 'transcribing' status - this covers audio extraction + transcription
            # (transcription is the longest step, so we show this status during the whole process_media_full call)
            if progress_callback:
                await progress_callback('transcribing')
            
            # Use unified processing pipeline (includes audio extraction and transcription)
            result = MediaProcessingService.process_media_full(
                file_path=video_path,
                output_dir=self._temp_media_dir,
                media_id=media_id,
                db=db,
                ai_service_id=ai_service_id,
                language=forced_language,
                chunk_min_duration=chunk_min_duration,
                chunk_max_duration=chunk_max_duration,
                chunk_overlap=chunk_overlap,
                filename=filename,
                log_chunks=True,
                cleanup_audio=True
            )
            
            chunks_data = result['chunks']
            
            # Step 2: Prepare documents for indexing in temporary collection
            collection_name = TEMP_COLLECTION_PREFIX + session_id
            documents = []
            
            created_at = datetime.utcnow().isoformat()
            
            for chunk in chunks_data:
                metadata = {
                    "session_id": session_id,
                    "media_id": media_id,
                    "chunk_index": chunk.get('chunk_index', 0),
                    "start_time": chunk.get('start_time', 0),
                    "end_time": chunk.get('end_time', 0),
                    "duration": chunk.get('end_time', 0) - chunk.get('start_time', 0),
                    "content_type": "temp_video_chunk",
                    "filename": chunk.get('filename', os.path.basename(video_path)),
                    "language": chunk.get('language', result['language']),
                    "total_duration": chunk.get('total_duration', result['duration']),
                    "created_at": created_at
                }
                
                doc = Document(
                    page_content=chunk.get('text', ''),
                    metadata=metadata
                )
                documents.append(doc)
            
            # Step 3: Index documents in temporary collection
            if progress_callback:
                await progress_callback('indexing')
            if documents:
                vector_store = VectorStoreFactory.get_vector_store(db_obj, 'PGVECTOR')
                vector_store.index_documents(
                    collection_name,
                    documents,
                    embedding_service
                )
                logger.info(f"Indexed {len(documents)} chunks in collection {collection_name}")
            
            return {
                "success": True,
                "collection_name": collection_name,
                "chunk_count": len(documents),
                "media_id": media_id,
                "language": result['language'],
                "duration": result['duration'],
                "full_transcript": result.get('full_transcript', '')
            }
            
        except Exception as e:
            logger.error(f"Error processing video for session {session_id}: {str(e)}")
            raise
    
    @staticmethod
    def get_temp_collection_name(session_id: str) -> str:
        """Get the collection name for a session."""
        return TEMP_COLLECTION_PREFIX + session_id
    
    @staticmethod
    def search_temp_collection(
        session_id: str,
        query: str,
        embedding_service,
        k: int = 10
    ) -> List[Document]:
        """
        Search in a temporary session collection.
        
        Args:
            session_id: Session identifier
            query: Search query
            embedding_service: Embedding service to use
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        collection_name = TEMP_COLLECTION_PREFIX + session_id
        
        try:
            vector_store = VectorStoreFactory.get_vector_store(db_obj, 'PGVECTOR')
            return vector_store.search_similar_documents(
                collection_name,
                query,
                embedding_service=embedding_service,
                filter_metadata={},
                k=k
            )
        except Exception as e:
            logger.warning(f"Error searching temp collection {collection_name}: {str(e)}")
            return []
    
    @staticmethod
    def cleanup_session(session_id: str, embedding_service=None) -> bool:
        """
        Clean up temporary collection for a session.
        
        Args:
            session_id: Session identifier
            embedding_service: Optional embedding service (may be needed for some operations)
            
        Returns:
            True if cleanup was successful
        """
        collection_name = TEMP_COLLECTION_PREFIX + session_id
        
        try:
            vector_store = VectorStoreFactory.get_vector_store(db_obj, 'PGVECTOR')
            vector_store.delete_collection(collection_name, embedding_service)
            logger.info(f"Cleaned up temporary collection: {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Error cleaning up temp collection {collection_name}: {str(e)}")
            return False
    
    @staticmethod
    def delete_media_chunks(session_id: str, media_id: str) -> bool:
        """
        Delete chunks for a specific media from the temporary collection.
        
        This allows removing a single video's chunks without deleting the entire
        session collection (useful when a user removes one video but keeps others).
        
        Args:
            session_id: Session identifier
            media_id: Media identifier to delete chunks for
            
        Returns:
            True if deletion was successful
        """
        collection_name = TEMP_COLLECTION_PREFIX + session_id
        
        try:
            vector_store = VectorStoreFactory.get_vector_store(db_obj, 'PGVECTOR')
            
            # Delete documents matching the media_id
            vector_store.delete_documents(
                collection_name,
                ids={"media_id": {"$eq": media_id}},
                embedding_service=None
            )
            logger.info(f"Deleted chunks for media {media_id} from collection {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Error deleting media chunks for {media_id}: {str(e)}")
            return False
    
    @staticmethod
    def temp_collection_exists(session_id: str) -> bool:
        """Check if a temporary collection exists for a session."""
        collection_name = TEMP_COLLECTION_PREFIX + session_id
        
        try:
            vector_store = VectorStoreFactory.get_vector_store(db_obj, 'PGVECTOR')
            return vector_store.collection_exists(collection_name)
        except Exception as e:
            logger.warning(f"Error checking temp collection {collection_name}: {str(e)}")
            return False
