from models.media import Media
from db.database import SessionLocal
from services.media_processing_service import MediaProcessingService
from services.silo_service import SiloService
from utils.logger import get_logger
import os
import yt_dlp
from datetime import datetime

REPO_BASE_FOLDER = os.path.abspath(os.getenv('REPO_BASE_FOLDER'))
logger = get_logger(__name__)

def process_media_task_sync(media_id: int):
    """
    Process media: download (if YouTube), extract audio, transcribe, chunk, index
    
    Uses MediaProcessingService.process_media_full() for the core processing,
    which is shared with the playground video processing.
    
    Flow:
    1. Download video if YouTube source
    2. Extract and normalize audio (via process_media_full)
    3. Transcribe using Whisper (via process_media_full)
    4. Create chunks from transcription (via process_media_full)
    5. Index chunks in vector database (via SiloService)
    6. Update media status
    """
    db = SessionLocal()
    
    try:
        # Fetch media
        media = db.query(Media).filter(Media.media_id == media_id).first()
        if not media:
            logger.error(f"Media {media_id} not found")
            return
        
        logger.info(f"Starting processing for media {media_id} ({media.source_type})")
        
        # Step 1: Download if YouTube (using unified download function)
        if media.source_type == 'youtube':
            media.status = 'downloading'
            db.commit()
            
            output_dir = os.path.join(REPO_BASE_FOLDER, str(media.repository_id))
            file_path = _download_youtube_video(media.source_url, str(media_id), output_dir)
            media.file_path = file_path
            db.commit()
            
            logger.info(f"Downloaded YouTube video for media {media_id}")
        
        # Step 2, 3 & 4: Extract audio, transcribe, and create chunks using unified pipeline
        media.status = 'processing'
        db.commit()
        
        output_dir = os.path.join(REPO_BASE_FOLDER, str(media.repository_id))
        
        result = MediaProcessingService.process_media_full(
            file_path=media.file_path,
            output_dir=output_dir,
            media_id=str(media_id),
            db=db,
            ai_service_id=media.transcription_service_id,
            language=media.forced_language,
            chunk_min_duration=media.chunk_min_duration or 30,
            chunk_max_duration=media.chunk_max_duration or 120,
            chunk_overlap=media.chunk_overlap or 0,
            filename=media.name,
            log_chunks=True,
            cleanup_audio=True
        )
        
        # Update media with transcription metadata
        media.language = result['language']
        media.duration = float(result['duration'])
        db.commit()
        
        chunks_data = result['chunks']
        logger.info(f"Created {len(chunks_data)} chunks for media {media_id}")

        # Step 5: Index chunks via SiloService (repository-specific indexing)
        media.status = 'indexing'
        db.commit()

        for chunk_data in chunks_data:
            SiloService.index_media_chunk(chunk_data, media, db)

        logger.info(f"Indexed {len(chunks_data)} chunks for media {media_id}")
        
        # Step 6: Mark as ready
        media.status = 'ready'
        media.processed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"✅ Media {media_id} processed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error processing media {media_id}: {str(e)}")
        
        # Update status to error
        try:
            media = db.query(Media).filter(Media.media_id == media_id).first()
            if media:
                media.status = 'error'
                media.error_message = str(e)[:500]  # Limit error message length
                db.commit()
        except Exception as update_error:
            logger.error(f"Failed to update error status: {str(update_error)}")
        
    finally:
        db.close()


def _download_youtube_video(url: str, media_id: str, output_dir: str, return_title: bool = False):
    """
    Download YouTube video using yt-dlp (unified function for both repository and playground)
    
    Args:
        url: YouTube URL
        media_id: Media ID for filename
        output_dir: Directory to save the video
        return_title: If True, return tuple (path, title); if False, return just path
    
    Returns:
        Path to downloaded video file, or tuple (path, title) if return_title=True
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{media_id}.%(ext)s")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': False,
        'merge_output_format': 'mp4',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            video_title = info.get('title', f'YouTube Video {media_id}')
            
            # yt-dlp may add .mp4 extension
            actual_path = filename
            if not os.path.exists(actual_path):
                # Try with .mp4 extension
                actual_path = os.path.join(output_dir, f"{media_id}.mp4")
            
            logger.info(f"Downloaded YouTube video to: {actual_path}")
            
            if return_title:
                return actual_path, video_title
            return actual_path
            
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {str(e)}")
        raise


def process_playground_youtube_task(
    youtube_url: str,
    session_id: str,
    embedding_service_id: int,
    ai_service_id: int,
    media_id: str,
    file_id: str,
    session_key: str,
    tmp_base_folder: str,
    forced_language: str = None,
    chunk_min_duration: int = None,
    chunk_max_duration: int = None,
    chunk_overlap: int = None
):
    """
    Background task to download YouTube video and process for playground (temporary collection).
    
    Configuration:
    - forced_language: Force transcription language (e.g., 'es', 'en'). None for auto-detect.
    - chunk_min_duration: Minimum chunk duration in seconds (default: 30)
    - chunk_max_duration: Maximum chunk duration in seconds (default: 120)
    - chunk_overlap: Overlap between chunks in seconds (default: 5)
    
    Flow:
    1. Download YouTube video using yt-dlp
    2. Extract and normalize audio
    3. Transcribe using Whisper
    4. Create chunks from transcription
    5. Index chunks in temporary vector collection
    6. Update file metadata status
    """
    print(f"[BACKGROUND TASK] process_playground_youtube_task STARTED for {youtube_url}")
    logger.info(f"[BACKGROUND TASK] process_playground_youtube_task STARTED for {youtube_url}")
    
    from services.temporary_media_service import TemporaryMediaService
    from services.file_management_service import FileManagementService
    from repositories.embedding_service_repository import EmbeddingServiceRepository
    import asyncio
    
    db = SessionLocal()
    video_path = None
    file_service = FileManagementService()
    
    # Create a reusable event loop for the entire task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def update_status_sync(status: str, extra_updates: dict = None):
        """Helper to update file status synchronously (outside async context)"""
        updates = {'processing_status': status}
        if extra_updates:
            updates.update(extra_updates)
        loop.run_until_complete(
            file_service.update_file_metadata_by_session_key(
                session_key=session_key,
                file_id=file_id,
                updates=updates
            )
        )
        logger.info(f"Updated playground YouTube status: {status}")
    
    async def update_status_async(status: str):
        """Async callback for progress updates during processing"""
        updates = {'processing_status': status}
        await file_service.update_file_metadata_by_session_key(
            session_key=session_key,
            file_id=file_id,
            updates=updates
        )
        logger.info(f"Updated playground YouTube status: {status}")
    
    try:
        logger.info(f"========== STARTING PLAYGROUND YOUTUBE PROCESSING ==========")
        logger.info(f"YouTube URL: {youtube_url}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"AI Service ID: {ai_service_id}")
        logger.info(f"Embedding Service ID: {embedding_service_id}")
        
        # Step 1: Download YouTube video (sync - before entering async context)
        update_status_sync('downloading')
        output_dir = os.path.join(tmp_base_folder, "persistent", "youtube_downloads")
        video_path, video_title = _download_youtube_video(youtube_url, media_id, output_dir, return_title=True)
        logger.info(f"Downloaded YouTube video: {video_title} to {video_path}")
        
        # Get embedding service from DB
        embedding_service = EmbeddingServiceRepository.get_by_id(db, embedding_service_id)
        if not embedding_service:
            raise Exception(f"Embedding service {embedding_service_id} not found")
        
        # Use filename based on video title
        filename = f"{video_title[:50]}.mp4" if video_title else f"youtube_{media_id}.mp4"
        
        # Use TemporaryMediaService for the actual processing
        temp_media_service = TemporaryMediaService(tmp_base_folder=tmp_base_folder)
        
        result = loop.run_until_complete(
            temp_media_service.process_video_for_session(
                video_path=video_path,
                session_id=session_id,
                embedding_service=embedding_service,
                ai_service_id=ai_service_id,
                db=db,
                filename=filename,
                media_id=media_id,
                forced_language=forced_language,
                chunk_min_duration=chunk_min_duration or 30,
                chunk_max_duration=chunk_max_duration or 120,
                chunk_overlap=chunk_overlap if chunk_overlap is not None else 5,
                progress_callback=update_status_async
            )
        )
        
        if result['success']:
            logger.info(f"Processing successful: {result['chunk_count']} chunks indexed")
            
            # Update file metadata to 'ready'
            update_status_sync('ready', {
                'temp_session_id': session_id,
                'temp_media_id': media_id,
                'filename': filename,
                'file_path': os.path.relpath(video_path, tmp_base_folder) if video_path else None
            })
            logger.info(f"✅ Playground YouTube video processed successfully: {filename}")
        else:
            raise Exception("Processing returned unsuccessful result")
        
    except Exception as e:
        logger.error(f"❌ Error processing playground YouTube video: {str(e)}")
        
        # Update status to error
        try:
            update_status_sync('error', {'error_message': str(e)[:500]})
        except Exception as update_error:
            logger.error(f"Failed to update error status: {str(update_error)}")
        
    finally:
        loop.close()
        db.close()

# _download_youtube is now replaced by _download_youtube_video (unified function above)
# _extract_audio is now in MediaProcessingService


def process_playground_video_task(
    video_path: str,
    session_id: str,
    embedding_service_id: int,
    ai_service_id: int,
    filename: str,
    media_id: str,
    file_id: str,
    session_key: str,  # For updating file metadata
    forced_language: str = None,
    chunk_min_duration: int = None,
    chunk_max_duration: int = None,
    chunk_overlap: int = None
):
    """
    Background task to process video for playground (temporary collection).
    
    Delegates to TemporaryMediaService for the actual processing to avoid code duplication.
    This wrapper handles:
    - Creating a sync context for the async service
    - Managing the database session lifecycle
    - Updating file metadata on completion/error
    
    Flow (via TemporaryMediaService):
    1. Extract and normalize audio
    2. Transcribe using Whisper
    3. Create chunks from transcription
    4. Index chunks in temporary vector collection
    5. Update file metadata status
    """
    print(f"[BACKGROUND TASK] process_playground_video_task STARTED for {filename}")
    logger.info(f"[BACKGROUND TASK] process_playground_video_task STARTED for {filename}")
    
    from services.temporary_media_service import TemporaryMediaService
    from services.file_management_service import FileManagementService
    from repositories.embedding_service_repository import EmbeddingServiceRepository
    import asyncio
    
    db = SessionLocal()
    file_service = FileManagementService()
    
    # Create a reusable event loop for the entire task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def update_status_sync(status: str, extra_updates: dict = None):
        """Helper to update file status synchronously (outside async context)"""
        updates = {'processing_status': status}
        if extra_updates:
            updates.update(extra_updates)
        loop.run_until_complete(
            file_service.update_file_metadata_by_session_key(
                session_key=session_key,
                file_id=file_id,
                updates=updates
            )
        )
        logger.info(f"Updated playground video status: {status}")
    
    async def update_status_async(status: str):
        """Async callback for progress updates during processing"""
        updates = {'processing_status': status}
        await file_service.update_file_metadata_by_session_key(
            session_key=session_key,
            file_id=file_id,
            updates=updates
        )
        logger.info(f"Updated playground video status: {status}")
    
    try:
        logger.info(f"========== STARTING PLAYGROUND VIDEO PROCESSING ==========")
        logger.info(f"Filename: {filename}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Video path: {video_path}")
        logger.info(f"AI Service ID: {ai_service_id}")
        logger.info(f"Embedding Service ID: {embedding_service_id}")
        logger.info(f"Chunking config: min={chunk_min_duration}, max={chunk_max_duration}, overlap={chunk_overlap}")
        logger.info(f"Forced language: {forced_language}")
        
        # Get embedding service from DB
        embedding_service = EmbeddingServiceRepository.get_by_id(db, embedding_service_id)
        if not embedding_service:
            raise Exception(f"Embedding service {embedding_service_id} not found")
        
        # Use TemporaryMediaService for the actual processing
        temp_media_service = TemporaryMediaService()
        
        result = loop.run_until_complete(
            temp_media_service.process_video_for_session(
                video_path=video_path,
                session_id=session_id,
                embedding_service=embedding_service,
                ai_service_id=ai_service_id,
                db=db,
                filename=filename,
                media_id=media_id,
                forced_language=forced_language,
                chunk_min_duration=chunk_min_duration or 30,
                chunk_max_duration=chunk_max_duration or 120,
                chunk_overlap=chunk_overlap if chunk_overlap is not None else 5,
                progress_callback=update_status_async
            )
        )
        
        if result['success']:
            logger.info(f"Processing successful: {result['chunk_count']} chunks indexed")
            
            # Update file metadata to 'ready'
            update_status_sync('ready', {
                'temp_session_id': session_id,
                'temp_media_id': media_id
            })
            logger.info(f"✅ Playground video processed successfully: {filename}")
        else:
            raise Exception("Processing returned unsuccessful result")
        
    except Exception as e:
        logger.error(f"❌ Error processing playground video: {str(e)}")
        
        # Update status to error
        try:
            update_status_sync('error', {'error_message': str(e)[:500]})
        except Exception as update_error:
            logger.error(f"Failed to update error status: {str(update_error)}")
        
    finally:
        loop.close()
        db.close()