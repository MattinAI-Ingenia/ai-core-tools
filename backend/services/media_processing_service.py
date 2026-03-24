"""
Media Processing Service - Common audio extraction and transcription utilities.

Encapsulates shared media processing logic used by both:
- Repository media processing (permanent storage)
- Playground/session media processing (temporary context)

Provides:
- Audio extraction and normalization
- Transcription + chunking pipeline
- Video clip extraction
"""

import os
import hashlib
import ffmpeg
from typing import Optional, Dict, Any, List
from pydub import AudioSegment
from sqlalchemy.orm import Session

from utils.logger import get_logger
from services.transcription_service import TranscriptionService

logger = get_logger(__name__)


class MediaProcessingService:
    """Common service for media processing operations."""
    
    @staticmethod
    def extract_audio(
        file_path: str,
        output_dir: str,
        media_id: str
    ) -> str:
        """
        Extract and normalize audio from video/audio file.
        
        Args:
            file_path: Path to video/audio file
            output_dir: Directory to save the extracted audio
            media_id: Unique identifier for naming the output file
            
        Returns:
            Path to normalized audio file (WAV, 16kHz, mono)
            
        Raises:
            Exception: If audio extraction fails
        """
        audio_path = os.path.join(output_dir, f"{media_id}_audio.wav")
        
        try:
            # Load audio from file (works with video and audio formats)
            audio = AudioSegment.from_file(file_path)
            
            # Normalize to mono, 16kHz (optimal for Whisper)
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz
            
            # Export as WAV
            audio.export(audio_path, format='wav')
            
            logger.info(f"Extracted and normalized audio to: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error extracting audio from {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def transcribe_and_chunk(
        audio_path: str,
        db: Session,
        ai_service_id: int,
        language: Optional[str] = None,
        chunk_min_duration: int = 30,
        chunk_max_duration: int = 120,
        chunk_overlap: int = 0
    ) -> Dict[str, Any]:
        """
        Transcribe audio and create chunks in one operation.
        
        Combines TranscriptionService.transcribe_audio() and create_chunks()
        into a single convenient method.
        
        Args:
            audio_path: Path to audio file (WAV recommended)
            db: Database session for service lookup
            ai_service_id: AI Service ID with Whisper configuration
            language: Force language (e.g., 'es', 'en'). None for auto-detect.
            chunk_min_duration: Minimum chunk duration in seconds
            chunk_max_duration: Maximum chunk duration in seconds  
            chunk_overlap: Overlap between chunks in seconds
            
        Returns:
            {
                'language': str,
                'duration': float,
                'segments': List[dict],  # Original segments
                'chunks': List[dict]  # Processed chunks
            }
            
        Raises:
            ValueError: If transcription fails or service not found
        """
        try:
            # Step 1: Transcribe audio
            logger.info(f"Transcribing audio with AI service {ai_service_id}")
            transcription = TranscriptionService.transcribe_audio(
                audio_path,
                language=language,
                db=db,
                ai_service_id=ai_service_id
            )
            
            logger.info(
                f"Transcription complete: {len(transcription['segments'])} segments, "
                f"language: {transcription['language']}, duration: {transcription['duration']:.1f}s"
            )
            
            # Step 2: Create chunks
            chunks = TranscriptionService.create_chunks(
                transcription['segments'],
                min_window=chunk_min_duration,
                max_window=chunk_max_duration,
                overlap=chunk_overlap
            )
            
            logger.info(f"Created {len(chunks)} chunks from transcription")
            
            return {
                'language': transcription['language'],
                'duration': transcription['duration'],
                'segments': transcription['segments'],
                'chunks': chunks,
                'full_transcript': transcription.get('text', '')
            }
            
        except Exception as e:
            logger.error(f"Error in transcribe_and_chunk: {str(e)}")
            raise
    
    @staticmethod
    def process_media_full(
        file_path: str,
        output_dir: str,
        media_id: str,
        db: Session,
        ai_service_id: int,
        language: Optional[str] = None,
        chunk_min_duration: int = 30,
        chunk_max_duration: int = 120,
        chunk_overlap: int = 0,
        filename: Optional[str] = None,
        log_chunks: bool = True,
        cleanup_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Full media processing pipeline: extract audio, transcribe, chunk.
        
        This is the unified processing method used by both:
        - Repository media processing (permanent storage)
        - Playground/session media processing (temporary context)
        
        Args:
            file_path: Path to video/audio file
            output_dir: Directory for temporary files (audio)
            media_id: Unique identifier for this media
            db: Database session for service lookup
            ai_service_id: AI Service ID with Whisper configuration
            language: Force language (e.g., 'es', 'en'). None for auto-detect.
            chunk_min_duration: Minimum chunk duration in seconds
            chunk_max_duration: Maximum chunk duration in seconds
            chunk_overlap: Overlap between chunks in seconds
            filename: Original filename for metadata (defaults to basename of file_path)
            log_chunks: Whether to log detailed chunk information
            cleanup_audio: Whether to delete audio file after processing
            
        Returns:
            {
                'language': str,
                'duration': float,
                'chunks': List[dict],  # Processed chunks with all metadata
                'full_transcript': str,
                'audio_path': str  # Path to audio (if not cleaned up)
            }
            
        Raises:
            Exception: If processing fails
        """
        audio_path = None
        
        try:
            # Step 1: Extract audio
            audio_path = MediaProcessingService.extract_audio(file_path, output_dir, media_id)
            logger.info(f"Extracted audio: {audio_path}")
            
            # Step 2: Transcribe and create chunks
            result = MediaProcessingService.transcribe_and_chunk(
                audio_path=audio_path,
                db=db,
                ai_service_id=ai_service_id,
                language=language,
                chunk_min_duration=chunk_min_duration,
                chunk_max_duration=chunk_max_duration,
                chunk_overlap=chunk_overlap
            )
            
            chunks_data = result['chunks']
            resolved_filename = filename or os.path.basename(file_path)
            
            # Log chunking configuration and results
            if log_chunks:
                logger.info(
                    f"📦 Chunking completed for {media_id}: "
                    f"min_duration={chunk_min_duration}s, max_duration={chunk_max_duration}s, overlap={chunk_overlap}s"
                )
                logger.info(f"📦 Total chunks created: {len(chunks_data)}")
                
                # Log details of each chunk
                for idx, chunk in enumerate(chunks_data):
                    chunk_duration = chunk.get('end_time', 0) - chunk.get('start_time', 0)
                    logger.info(
                        f"📦 Chunk {idx + 1}/{len(chunks_data)}: "
                        f"[{chunk.get('start_time', 0):.1f}s - {chunk.get('end_time', 0):.1f}s] "
                        f"duration={chunk_duration:.1f}s, chars={len(chunk.get('text', ''))}"
                    )
                    chunk_text_preview = chunk.get('text', '')[:100] + ('...' if len(chunk.get('text', '')) > 100 else '')
                    logger.debug(f"📦 Chunk {idx + 1} preview: {chunk_text_preview}")
            
            # Enrich chunks with common metadata
            for idx, chunk in enumerate(chunks_data):
                chunk['chunk_index'] = idx
                chunk['filename'] = resolved_filename
                chunk['language'] = result['language']
                chunk['total_duration'] = result['duration']
            
            # Step 3: Cleanup audio file if requested
            if cleanup_audio and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                audio_path = None
            
            return {
                'language': result['language'],
                'duration': result['duration'],
                'chunks': chunks_data,
                'full_transcript': result.get('full_transcript', ''),
                'audio_path': audio_path
            }
            
        except Exception as e:
            # Cleanup on error
            if cleanup_audio and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            logger.error(f"Error in process_media_full for {media_id}: {str(e)}")
            raise
    