import os
import uuid
import tempfile
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile, HTTPException

from tools.PDFTools import extract_text_from_pdf, convert_pdf_to_images, check_pdf_has_text
from utils.logger import get_logger

logger = get_logger(__name__)


class FileReference:
    """Represents a file reference for agent consumption with visual feedback data"""
    
    # MIME type mapping
    MIME_TYPES = {
        "pdf": "application/pdf",
        "text": "text/plain",
        "image": "image/jpeg",  # Default, will be overridden based on extension
        "document": "application/msword",
        "unknown": "application/octet-stream"
    }
    
    IMAGE_MIME_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp"
    }
    
    def __init__(
        self, 
        file_id: str, 
        filename: str, 
        file_type: str, 
        content: str, 
        file_path: str = None,
        file_size_bytes: int = None,
        conversation_id: str = None,
        temp_session_id: str = None,
        temp_media_id: str = None,
        processing_status: str = None  # Override calculated status when loading from disk
    ):
        self.file_id = file_id
        self.filename = filename
        self.file_type = file_type
        self.content = content
        self.file_path = file_path  # Relative path to TMP_BASE_FOLDER
        self.file_size_bytes = file_size_bytes
        self.conversation_id = conversation_id
        self.uploaded_at = datetime.utcnow()
        
        # Temporary video processing tracking
        self.temp_session_id = temp_session_id  # Session ID for temp vector collection
        self.temp_media_id = temp_media_id  # Media ID for chunks in temp collection
        
        # Determine MIME type
        self.mime_type = self._get_mime_type()
        
        # Calculate processing status and content info
        # Use provided processing_status if given (when loading from disk), otherwise calculate
        self.processing_status = processing_status if processing_status else self._get_processing_status()
        self.has_extractable_content = self._has_extractable_content()
        self.content_preview = self._get_content_preview()
    
    def _get_mime_type(self) -> str:
        """Get MIME type based on file type and extension"""
        if self.file_type == "image":
            ext = os.path.splitext(self.filename)[1].lower() if self.filename else ""
            return self.IMAGE_MIME_TYPES.get(ext, "image/jpeg")
        return self.MIME_TYPES.get(self.file_type, "application/octet-stream")
    
    def _get_processing_status(self) -> str:
        """Determine processing status based on content and file type"""
        if not self.content:
            return "error"
        if self.content.startswith("Error"):
            return "error"
        # Video/audio files have special processing states
        if self.file_type in ("video", "audio"):
            if self.content.startswith("__YOUTUBE_PENDING_PROCESSING__"):
                return "downloading"  # YouTube video being downloaded
            if self.content.startswith("__VIDEO_PENDING_PROCESSING__:"):
                return "pending"  # Uploaded, waiting for transcription
            if self.content.startswith("__VIDEO_PROCESSING__:"):
                return "processing"  # Transcription in progress
            if self.content.startswith("__VIDEO_READY__:"):
                return "ready"  # Transcription complete
            return "ready"  # Legacy content or already processed
        # Images are always "ready" - they're sent directly to vision models
        # No text extraction needed for images
        if self.file_type == "image":
            return "ready"
        # Documents (.doc, .docx) don't have text extraction implemented yet
        if "not implemented" in self.content.lower():
            return "uploaded"  # File uploaded but not fully processed
        return "ready"
    
    def _has_extractable_content(self) -> bool:
        """Check if meaningful content was extracted"""
        if not self.content:
            return False
        # Video/audio files - check processing state
        if self.file_type in ("video", "audio"):
            # Only ready videos/audios have extractable content
            return self.content.startswith("__VIDEO_READY__:")
        # Check for placeholder messages
        placeholder_indicators = [
            "not implemented",
            "Error processing",
            "Image file:",
            "Document file:",
            "File:",
            "__VIDEO_PENDING_PROCESSING__:",
            "__VIDEO_PROCESSING__:",
            "__YOUTUBE_PENDING_PROCESSING__",
            "__YOUTUBE_PENDING__:"
        ]
        return not any(indicator in self.content for indicator in placeholder_indicators)
    
    def _get_content_preview(self, max_length: int = 200) -> Optional[str]:
        """Get preview of extracted content"""
        # Video/audio files - only show preview when ready, not status messages
        if self.file_type in ("video", "audio"):
            if self.content.startswith("__YOUTUBE_PENDING_PROCESSING__"):
                return None  # Status shown in badge, no need for redundant text
            if self.content.startswith("__YOUTUBE_PENDING__:"):
                return None  # Status shown in badge
            if self.content.startswith("__VIDEO_PENDING_PROCESSING__:"):
                return None  # Status shown in badge
            if self.content.startswith("__VIDEO_PROCESSING__:"):
                return None  # Status shown in badge
            if self.content.startswith("__VIDEO_READY__:"):
                # Extract actual transcript preview
                transcript = self.content.replace("__VIDEO_READY__:", "", 1)
                if len(transcript) <= max_length:
                    return transcript
                return transcript[:max_length] + "..."
        if not self.has_extractable_content:
            return None
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size to human readable string"""
        if size_bytes is None:
            return "Unknown"
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with visual feedback data"""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "content": self.content,
            "file_path": self.file_path,
            "uploaded_at": self.uploaded_at.isoformat(),
            # Visual feedback fields
            "file_size_bytes": self.file_size_bytes,
            "file_size_display": self.format_file_size(self.file_size_bytes),
            "processing_status": self.processing_status,
            "content_preview": self.content_preview,
            "has_extractable_content": self.has_extractable_content,
            "mime_type": self.mime_type,
            "conversation_id": self.conversation_id,
            # Temporary video processing fields
            "temp_session_id": self.temp_session_id,
            "temp_media_id": self.temp_media_id
        }


class FileManagementService:
    """Unified file management - used by both public and internal APIs"""
    
    def __init__(self):
        # Persistent file storage on disk
        # Structure: {session_key: {file_id: FileReference}}
        self._files: Dict[str, Dict[str, FileReference]] = {}
        
        # Get TMP_BASE_FOLDER from config
        from utils.config import get_app_config
        app_config = get_app_config()
        self._tmp_base_folder = app_config['TMP_BASE_FOLDER']
        self._persistent_dir = os.path.join(self._tmp_base_folder, "persistent")
        self._temp_dir = os.path.join(self._tmp_base_folder, "uploads")
        
        # Ensure directories exist
        os.makedirs(self._persistent_dir, exist_ok=True)
        os.makedirs(self._temp_dir, exist_ok=True)
        os.makedirs(os.path.join(self._tmp_base_folder, "downloads"), exist_ok=True)
        os.makedirs(os.path.join(self._tmp_base_folder, "images"), exist_ok=True)
        
        # Files will be loaded on-demand per session
    
    async def upload_file(
        self, 
        file: UploadFile, 
        agent_id: int,
        user_context: Dict = None,
        conversation_id: Optional[int] = None
    ) -> FileReference:
        """
        Upload file for agent consumption
        
        Args:
            file: Uploaded file
            agent_id: ID of the agent
            user_context: User context (api_key, user_id, etc.)
            conversation_id: Optional conversation ID to organize files
            
        Returns:
            FileReference object
        """
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Determine file type
            file_type = self._get_file_type(file.filename)
            logger.info(f"Uploading file: {file.filename}, determined type: {file_type}")
            
            # Process file based on type (also returns file size)
            content, temp_path, file_size = await self._process_file_content(file, file_type)
            
            # Create file reference with visual feedback data
            file_ref = FileReference(
                file_id=file_id,
                filename=file.filename,
                file_type=file_type,
                content=content,
                file_path=None,  # Will be set by _save_file_to_disk
                file_size_bytes=file_size,
                conversation_id=str(conversation_id) if conversation_id else None
            )
            
            # Create session key for this agent, user, and conversation
            # conversation_id ensures files are isolated per conversation
            session_key = self._get_session_key(agent_id, user_context, str(conversation_id) if conversation_id else None)
            
            # Initialize session if not exists
            if session_key not in self._files:
                self._files[session_key] = {}
            
            # Save file to disk for persistence (including original file)
            # Note: This also updates file_ref.content for videos to use persistent path
            await self._save_file_to_disk(session_key, file_id, file_ref, temp_path, conversation_id)
            
            # Store file reference in session (after file_path is set)
            self._files[session_key][file_id] = file_ref
            
            # Clean up temporary file after saving to persistent storage
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.info(f"Uploaded file {file.filename} for agent {agent_id}, session {session_key}")
            return file_ref
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    async def process_file_for_agent(
        self, 
        file_path: str, 
        agent: Any
    ) -> str:
        """
        Process file for agent consumption - reuse existing PDF tools
        
        Args:
            file_path: Path to the file
            agent: Agent object
            
        Returns:
            Processed content string
        """
        try:
            # Determine file type from path
            file_type = self._get_file_type_from_path(file_path)
            
            if file_type == "pdf":
                # Use existing PDF tools
                return extract_text_from_pdf(file_path)
            elif file_type in ["txt", "md", "json"]:
                # Read text files directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # For other file types, return basic info
                return f"File: {os.path.basename(file_path)} (type: {file_type})"
                
        except Exception as e:
            logger.error(f"Error processing file for agent: {str(e)}")
            return f"Error processing file: {str(e)}"
    
    async def get_file_reference(self, file_id: str) -> Optional[FileReference]:
        """
        Get file reference by ID
        
        Args:
            file_id: File ID
            
        Returns:
            FileReference or None if not found
        """
        return self._files.get(file_id)
    
    async def get_file(
        self, 
        file_id: str, 
        agent_id: int,
        user_context: Dict = None,
        conversation_id: str = None
    ) -> Optional[FileReference]:
        """
        Get file reference for a specific session.
        
        Args:
            file_id: File ID
            agent_id: ID of the agent
            user_context: User context
            conversation_id: Optional conversation ID
            
        Returns:
            FileReference or None if not found
        """
        try:
            # Get session key for this agent, user, and optionally conversation
            session_key = self._get_session_key(agent_id, user_context, conversation_id)
            
            # Load files for this session if not already loaded
            if session_key not in self._files:
                self._load_session_files(session_key)
            
            # Return file from this session
            if session_key in self._files and file_id in self._files[session_key]:
                return self._files[session_key][file_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting file: {str(e)}")
            return None
    
    async def list_attached_files(
        self, 
        agent_id: int, 
        user_context: Dict = None,
        conversation_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        List attached files for a user session, optionally filtered by conversation.
        
        Args:
            agent_id: ID of the agent
            user_context: User context
            conversation_id: Optional conversation ID for conversation-specific files
            
        Returns:
            List of file references
        """
        try:
            # Get session key for this agent, user, and optionally conversation
            session_key = self._get_session_key(agent_id, user_context, conversation_id)
            
            # Always reload from disk to get latest status updates from background tasks
            # (background tasks run in separate instances and update disk directly)
            self._load_session_files(session_key)
            
            # Return files for this session
            if session_key in self._files:
                files_list = [file_ref.to_dict() for file_ref in self._files[session_key].values()]
                logger.info(f"Returning {len(files_list)} files for session {session_key}")
                for file_data in files_list:
                    logger.info(f"File: {file_data['filename']}, Path: {file_data.get('file_path', 'None')}")
                return files_list
            else:
                return []
            
        except Exception as e:
            logger.error(f"Error listing attached files: {str(e)}")
            return []
    
    async def remove_file(
        self, 
        file_id: str, 
        agent_id: int,
        user_context: Dict = None,
        conversation_id: str = None
    ) -> bool:
        """
        Remove attached file from a session (optionally conversation-specific).
        
        Args:
            file_id: File ID to remove
            agent_id: ID of the agent
            user_context: User context
            conversation_id: Optional conversation ID for conversation-specific files
            
        Returns:
            True if removed successfully
        """
        try:
            # Get session key for this agent, user, and optionally conversation
            session_key = self._get_session_key(agent_id, user_context, conversation_id)
            
            # Load files for this session if not already loaded
            if session_key not in self._files:
                self._load_session_files(session_key)
            
            if session_key in self._files and file_id in self._files[session_key]:
                file_ref = self._files[session_key][file_id]
                
                # If this is a processed video/audio, clean up its chunks from temp collection
                if file_ref.file_type in ("video", "audio") and file_ref.temp_session_id and file_ref.temp_media_id:
                    await self._cleanup_video_chunks(file_ref.temp_session_id, file_ref.temp_media_id)
                # If video is still processing, also try to clean up any partial chunks
                elif file_ref.file_type in ("video", "audio") and file_ref.processing_status == 'processing':
                    # Calculate temp_session_id the same way as during processing
                    user_id = user_context.get('user_id', 'anonymous') if user_context else 'anonymous'
                    conv_id = conversation_id or file_ref.conversation_id or 'no_conv'
                    session_key_str = f"{agent_id}_{user_id}_{conv_id}"
                    temp_session_id = hashlib.sha256(session_key_str.encode()).hexdigest()[:16]
                    temp_media_id = file_id[:8]  # Same as used during processing
                    
                    logger.info(f"Video in progress, attempting cleanup: temp_session_{temp_session_id}, media_id={temp_media_id}")
                    await self._cleanup_video_chunks(temp_session_id, temp_media_id)
                
                del self._files[session_key][file_id]
                
                # Also remove from disk
                await self._remove_file_from_disk(session_key, file_id)
                
                logger.info(f"Removed file {file_id} from session {session_key}")
                return True
            else:
                logger.warning(f"File {file_id} not found for removal in session {session_key}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing file: {str(e)}")
            return False
    
    async def update_file_metadata(
        self,
        agent_id: int,
        file_id: str,
        user_context: Dict = None,
        conversation_id: str = None,
        updates: Dict = None
    ) -> bool:
        """
        Update metadata for a file reference.
        
        Args:
            agent_id: ID of the agent
            file_id: File ID to update
            user_context: User context
            conversation_id: Optional conversation ID
            updates: Dictionary of fields to update
            
        Returns:
            True if updated successfully
        """
        if not updates:
            return True
            
        try:
            session_key = self._get_session_key(agent_id, user_context, conversation_id)
            
            # Load files for this session if not already loaded
            if session_key not in self._files:
                self._load_session_files(session_key)
            
            if session_key not in self._files or file_id not in self._files[session_key]:
                logger.warning(f"File {file_id} not found for update in session {session_key}")
                return False
            
            file_ref = self._files[session_key][file_id]
            
            # Update allowed fields
            for key, value in updates.items():
                if hasattr(file_ref, key):
                    setattr(file_ref, key, value)
            
            # Update the metadata file on disk
            session_dir = os.path.join(self._persistent_dir, session_key)
            metadata_file = os.path.join(session_dir, f"{file_id}.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'w') as f:
                    import json
                    json.dump(file_ref.to_dict(), f, indent=2)
            
            # Also update the content file if content was updated
            if 'content' in updates:
                content_file = os.path.join(session_dir, f"{file_id}.content")
                with open(content_file, 'w', encoding='utf-8') as f:
                    f.write(updates['content'])
                logger.info(f"Updated content file for {file_id}")
            
            logger.info(f"Updated file {file_id} metadata: {list(updates.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating file metadata: {str(e)}")
            return False

    async def update_file_metadata_by_session_key(
        self,
        session_key: str,
        file_id: str,
        updates: Dict = None
    ) -> bool:
        """
        Update metadata for a file reference using session_key directly.
        Used by background tasks that don't have user_context.
        
        Args:
            session_key: Session key (e.g., agent_1_user_1_app_1_conv_14)
            file_id: File ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if updated successfully
        """
        logger.info(f"update_file_metadata_by_session_key called - session_key: {session_key}, file_id: {file_id}, updates: {updates}")
        
        if not updates:
            return True
            
        try:
            # Load files for this session if not already loaded
            if session_key not in self._files:
                logger.info(f"Session {session_key} not in memory, loading from disk...")
                self._load_session_files(session_key)
            
            logger.info(f"Checking if file {file_id} exists in session. Available files: {list(self._files.get(session_key, {}).keys())}")
            
            if session_key not in self._files or file_id not in self._files[session_key]:
                logger.warning(f"File {file_id} not found for update in session {session_key}")
                return False
            
            file_ref = self._files[session_key][file_id]
            
            # Update allowed fields
            for key, value in updates.items():
                if hasattr(file_ref, key):
                    setattr(file_ref, key, value)
            
            # Update the metadata file on disk
            session_dir = os.path.join(self._persistent_dir, session_key)
            metadata_file = os.path.join(session_dir, f"{file_id}.json")
            
            # Always write the metadata file (create if not exists)
            os.makedirs(session_dir, exist_ok=True)
            with open(metadata_file, 'w') as f:
                import json
                json.dump(file_ref.to_dict(), f, indent=2)
            logger.info(f"Wrote metadata to disk: {metadata_file} with processing_status={file_ref.processing_status}")
            
            logger.info(f"Updated file {file_id} metadata by session_key: {list(updates.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating file metadata by session_key: {str(e)}")
            return False

    def _get_file_type(self, filename: str) -> str:
        """Get file type from filename"""
        if not filename:
            return "unknown"
        
        ext = os.path.splitext(filename)[1].lower()
        
        # PDF files
        if ext == '.pdf':
            return "pdf"
        
        # Text files
        elif ext in ['.txt', '.md', '.json', '.csv']:
            return "text"
        
        # Image files
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return "image"
        
        # Document files
        elif ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
            return "document"
        
        # Video files - for transcription and temporal indexing
        elif ext in ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v']:
            return "video"
        
        # Audio files - for transcription and temporal indexing (same processing as video)
        elif ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma']:
            return "audio"
        
        else:
            return "unknown"
    
    def _get_file_type_from_path(self, file_path: str) -> str:
        """Get file type from file path"""
        return self._get_file_type(os.path.basename(file_path))
    
    async def _process_file_content(self, file: UploadFile, file_type: str) -> tuple[str, str, int]:
        """
        Process file content based on file type
        
        Args:
            file: Uploaded file
            file_type: Type of file
            
        Returns:
            Tuple of (processed_content, temp_file_path, file_size_bytes)
        """
        try:
            # Save file temporarily and get size
            temp_path, file_size = await self._save_uploaded_file_with_size(file)
            
            try:
                if file_type == "pdf":
                    # Use existing PDF tools
                    content = extract_text_from_pdf(temp_path)
                    return content, temp_path, file_size
                
                elif file_type == "text":
                    # Read text files
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return content, temp_path, file_size
                
                elif file_type == "image":
                    # For images, return basic info (in production, use OCR)
                    content = f"Image file: {file.filename} (OCR processing not implemented)"
                    return content, temp_path, file_size
                
                elif file_type == "document":
                    # For documents, return basic info (in production, use document processing)
                    content = f"Document file: {file.filename} (Document processing not implemented)"
                    return content, temp_path, file_size
                
                elif file_type in ("video", "audio"):
                    # For video/audio files, mark for temporal processing
                    # The actual transcription will be done asynchronously by TemporaryMediaService
                    content = f"__VIDEO_PENDING_PROCESSING__:{temp_path}"
                    return content, temp_path, file_size
                
                else:
                    # For unknown types, return basic info
                    content = f"File: {file.filename} (type: {file_type})"
                    return content, temp_path, file_size
                    
            except Exception as e:
                logger.error(f"Error processing file content: {str(e)}")
                return f"Error processing file: {str(e)}", temp_path, file_size
                    
        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")
            return f"Error processing file: {str(e)}", None, 0
    
    async def _save_uploaded_file_with_size(self, file: UploadFile) -> tuple[str, int]:
        """Save uploaded file to temporary location and return path with file size"""
        # Create temporary file in TMP_BASE_FOLDER/uploads
        suffix = os.path.splitext(file.filename)[1] if file.filename else ""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=self._temp_dir)
        
        try:
            # Write file content and track size
            content = await file.read()
            file_size = len(content)
            temp_file.write(content)
            temp_file.flush()
            
            return temp_file.name, file_size
        finally:
            temp_file.close()
    
    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location (legacy method for compatibility)"""
        path, _ = await self._save_uploaded_file_with_size(file)
        return path
    
    def _get_session_key(self, agent_id: int, user_context: Dict = None, conversation_id: str = None) -> str:
        """
        Generate session key for agent, user, and optionally conversation combination.
        
        When conversation_id is provided, files are isolated to that specific conversation.
        This allows users to have separate file contexts for different conversations with the same agent.
        
        Args:
            agent_id: ID of the agent
            user_context: User context (api_key, user_id, etc.)
            conversation_id: Optional conversation ID for conversation-specific file isolation
            
        Returns:
            Session key string
        """
        if user_context:
            user_id = user_context.get('user_id', 'anonymous')
            app_id = user_context.get('app_id', 'default')
            # Check if conversation_id is in user_context or passed explicitly
            conv_id = conversation_id or user_context.get('conversation_id')
            
            if conv_id:
                # Conversation-specific file storage
                session_key = f"agent_{agent_id}_user_{user_id}_app_{app_id}_conv_{conv_id}"
            else:
                # Global agent session (files shared across all conversations)
                session_key = f"agent_{agent_id}_user_{user_id}_app_{app_id}"
            
            logger.debug(f"_get_session_key: agent_id={agent_id}, user_id={user_id}, conv_id={conv_id} -> {session_key}")
            return session_key
        else:
            return f"agent_{agent_id}_anonymous"

    async def _save_file_to_disk(self, session_key: str, file_id: str, file_ref: FileReference, original_file_path: str = None, conversation_id: Optional[int] = None):
        """Save file reference to disk for persistence"""
        logger.info(f"_save_file_to_disk called: session_key={session_key}, file_id={file_id}")
        try:
            session_dir = os.path.join(self._persistent_dir, session_key)
            logger.info(f"Creating session dir: {session_dir}")
            os.makedirs(session_dir, exist_ok=True)
            
            # Save original file FIRST (to set file_path before saving metadata)
            if original_file_path and os.path.exists(original_file_path):
                original_filename = os.path.basename(original_file_path)
                original_extension = os.path.splitext(original_filename)[1]
                
                # Determine target directory
                if conversation_id:
                    target_dir = os.path.join(self._tmp_base_folder, "conversations", str(conversation_id))
                else:
                    target_dir = session_dir
                
                os.makedirs(target_dir, exist_ok=True)
                
                original_file = os.path.join(target_dir, f"{file_id}{original_extension}")
                
                import shutil
                shutil.copy2(original_file_path, original_file)
                
                # Calculate relative path from TMP_BASE_FOLDER
                relative_path = os.path.relpath(original_file, self._tmp_base_folder)
                # Ensure forward slashes for URLs
                file_ref.file_path = relative_path.replace(os.sep, '/')
                
                logger.info(f"Saved original file {original_filename} to {original_file} (relative: {relative_path})")
                logger.info(f"FileReference file_path set to: {file_ref.file_path}")
                
                # For video/audio files, update content to use the persistent file path
                if file_ref.file_type in ("video", "audio"):
                    file_ref.content = f"__VIDEO_PENDING_PROCESSING__:{original_file}"
                    logger.info(f"Updated video/audio content to persistent path: {original_file}")
            
            # Save file metadata AFTER setting file_path
            metadata_file = os.path.join(session_dir, f"{file_id}.json")
            logger.info(f"Saving metadata to: {metadata_file}")
            with open(metadata_file, 'w') as f:
                import json
                json.dump(file_ref.to_dict(), f, indent=2)
            logger.info(f"Metadata saved successfully to {metadata_file}")
            
            # Save file content (extracted text)
            content_file = os.path.join(session_dir, f"{file_id}.content")
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(file_ref.content)
                
            logger.info(f"Saved file {file_id} to disk: {session_dir}")
            
        except Exception as e:
            logger.error(f"Error saving file to disk: {str(e)}")

    async def _save_youtube_file_to_disk(self, session_key: str, file_id: str, file_ref: FileReference, youtube_url: str):
        """Save YouTube file reference to disk for persistence (without the actual video file yet)"""
        logger.info(f"_save_youtube_file_to_disk called: session_key={session_key}, file_id={file_id}, url={youtube_url}")
        try:
            session_dir = os.path.join(self._persistent_dir, session_key)
            logger.info(f"Creating session dir: {session_dir}")
            os.makedirs(session_dir, exist_ok=True)
            
            # Save file metadata with youtube URL
            metadata = file_ref.to_dict()
            metadata['source_type'] = 'youtube'
            metadata['source_url'] = youtube_url
            
            metadata_file = os.path.join(session_dir, f"{file_id}.json")
            logger.info(f"Saving YouTube metadata to: {metadata_file}")
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            logger.info(f"YouTube metadata saved successfully to {metadata_file}")
            
            # Save placeholder content
            content_file = os.path.join(session_dir, f"{file_id}.content")
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(f"__YOUTUBE_PENDING__:{youtube_url}")
                
            logger.info(f"Saved YouTube file reference {file_id} to disk: {session_dir}")
            
        except Exception as e:
            logger.error(f"Error saving YouTube file to disk: {str(e)}")

    def _load_session_files(self, session_key: str):
        """Load existing files from disk for a specific session"""
        try:
            if not os.path.exists(self._persistent_dir):
                return
                
            session_path = os.path.join(self._persistent_dir, session_key)
            if not os.path.exists(session_path) or not os.path.isdir(session_path):
                return
                
            # Initialize session if not exists
            if session_key not in self._files:
                self._files[session_key] = {}
            
            # Load files for this session
            for filename in os.listdir(session_path):
                if filename.endswith('.json'):
                    file_id = filename[:-5]  # Remove .json extension
                    metadata_file = os.path.join(session_path, filename)
                    content_file = os.path.join(session_path, f"{file_id}.content")
                    
                    if os.path.exists(content_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                import json
                                metadata = json.load(f)
                            
                            with open(content_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Recreate FileReference with visual feedback data
                            file_ref = FileReference(
                                file_id=metadata['file_id'],
                                filename=metadata['filename'],
                                file_type=metadata['file_type'],
                                content=content,
                                file_path=metadata.get('file_path'),
                                file_size_bytes=metadata.get('file_size_bytes'),
                                conversation_id=metadata.get('conversation_id'),
                                temp_session_id=metadata.get('temp_session_id'),
                                temp_media_id=metadata.get('temp_media_id'),
                                processing_status=metadata.get('processing_status', 'pending')
                            )
                            
                            logger.info(f"Loaded persistent file {file_id} for session {session_key}")
                            logger.info(f"Loaded file_path: {file_ref.file_path}")
                            logger.info(f"Loaded processing_status: {file_ref.processing_status}")
                            if not file_ref.file_path:
                                # Look for the original file in the session directory
                                for filename in os.listdir(session_path):
                                    if filename.startswith(file_id) and not filename.endswith(('.json', '.content')):
                                        original_file = os.path.join(session_path, filename)
                                        if os.path.exists(original_file):
                                            # Calculate relative path
                                            relative_path = os.path.relpath(original_file, self._tmp_base_folder)
                                            file_ref.file_path = relative_path
                                            
                                            # Update metadata file with the new path
                                            metadata['file_path'] = relative_path
                                            with open(metadata_file, 'w') as f:
                                                import json
                                                json.dump(metadata, f, indent=2)
                                            
                                            logger.info(f"Regenerated file_path for {file_id}: {relative_path}")
                                            break
                            
                            self._files[session_key][file_id] = file_ref
                            
                        except Exception as e:
                            logger.error(f"Error loading file {file_id}: {str(e)}")
                            
            logger.info(f"Loaded {len(self._files.get(session_key, {}))} persistent files for session {session_key}")
            
        except Exception as e:
            logger.error(f"Error loading persistent files for session {session_key}: {str(e)}")

    async def _remove_file_from_disk(self, session_key: str, file_id: str):
        """Remove file from disk"""
        try:
            session_dir = os.path.join(self._persistent_dir, session_key)
            metadata_file = os.path.join(session_dir, f"{file_id}.json")
            content_file = os.path.join(session_dir, f"{file_id}.content")
            
            # Remove metadata and content files
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            if os.path.exists(content_file):
                os.remove(content_file)
            
            # Remove original file (look for any file with the file_id as prefix)
            for filename in os.listdir(session_dir):
                if filename.startswith(file_id) and not filename.endswith(('.json', '.content')):
                    original_file = os.path.join(session_dir, filename)
                    if os.path.exists(original_file):
                        os.remove(original_file)
                        logger.info(f"Removed original file {filename}")
                
            logger.info(f"Removed file {file_id} from disk")
            
        except Exception as e:
            logger.error(f"Error removing file from disk: {str(e)}")

    async def _cleanup_video_chunks(self, temp_session_id: str, temp_media_id: str):
        """
        Clean up video chunks from temporary vector collection.
        
        Args:
            temp_session_id: Temporary session ID for the collection
            temp_media_id: Media ID to identify chunks to delete
        """
        try:
            from services.temporary_media_service import TemporaryMediaService
            
            # Delete only chunks for this specific media_id
            TemporaryMediaService.delete_media_chunks(temp_session_id, temp_media_id)
            logger.info(f"Cleaned up video chunks for media {temp_media_id} from session {temp_session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up video chunks: {str(e)}")

    def get_file_stats(self) -> Dict[str, Any]:
        """Get file management statistics"""
        try:
            file_types = {}
            total_size = 0
            total_files = 0
            
            for session_files in self._files.values():
                for file_ref in session_files.values():
                    file_type = file_ref.file_type
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    total_size += len(file_ref.content)
                    total_files += 1
            
            return {
                "total_files": total_files,
                "total_sessions": len(self._files),
                "file_types": file_types,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting file stats: {str(e)}")
            return {} 