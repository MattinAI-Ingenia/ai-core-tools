import os
import re
import asyncio
import ast
import json
import hashlib
import subprocess
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from models.agent import Agent
from models.ocr_agent import OCRAgent
from tools.PDFTools import extract_text_from_pdf, convert_pdf_to_images, check_pdf_has_text
from tools.ocrAgentTools import (
    convert_image_to_base64,
    extract_text_from_image,
    format_data_with_text_llm,
    format_data_from_vision,
    get_data_from_extracted_text,
    get_document_data_from_pages
)
from tools.aiServiceTools import get_llm
from tools.outputParserTools import create_model_from_json_schema
from services.agent_service import AgentService
from services.session_management_service import SessionManagementService
from repositories.agent_execution_repository import AgentExecutionRepository
from utils.logger import get_logger
from utils.config import get_app_config

logger = get_logger(__name__)


class AgentExecutionService:
    """Unified service for agent execution - used by both public and internal APIs"""
    
    # Shared thread pool for blocking operations (LLM calls, file I/O, etc.)
    # This prevents blocking the event loop
    _executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="agent_exec")
    
    def __init__(self, db: Session = None):
        self.agent_service = AgentService()
        self.session_service = SessionManagementService()
        self.agent_execution_repo = AgentExecutionRepository()
        self.db = db
    
    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """
        Get video duration in seconds using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Duration in seconds, or 0 if unable to get duration
        """
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error', '-show_entries',
                    'format=duration', '-of',
                    'default=noprint_wrappers=1:nokey=1', video_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            duration = float(result.stdout.decode().strip())
            logger.info(f"Video duration for {video_path}: {duration}s")
            return duration
        except Exception as e:
            logger.warning(f"Could not get video duration for {video_path}: {e}")
            return 0
    
    @staticmethod
    def _extract_timestamps_from_response(response_text: str, video_duration: float = None) -> List[Tuple[float, float]]:
        """
        Extract timestamp references from the agent's response text.
        
        Supports formats like:
        - "At 01:23" or "at 1:23"
        - "From 02:15 to 02:45" or "desde 02:15 hasta 02:45"
        - "Between 1:00 and 2:30" or "entre 1:00 y 2:30"
        - "entre los minutos 00:25 y 00:42"
        - "desde el inicio hasta el minuto 00:30"
        - "Timestamp: 03:45" or "minuto 3:45"
        - "00:30 - 01:15" or "0:30-1:15"
        - "desde el inicio hasta 00:30" -> 00:00 to 00:30
        - "desde 00:25 hasta el final" -> 00:25 to video_duration
        - "al inicio", "al principio", "al comienzo" -> 00:00
        - "al final", "al fin" -> end of video
        - "desde el segundo 25 al 42" -> 25s to 42s
        - "del segundo 25 al segundo 42" -> 25s to 42s
        
        Args:
            response_text: The agent's response text
            video_duration: Total video duration in seconds (required for "final" keywords)
        
        Returns:
            List of (start_seconds, end_seconds) tuples, sorted chronologically
        """
        timestamps = []
        
        # Default video duration if not provided
        default_end = video_duration if video_duration else 60.0
        
        def parse_timestamp(ts: str) -> float:
            """Convert MM:SS or M:SS to seconds"""
            parts = ts.strip().split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return 0
        
        def parse_seconds(s: str) -> float:
            """Convert a simple number string to seconds"""
            try:
                return float(s.strip())
            except:
                return 0.0
        
        def parse_time_or_keyword(text: str, is_end: bool = False) -> float:
            """Parse a timestamp, seconds value, or keyword like inicio/final"""
            text = text.strip().lower()
            
            # Check for start keywords
            if re.search(r'(inicio|principio|comienzo|beginning|start)', text):
                return 0.0
            
            # Check for end keywords
            if re.search(r'(final|fin|end)', text):
                return default_end
            
            # Try to parse as MM:SS
            match = re.search(r'(\d{1,2}:\d{2})', text)
            if match:
                return parse_timestamp(match.group(1))
            
            # Try to parse as seconds (e.g., "segundo 25" or just "25")
            sec_match = re.search(r'(?:segundos?\s+)?(\d+)', text)
            if sec_match:
                return parse_seconds(sec_match.group(1))
            
            return 0.0 if not is_end else default_end
        
        # Pattern for timestamps in MM:SS or M:SS format (with optional "minuto/minute" prefix)
        ts_pattern = r'(\d{1,2}:\d{2})'
        ts_with_prefix = rf'(?:(?:el\s+)?minutos?\s+)?{ts_pattern}'
        
        # Pattern for seconds as simple numbers (e.g., "segundo 25", "segundos 25")
        sec_pattern = r'(\d+)'
        sec_with_prefix = rf'(?:(?:el\s+)?segundos?\s+){sec_pattern}'
        
        # Pattern for start keywords (inicio, comienzo, principio, start, beginning)
        start_keyword = r'(?:el\s+)?(?:the\s+)?(?:inicio|principio|comienzo|beginning|start)(?:\s+del\s+video)?'
        
        # Pattern for end keywords (final, fin, end)
        end_keyword = r'(?:el\s+)?(?:the\s+)?(?:final|fin|end)(?:\s+del\s+video)?'
        
        # Combined pattern: timestamp (with optional prefix) OR seconds OR start/end keyword
        time_or_start = rf'(?:{ts_with_prefix}|{sec_with_prefix}|{start_keyword})'
        time_or_end = rf'(?:{ts_with_prefix}|{sec_with_prefix}|{end_keyword})'
        
        # For seconds ranges, also allow bare numbers after "al" or "hasta"
        sec_or_end = rf'(?:{ts_with_prefix}|{sec_with_prefix}|{sec_pattern}|{end_keyword})'
        
        # Track matched positions to avoid double-matching
        matched_positions = set()
        
        # === RANGE PATTERNS (process these first) ===
        # These handle "desde X hasta Y", "from X to Y", "entre X y Y", etc.
        # where X and Y can be timestamps, seconds, OR keywords like "inicio"/"final"
        
        def extract_two_times_from_text(text: str) -> Tuple[float, float]:
            """
            Extract two time values from a range text.
            Returns (start, end) tuple.
            Handles MM:SS timestamps, seconds as numbers, and keywords.
            """
            text_lower = text.lower()
            
            # Check for start/end keywords
            has_start_keyword = bool(re.search(r'(inicio|principio|comienzo|beginning|start)', text_lower))
            has_end_keyword = bool(re.search(r'(final|fin|end)(?:\s|$)', text_lower))
            
            # Find all MM:SS timestamps
            ts_matches = re.findall(r'\d{1,2}:\d{2}', text)
            
            # Find all bare numbers (for seconds) - exclude those in timestamps
            # First remove timestamps from text to find bare seconds
            text_without_ts = re.sub(r'\d{1,2}:\d{2}', '', text)
            sec_matches = re.findall(r'\b(\d+)\b', text_without_ts)
            
            # Determine start value
            start_val = 0.0
            if has_start_keyword:
                start_val = 0.0
            elif ts_matches:
                start_val = parse_timestamp(ts_matches[0])
            elif sec_matches:
                start_val = parse_seconds(sec_matches[0])
            
            # Determine end value
            end_val = default_end
            if has_end_keyword:
                end_val = default_end
            elif len(ts_matches) >= 2:
                end_val = parse_timestamp(ts_matches[1])
            elif len(ts_matches) == 1 and has_start_keyword:
                # "desde el inicio hasta 00:30" - the single timestamp is the end
                end_val = parse_timestamp(ts_matches[0])
            elif ts_matches and sec_matches:
                # Start was timestamp, end is seconds
                end_val = parse_seconds(sec_matches[0])
            elif len(sec_matches) >= 2:
                end_val = parse_seconds(sec_matches[1])
            elif len(sec_matches) == 1 and has_start_keyword:
                # "desde el inicio hasta el segundo 60" - single number is the end
                end_val = parse_seconds(sec_matches[0])
            
            return (start_val, end_val)
        
        range_patterns = [
            # "desde el segundo 25 al 42" or "del segundo 25 al segundo 42"
            rf'(?:desde|del?)\s+(?:el\s+)?segundos?\s+{sec_pattern}\s+(?:hasta|al?)\s+(?:(?:el\s+)?segundos?\s+)?{sec_or_end}',
            # "desde el inicio hasta el minuto 00:30" or "desde 00:25 hasta el final"
            rf'(?:desde|from|de)\s+{time_or_start}\s+(?:hasta|to|a)\s+{time_or_end}',
            # "entre los minutos X y Y" or "entre X y Y" or "between X and Y"
            rf'(?:entre|between)\s+(?:los\s+)?{time_or_start}\s+(?:y|and)\s+{time_or_end}',
            # "entre los segundos X y Y"
            rf'(?:entre|between)\s+(?:los\s+)?segundos?\s+{sec_pattern}\s+(?:y|and)\s+(?:(?:el\s+)?segundos?\s+)?{sec_or_end}',
            # "minuto X hasta Y" or "minuto X al Y"
            rf'(?:minutos?|el\s+minuto|min)\s*{ts_pattern}\s*(?:hasta|al?|to|-)\s*(?:el\s*)?(?:minutos?\s*)?{time_or_end}',
            # "X hasta Y" or "X to Y" (timestamps only, to avoid false matches)
            rf'{ts_pattern}\s*(?:hasta|to)\s*(?:el\s*)?(?:minutos?\s*)?{ts_pattern}',
            # Direct range with dashes: "01:23 - 02:45" or "1:23–2:45"
            rf'{ts_pattern}\s*[-–—]\s*{ts_pattern}',
        ]
        
        for pattern in range_patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                # Extract the full match text
                full_match = match.group(0)
                
                # Extract both time values from the full match text
                start, end = extract_two_times_from_text(full_match)
                    
                # Validate range
                if start <= end:
                    timestamps.append((start, end))
                    matched_positions.update(range(match.start(), match.end()))
                    logger.debug(f"Extracted range from '{full_match}': {start:.1f}s - {end:.1f}s")
        
        # === STANDALONE START/END KEYWORDS ===
        # Only match these if NOT already part of a range
        
        # Check for standalone "inicio" mentions (not part of a range)
        # Avoid matching idiomatic expressions like "al inicio del día", "al principio de la mañana", etc.
        start_standalone_patterns = [
            r'(?:al|en el)\s+(?:inicio|principio|comienzo)(?:\s+del\s+video)?(?!\s+(?:del?\s+)?(?:día|semana|mes|año|jornada|mañana|tarde|recorrido|paseo|viaje|evento|conversación|historia|relato))',
            r'(?:inicio|principio|comienzo)\s+del\s+video',
        ]
        
        for pattern in start_standalone_patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                if not any(pos in matched_positions for pos in range(match.start(), match.end())):
                    # Standalone "inicio" - create clip from 0 to 30s (or what makes sense)
                    timestamps.append((0, min(30, default_end)))
                    matched_positions.update(range(match.start(), match.end()))
                    logger.debug(f"Detected standalone 'inicio' keyword")
        
        # Check for standalone "final" mentions (not part of a range)
        # Avoid matching idiomatic expressions like "al final del día", "al final de la semana", etc.
        end_standalone_patterns = [
            r'(?:al|en el)\s+(?:final|fin)(?:\s+del\s+video)?(?!\s+(?:del?\s+)?(?:día|semana|mes|año|jornada|recorrido|paseo|viaje|evento|conversación|historia|relato))',
            r'(?:final|fin)\s+del\s+video',
        ]
        
        for pattern in end_standalone_patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                if not any(pos in matched_positions for pos in range(match.start(), match.end())):
                    # Standalone "final" - create clip from last 30s to end
                    end_start = max(0, default_end - 30)
                    timestamps.append((end_start, default_end))
                    matched_positions.update(range(match.start(), match.end()))
                    logger.debug(f"Detected standalone 'final' keyword: {end_start:.1f}s - {default_end:.1f}s")
        
        # === SINGLE TIMESTAMP PATTERNS ===
        # "at X", "en X", "minuto X", "segundo X", etc.
        
        single_patterns = [
            rf'(?:at|en|a las?|around|cerca de|minuto|timestamp:?)\s*{ts_pattern}',
            rf'\[{ts_pattern}\]',  # Timestamps in brackets
        ]
        
        # Also handle single second references like "segundo 25" or "en el segundo 30"
        single_sec_patterns = [
            rf'(?:en\s+)?(?:el\s+)?segundos?\s+{sec_pattern}(?!\s*(?:hasta|al?|y|and|to))',  # Avoid matching start of a range
        ]
        
        for pattern in single_patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                # Skip if this match overlaps with an already processed range
                if any(pos in matched_positions for pos in range(match.start(), match.end())):
                    continue
                ts = parse_timestamp(match.group(1))
                # For single timestamps, create a small range (±15 seconds for precision)
                start = max(0, ts - 15)
                end = min(ts + 15, default_end) if video_duration else ts + 15
                timestamps.append((start, end))
                matched_positions.update(range(match.start(), match.end()))
                logger.debug(f"Extracted single timestamp: {ts}s -> range {start:.1f}s - {end:.1f}s")
        
        for pattern in single_sec_patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                # Skip if this match overlaps with an already processed range
                if any(pos in matched_positions for pos in range(match.start(), match.end())):
                    continue
                ts = parse_seconds(match.group(1))
                # For single second references, create a small range (±15 seconds for precision)
                start = max(0, ts - 15)
                end = min(ts + 15, default_end) if video_duration else ts + 15
                timestamps.append((start, end))
                matched_positions.update(range(match.start(), match.end()))
                logger.debug(f"Extracted single second reference: {ts}s -> range {start:.1f}s - {end:.1f}s")
        
        # Deduplicate timestamps and sort chronologically by start time
        unique_timestamps = list(set(timestamps))
        unique_timestamps.sort(key=lambda x: (x[0], x[1]))  # Sort by start_time, then end_time
        
        if unique_timestamps:
            logger.info(f"Extracted {len(unique_timestamps)} timestamp ranges: {unique_timestamps}")
        
        return unique_timestamps
    
    @staticmethod
    def _create_clips_for_mentioned_timestamps(
        mentioned_timestamps: List[Tuple[float, float]],
        video_files_info: Dict,
        user_context: Dict
    ) -> List[Dict]:
        """
        Create video references for the exact timestamps mentioned by the agent.
        
        Args:
            mentioned_timestamps: List of (start_seconds, end_seconds) from the response
            video_files_info: Dict mapping file_id -> {filename, file_path}
            user_context: User context for URL signing
            
        Returns:
            List of video reference dicts for the cited timestamps
        """
        from utils.security import generate_signed_url
        
        if not mentioned_timestamps or not video_files_info:
            return []
        
        references = []
        username = str(user_context.get('user_id', 'anonymous')) if user_context else 'anonymous'
        
        # Helper function to format seconds as MM:SS
        def format_timestamp(seconds: float) -> str:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
        
        logger.info(f"Creating {len(mentioned_timestamps)} references for timestamps mentioned by agent")
        
        # Get the first video file
        file_id = None
        file_path = None
        filename = None
        
        for fid, finfo in video_files_info.items():
            file_id = fid
            file_path = finfo.get('file_path')
            filename = finfo.get('filename', 'video')
            break
        
        if not file_path:
            logger.warning("No video file path available")
            return []
        
        # Generate video URL for the full video
        video_url = generate_signed_url(file_path, username)
        
        for i, (start_time, end_time) in enumerate(mentioned_timestamps, 1):
            start_formatted = format_timestamp(start_time)
            end_formatted = format_timestamp(end_time)
            duration = end_time - start_time
            
            references.append({
                "file_id": file_id,
                "filename": filename,
                "start_time": start_time,
                "end_time": end_time,
                "start_formatted": start_formatted,
                "end_formatted": end_formatted,
                "text_preview": f"Segment mentioned by agent ({start_formatted} - {end_formatted}, {duration:.0f}s)",
                "video_url": video_url,
                "is_agent_cited": True
            })
        
        return references
    
    @staticmethod
    def _filter_references_by_mentioned_timestamps(
        video_references: List[Dict],
        mentioned_timestamps: List[Tuple[float, float]],
        tolerance_seconds: float = 15.0
    ) -> List[Dict]:
        """
        Filter video references to only include those that match timestamps
        mentioned in the agent's response.
        
        Args:
            video_references: List of video reference dicts with start_time/end_time
            mentioned_timestamps: List of (start, end) tuples from the response
            tolerance_seconds: How much overlap to require for a match
            
        Returns:
            Filtered list of video references that match mentioned timestamps
        """
        if not mentioned_timestamps:
            # If no timestamps were extracted, return all references
            # (agent didn't cite specific times)
            return video_references
        
        def overlaps(ref_start: float, ref_end: float, mentioned_start: float, mentioned_end: float) -> bool:
            """Check if two time ranges overlap within tolerance"""
            # Expand mentioned range by tolerance
            expanded_start = mentioned_start - tolerance_seconds
            expanded_end = mentioned_end + tolerance_seconds
            
            # Check for overlap
            return not (ref_end < expanded_start or ref_start > expanded_end)
        
        filtered = []
        for ref in video_references:
            ref_start = ref.get('start_time', 0)
            ref_end = ref.get('end_time', 0)
            
            # Check if this reference overlaps with any mentioned timestamp
            for mentioned_start, mentioned_end in mentioned_timestamps:
                if overlaps(ref_start, ref_end, mentioned_start, mentioned_end):
                    filtered.append(ref)
                    break  # Don't add same reference multiple times
        
        # Re-sort filtered references by start_time
        filtered.sort(key=lambda r: r.get('start_time', 0))
        
        logger.info(f"Filtered video references: {len(filtered)}/{len(video_references)} match mentioned timestamps")
        
        return filtered
    
    async def execute_agent_chat_with_file_refs(
        self, 
        agent_id: int, 
        message: str, 
        file_references: List = None,
        search_params: Dict = None,
        user_context: Dict = None,
        conversation_id: int = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Execute agent chat with persistent file references
        
        Args:
            agent_id: ID of the agent to execute
            message: User message
            file_references: List of FileReference objects from FileManagementService
            search_params: Optional search parameters for silo-based agents
            user_context: User context (api_key, user_id, etc.)
            conversation_id: Optional conversation ID to continue existing conversation
            
        Returns:
            Dict containing agent response and metadata
        """
        try:
            # Get agent
            agent = self.agent_service.get_agent(db, agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Validate user has access to this agent
            await self._validate_agent_access(agent, user_context)
            
            # Process file references to extract content
            processed_files = []
            video_files_to_process = []
            video_file_refs = []  
            temp_session_id = None
            video_files_info = {}  # Map file_id -> {filename, file_path} for URL generation
            
            logger.info(f"========== PROCESSING {len(file_references) if file_references else 0} FILE REFERENCES ==========")
            
            if file_references:
                for file_ref in file_references:
                    logger.info(f"  - File: {file_ref.filename}, type: {file_ref.file_type}")
                    logger.info(f"    content starts with: {file_ref.content[:80] if file_ref.content else 'None'}...")
                    logger.info(f"    temp_session_id: {file_ref.temp_session_id}, temp_media_id: {file_ref.temp_media_id}")
                    
                    # Check if this is a video/audio pending processing
                    if file_ref.file_type in ("video", "audio") and file_ref.content.startswith("__VIDEO_PENDING_PROCESSING__:"):
                        video_path = file_ref.content.replace("__VIDEO_PENDING_PROCESSING__:", "")
                        video_files_to_process.append({
                            "path": video_path,
                            "filename": file_ref.filename,
                            "file_id": file_ref.file_id
                        })
                        video_file_refs.append(file_ref)
                        # Store video info for URL generation
                        video_files_info[file_ref.file_id] = {
                            "filename": file_ref.filename,
                            "file_path": file_ref.file_path
                        }
                    elif file_ref.file_type in ("video", "audio") and file_ref.temp_session_id:
                        # Video/audio already processed - use existing temp_session_id for context search
                        # Don't add to processed_files since the text content is not useful
                        if not temp_session_id:
                            temp_session_id = file_ref.temp_session_id
                            logger.info(f"Found existing temp_session_id from processed video/audio: {temp_session_id}")
                            logger.info(f"  file_id={file_ref.file_id}, filename={file_ref.filename}")
                        # Store video info for URL generation
                        video_files_info[file_ref.file_id] = {
                            "filename": file_ref.filename,
                            "file_path": file_ref.file_path
                        }
                    else:
                        processed_files.append({
                            "filename": file_ref.filename,
                            "content": file_ref.content,
                            "type": file_ref.file_type,
                            "file_id": file_ref.file_id,
                            "file_path": file_ref.file_path
                        })
            
            # Process video files for temporary indexing
            if video_files_to_process:
                logger.info(f"========== STARTING VIDEO PROCESSING: {len(video_files_to_process)} video(s) ==========")
                temp_session_id = await self._process_videos_for_temp_context(
                    video_files_to_process,
                    video_file_refs,
                    agent,
                    user_context,
                    conversation_id,
                    db
                )
            
            # Get or create conversation for memory-enabled agents
            session = None
            conversation = None
            if agent.has_memory:
                from services.conversation_service import ConversationService
                
                # If conversation_id provided, validate and use it
                if conversation_id:
                    conversation = ConversationService.get_conversation(
                        db=db,
                        conversation_id=conversation_id,
                        user_context=user_context,
                        agent_id=agent_id
                    )
                    if not conversation:
                        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
                    
                    # Extract session_id from conversation (without "conv_{agent_id}_" prefix)
                    session_suffix = conversation.session_id.replace(f"conv_{agent_id}_", "")
                    session = await self.session_service.get_user_session(
                        agent_id=agent_id,
                        user_context=user_context,
                        conversation_id=session_suffix
                    )
                else:
                    # Auto-create a conversation if none exists
                    conversation = ConversationService.create_conversation(
                        db=db,
                        agent_id=agent_id,
                        user_context=user_context,
                        title=None  # Auto-generate title
                    )
                    logger.info(f"Auto-created conversation {conversation.conversation_id} for agent {agent_id}")
                    
                    # Extract session_id from conversation (without "conv_{agent_id}_" prefix)
                    session_suffix = conversation.session_id.replace(f"conv_{agent_id}_", "")
                    session = await self.session_service.get_user_session(
                        agent_id=agent_id,
                        user_context=user_context,
                        conversation_id=session_suffix
                    )
            
            # Execute agent using LangChain IN A THREAD POOL (blocking LLM calls)
            # This prevents blocking the event loop and allows other requests to be processed
            
            # First, search video context (quick, non-blocking) to capture references
            video_references = []
            video_context_message = ""
            
            logger.info(f"execute_playground_agent - Video context search:")
            logger.info(f"  temp_session_id={temp_session_id}")
            logger.info(f"  video_files_info={video_files_info}")
            
            if temp_session_id and video_files_info:
                video_refs_result = self._search_video_context(
                    temp_session_id=temp_session_id,
                    message=message,
                    video_files_info=video_files_info,
                    user_context=user_context,
                    agent=agent,
                    db=db
                )
                video_context_message = video_refs_result.get("context_message", "")
                video_references = video_refs_result.get("references", [])
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                self._execute_langchain_agent,
                agent, message, processed_files, search_params, session, user_context, db, temp_session_id, video_context_message
            )
            
            # Parse response based on agent's output parser
            from tools.agentTools import parse_agent_response
            parsed_response = parse_agent_response(response, agent)
            
            # Update request count
            self._update_request_count(agent, db)
            
            # Update session timestamp to keep it alive
            if session:
                await self.session_service.touch_session(session.id)
            
            # Update conversation message count if using a specific conversation
            if conversation:
                from services.conversation_service import ConversationService
                # Get last message preview (truncate response if too long)
                last_message_preview = parsed_response[:200] if isinstance(parsed_response, str) else str(parsed_response)[:200]
                
                # Clean message for preview if it's a list (multimodal)
                # This ensures the conversation list shows clean text instead of JSON structure
                if isinstance(parsed_response, list):
                    try:
                        text_parts = []
                        for item in parsed_response:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        if text_parts:
                            last_message_preview = " ".join(text_parts)[:200]
                    except Exception:
                        pass
                
                # Increment by 2 (user message + agent response)
                ConversationService.increment_message_count(
                    db=db,
                    conversation_id=conversation.conversation_id,
                    last_message=last_message_preview,
                    increment_by=2
                )
            
            # Filter video references to only include segments mentioned in the response
            # OR create new clips from exact timestamps mentioned by the agent
            if video_files_info and parsed_response:
                # Get text content from response (handle both string and list formats)
                response_text = ""
                if isinstance(parsed_response, str):
                    response_text = parsed_response
                elif isinstance(parsed_response, list):
                    for item in parsed_response:
                        if isinstance(item, dict) and item.get("type") == "text":
                            response_text += item.get("text", "") + " "
                
                # Get video duration for resolving "final" keyword
                video_duration = None
                for file_id, file_info in video_files_info.items():
                    if file_info.get('file_path'):
                        video_duration = self._get_video_duration(file_info['file_path'])
                        break  # Use first video
                
                # Extract timestamps mentioned in the response
                mentioned_timestamps = self._extract_timestamps_from_response(response_text, video_duration)
                
                if mentioned_timestamps:
                    logger.info(f"Found {len(mentioned_timestamps)} timestamp references in response: {mentioned_timestamps}")
                    
                    # Create clips for the EXACT timestamps mentioned by the agent
                    # This merges overlapping ranges and creates precise clips
                    cited_clips = self._create_clips_for_mentioned_timestamps(
                        mentioned_timestamps=mentioned_timestamps,
                        video_files_info=video_files_info,
                        user_context=user_context
                    )
                    
                    if cited_clips:
                        # Replace video_references with the agent-cited clips
                        video_references = cited_clips
                        logger.info(f"Using {len(cited_clips)} clips from agent-cited timestamps")
                    elif video_references:
                        # Fallback: filter existing references by timestamps
                        video_references = self._filter_references_by_mentioned_timestamps(
                            video_references, mentioned_timestamps
                        )
            
            return {
                "response": parsed_response,
                "agent_id": agent_id,
                "conversation_id": conversation.conversation_id if conversation else None,
                "metadata": {
                    "agent_name": agent.name,
                    "agent_type": agent.type,
                    "files_processed": len(processed_files),
                    "has_memory": agent.has_memory,
                    "temp_video_session": temp_session_id
                },
                "video_references": video_references if video_references else None
            }
            
        except Exception as e:
            logger.error(f"Error executing agent chat: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    async def _process_videos_for_temp_context(
        self,
        video_files: List[Dict],
        video_file_refs: List,
        agent: Agent,
        user_context: Dict,
        conversation_id: int,
        db: Session
    ) -> str:
        """
        Process video files and index them in a temporary collection for session-based context.
        
        Args:
            video_files: List of dicts with 'path', 'filename', 'file_id'
            video_file_refs: List of FileReference objects corresponding to video_files
            agent: Agent instance (for embedding service access)
            user_context: User context
            conversation_id: Conversation ID for session identification
            db: Database session
            
        Returns:
            Temporary session ID used for the collection
        """
        from services.temporary_media_service import TemporaryMediaService
        from services.file_management_service import FileManagementService
        
        # Generate a unique session ID based on agent, user, and conversation
        user_id = user_context.get('user_id', 'anonymous') if user_context else 'anonymous'
        session_key = f"{agent.agent_id}_{user_id}_{conversation_id or 'no_conv'}"
        temp_session_id = hashlib.sha256(session_key.encode()).hexdigest()[:16]
        
        # Get embedding service - first try agent's silo, then fallback to any app-level embedding service
        embedding_service = None
        ai_service_id = None
        
        # Get fresh agent with relationships
        fresh_agent = self.agent_execution_repo.get_agent_with_relationships(db, agent.agent_id)
        
        if fresh_agent and fresh_agent.silo and fresh_agent.silo.embedding_service:
            embedding_service = fresh_agent.silo.embedding_service
            logger.info(f"Using embedding service from agent's silo: {embedding_service.name}")
        else:
            # Fallback: use any available embedding service from the app
            logger.info(f"Agent {agent.agent_id} has no silo - looking for app-level embedding service")
            from repositories.embedding_service_repository import EmbeddingServiceRepository
            app_embedding_services = EmbeddingServiceRepository.get_by_app_id(db, agent.app_id)
            
            if app_embedding_services:
                embedding_service = app_embedding_services[0]  # Use first available
                logger.info(f"Using fallback app-level embedding service: {embedding_service.name}")
            else:
                logger.warning(f"No embedding service available for app {agent.app_id} - video processing cannot continue")
                return None
        
        # Get AI service for transcription (look for OpenAI service with Whisper support)
        from repositories.ai_service_repository import AIServiceRepository
        ai_services = AIServiceRepository.get_by_app_id(db, agent.app_id)
        
        for service in ai_services:
            if service.provider == 'OpenAI':
                ai_service_id = service.service_id
                break
        
        if not ai_service_id:
            logger.error(f"No OpenAI AI service found for transcription in app {agent.app_id}")
            return None
        
        # Process each video file
        temp_service = TemporaryMediaService()
        file_service = FileManagementService()
        
        for i, video_data in enumerate(video_files):
            try:
                logger.info(f"========== PROCESSING VIDEO FILE: {video_data['filename']} ==========")
                
                # Use file_id as media_id for tracking
                media_id = video_data['file_id'][:8]  # Use first 8 chars of file_id
                
                result = await temp_service.process_video_for_session(
                    video_path=video_data['path'],
                    session_id=temp_session_id,
                    embedding_service=embedding_service,
                    ai_service_id=ai_service_id,
                    db=db,
                    filename=video_data['filename'],
                    media_id=media_id,  # Pass explicit media_id
                    chunk_min_duration=30,
                    chunk_max_duration=120,
                    chunk_overlap=5
                )
                
                if result['success']:
                    logger.info(f"========== VIDEO TRANSCRIPTION SUCCESS: {result['chunk_count']} chunks, language: {result.get('language', 'unknown')} ==========")
                    logger.info(f"Full transcript preview: {result.get('full_transcript', '')[:200]}...")
                    
                    # Update the FileReference with temp_session_id and temp_media_id
                    if i < len(video_file_refs):
                        file_ref = video_file_refs[i]
                        file_ref.temp_session_id = temp_session_id
                        file_ref.temp_media_id = result['media_id']
                        # Mark as processed (no longer pending)
                        file_ref.content = f"[Video processed: {video_data['filename']}]"
                        
                        # Update the persistent file metadata on disk
                        await file_service.update_file_metadata(
                            agent_id=agent.agent_id,
                            file_id=file_ref.file_id,
                            user_context=user_context,
                            conversation_id=str(conversation_id) if conversation_id else None,
                            updates={
                                'temp_session_id': temp_session_id,
                                'temp_media_id': result['media_id'],
                                'content': f"[Video processed: {video_data['filename']}]"
                            }
                        )
                else:
                    logger.warning(f"Failed to process video {video_data['filename']}")
                    
            except Exception as e:
                logger.error(f"Error processing video {video_data['filename']}: {str(e)}")
                continue
        
        return temp_session_id

    def _search_video_context(
        self,
        temp_session_id: str,
        message: str,
        video_files_info: Dict,
        user_context: Dict,
        agent: Agent,
        db: Session
    ) -> Dict[str, Any]:
        """
        Search video context and return both the context message for the LLM
        and the video references for the frontend player.
        
        Args:
            temp_session_id: Temporary session ID for the collection
            message: User's query message
            video_files_info: Dict mapping file_id -> {filename, file_path}
            user_context: User context for URL signing
            agent: Agent instance
            db: Database session
            
        Returns:
            Dict with 'context_message' (str) and 'references' (list of VideoReference dicts)
        """
        from services.temporary_media_service import TemporaryMediaService
        from repositories.embedding_service_repository import EmbeddingServiceRepository
        from utils.security import generate_signed_url
        
        result = {"context_message": "", "references": []}
        
        # Get embedding service for search
        embedding_service_for_search = None
        fresh_agent = self.agent_execution_repo.get_agent_with_relationships(db, agent.agent_id)
        
        if fresh_agent and fresh_agent.silo and fresh_agent.silo.embedding_service:
            embedding_service_for_search = fresh_agent.silo.embedding_service
        else:
            app_embedding_services = EmbeddingServiceRepository.get_by_app_id(db, agent.app_id)
            if app_embedding_services:
                embedding_service_for_search = app_embedding_services[0]
        
        if not embedding_service_for_search:
            logger.warning(f"No embedding service available for video context search")
            return result
        
        try:
            if not TemporaryMediaService.temp_collection_exists(temp_session_id):
                logger.warning(f"Temp collection {temp_session_id} does not exist")
                return result
            
            temp_results = TemporaryMediaService.search_temp_collection(
                session_id=temp_session_id,
                query=message,
                embedding_service=embedding_service_for_search,
                k=10  # Request more to account for potential duplicates
            )
            
            if not temp_results:
                return result
            
            logger.info(f"Found {len(temp_results)} video context segments")
            
            # Filter by relevance score - lower _score = more similar (distance-based)
            # STRICT filtering for video clips shown to user (not for LLM context)
            MIN_RELEVANCE_THRESHOLD = 0.4  # If best score > this, don't show video clips to user
            RELEVANCE_SCORE_THRESHOLD = 0.5  # Maximum distance to consider a segment relevant
            RELATIVE_SCORE_FACTOR = 1.3  # Results must be within 1.3x the best score
            
            # Get the best score (lowest distance = most similar)
            best_score = min(doc.metadata.get('_score', 999) for doc in temp_results)
            
            # Determine if we should show video references to user
            # (question must be specifically about video content)
            show_video_references = best_score <= MIN_RELEVANCE_THRESHOLD
            
            if not show_video_references:
                logger.info(f"Question not specifically about video content (best_score={best_score:.3f} > {MIN_RELEVANCE_THRESHOLD}). Will still provide context to LLM but skip video references for user.")
            
            max_allowed_score = min(best_score * RELATIVE_SCORE_FACTOR, RELEVANCE_SCORE_THRESHOLD)
            
            # Filter to only highly relevant results for processing
            relevant_results = [
                doc for doc in temp_results 
                if doc.metadata.get('_score', 999) <= max_allowed_score
            ]
            
            # If no relevant results after filtering, use top 3 anyway for LLM context
            if not relevant_results:
                relevant_results = temp_results[:3]
            
            logger.info(f"Filtered to {len(relevant_results)} relevant segments (best_score={best_score:.3f}, max_allowed={max_allowed_score:.3f})")
            
            # Helper function to format seconds as MM:SS
            def format_timestamp(seconds: float) -> str:
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins:02d}:{secs:02d}"
            
            # Get username for URL signing
            username = str(user_context.get('user_id', 'anonymous')) if user_context else 'anonymous'
            
            # Build context message for LLM
            video_context = "\n\n[VIDEO CONTEXT - Transcription segments from attached video]\n"
            video_context += "IMPORTANT: When answering based on this video content, always cite the timestamp (e.g., 'At 01:23...' or 'From minute 2:15 to 2:45...').\n\n"
            
            # Build video references for frontend - deduplicate by (filename, start_time)
            references = []
            seen_segments = set()  # Track (filename, start_time) to avoid duplicates
            rank = 0
            max_segments = 5  # Maximum unique segments to return
            
            for i, doc in enumerate(relevant_results, 1):
                start_time = doc.metadata.get('start_time', 0)
                end_time = doc.metadata.get('end_time', 0)
                filename = doc.metadata.get('filename', 'video')
                media_id = doc.metadata.get('media_id', '')
                text_content = doc.page_content.strip()
                relevance_score = doc.metadata.get('_score', 0)
                
                # Skip duplicate segments (same file and start time)
                segment_key = (filename.lower().strip(), round(float(start_time), 1))
                if segment_key in seen_segments:
                    logger.debug(f"Skipping duplicate segment: {filename} at {start_time}s")
                    continue
                seen_segments.add(segment_key)
                rank += 1
                
                start_formatted = format_timestamp(start_time)
                end_formatted = format_timestamp(end_time)
                
                # Add to LLM context
                video_context += f"[Segment {rank} | {filename} | Timestamp: {start_formatted} - {end_formatted}]\n"
                video_context += text_content
                video_context += "\n\n"
                
                # Find the file_path for this video using the video_files_info
                # We match by filename since media_id might not directly map
                video_url = ""
                file_id = ""
                
                logger.debug(f"Looking for video file matching filename: {filename}")
                logger.debug(f"Available video_files_info: {video_files_info}")
                
                for fid, finfo in video_files_info.items():
                    finfo_filename = finfo.get('filename', '')
                    # Compare normalized filenames (case-insensitive, strip whitespace)
                    if finfo_filename.lower().strip() == filename.lower().strip():
                        file_id = fid
                        file_path = finfo.get('file_path')
                        if file_path:
                            video_url = generate_signed_url(file_path, username)
                            logger.info(f"Generated video URL for {filename}: {video_url[:80]}...")
                        else:
                            logger.warning(f"Video file {filename} has no file_path in video_files_info")
                        break
                else:
                    logger.warning(f"No matching video file found for filename: {filename}")
                
                # Create video reference for frontend (only if we'll show references)
                if show_video_references:
                    references.append({
                        "file_id": file_id,
                        "filename": filename,
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "start_formatted": start_formatted,
                        "end_formatted": end_formatted,
                        "text_preview": text_content[:150] + "..." if len(text_content) > 150 else text_content,
                        "video_url": video_url,
                    })
                
                # Stop if we have enough unique segments
                if rank >= max_segments:
                    break
            
            video_context += "[END VIDEO CONTEXT]\n"
            
            # Always set context_message for LLM
            result["context_message"] = video_context
            
            # Only include video references if question is specifically about video
            if show_video_references and references:
                # Sort references by start_time for chronological ordering
                references.sort(key=lambda r: r["start_time"])
                
                result["references"] = references
                logger.info(f"Created {len(references)} video references with URLs (sorted by timestamp)")
            else:
                logger.info(f"Video context provided to LLM but no references shown to user (show_video_references={show_video_references})")
            
        except Exception as e:
            logger.error(f"Error searching video context: {str(e)}")
        
        return result

    async def execute_agent_chat(
        self, 
        agent_id: int, 
        message: str, 
        files: List[UploadFile] = None,
        search_params: Dict = None,
        user_context: Dict = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Execute agent chat - used by both playground and public API
        
        Args:
            agent_id: ID of the agent to execute
            message: User message
            files: Optional file attachments
            search_params: Optional search parameters for silo-based agents
            user_context: User context (api_key, user_id, etc.)
            
        Returns:
            Dict containing agent response and metadata
        """
        try:
            # Get agent
            agent = self.agent_service.get_agent(db, agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Validate user has access to this agent
            await self._validate_agent_access(agent, user_context)
            
            # Process files if provided
            processed_files = []
            if files:
                processed_files = await self._process_files_for_agent(files, agent)
            
            # Get user session for memory-enabled agents
            session = None
            if agent.has_memory:
                # Extract conversation_id from user_context to ensure correct session identification
                conversation_id = user_context.get("conversation_id") if user_context else None
                session = await self.session_service.get_user_session(agent_id, user_context, conversation_id)
            
            # Execute agent using LangChain IN A THREAD POOL (blocking LLM calls)
            # This prevents blocking the event loop and allows other requests to be processed
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                self._execute_langchain_agent,
                agent, message, processed_files, search_params, session, user_context, db, None
            )
            
            # Parse response based on agent's output parser
            from tools.agentTools import parse_agent_response
            parsed_response = parse_agent_response(response, agent)
            
            # Update request count
            self._update_request_count(agent, db)
            
            # Update session timestamp to keep it alive
            if session:
                await self.session_service.touch_session(session.id)
            
            return {
                "response": parsed_response,
                "agent_id": agent_id,
                "metadata": {
                    "agent_name": agent.name,
                    "agent_type": agent.type,
                    "files_processed": len(processed_files),
                    "has_memory": agent.has_memory
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing agent chat: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
    
    async def execute_agent_ocr(
        self, 
        agent_id: int, 
        pdf_file: UploadFile,
        user_context: Dict = None,
        for_api: bool = False,  # True for public API, False for playground
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Execute OCR processing - used by both playground and public API
        
        Args:
            agent_id: ID of the OCR agent
            pdf_file: PDF file to process
            user_context: User context (api_key, user_id, etc.)
            
        Returns:
            Dict containing OCR processing results
        """
        try:
            # Get OCR agent
            agent = self.agent_service.get_agent(db, agent_id, agent_type='ocr_agent')
            if not agent or not isinstance(agent, OCRAgent):
                raise HTTPException(status_code=404, detail="OCR Agent not found")
            
            # Validate user has access to this agent
            await self._validate_agent_access(agent, user_context)
            
            # Validate PDF file
            if not pdf_file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            # Save PDF to temporary location
            temp_pdf_path = await self._save_uploaded_file(pdf_file)
            
            try:
                # Process PDF using existing tools
                result = await self._process_pdf_with_ocr(agent, temp_pdf_path, db)
                
                # Update request count
                self._update_request_count(agent, db)
                
                if for_api:
                    # Public API: Return just the structured content (output parser result)
                    return result.get("content", result)
                else:
                    # Playground: Return full result with metadata for UI
                    content = result.get("content", "")
                    if isinstance(content, dict):
                        import json
                        extracted_text = json.dumps(content, indent=2, ensure_ascii=False)
                    else:
                        extracted_text = str(content)
                    
                    return {
                        "result": result,
                        "agent_id": agent_id,
                        "extracted_text": extracted_text,
                        "metadata": {
                            "agent_name": agent.name,
                            "pdf_filename": pdf_file.filename,
                            "pages_processed": len(result.get("pages", [])),
                            "confidence": result.get("confidence", 0.0)
                        }
                    }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                    
        except Exception as e:
            logger.error(f"Error executing OCR agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
    async def reset_agent_conversation(
        self, 
        agent_id: int,
        user_context: Dict = None,
        db: Session = None
    ) -> bool:
        """
        Reset conversation - used by both playground and public API
        
        Args:
            agent_id: ID of the agent
            user_context: User context (api_key, user_id, etc.)
            
        Returns:
            True if reset successful
        """
        try:
            # Get agent
            agent = self.agent_service.get_agent(db, agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Validate user has access to this agent
            await self._validate_agent_access(agent, user_context)
            
            # Extract conversation_id from user_context if present
            conversation_id_str = user_context.get("conversation_id") if user_context else None
            conversation_id = int(conversation_id_str) if conversation_id_str else None
            
            # Get conversation from DB to find the actual session_id
            # This is critical because the session uses conversation.session_id (e.g. conv_1_UUID)
            # not the numeric conversation_id
            session_suffix = None
            if conversation_id:
                from services.conversation_service import ConversationService
                conversation = ConversationService.get_conversation(
                    db=db,
                    conversation_id=conversation_id,
                    user_context=user_context,
                    agent_id=agent_id
                )
                if conversation and conversation.session_id:
                    # Extract session suffix (UUID part after "conv_{agent_id}_")
                    session_suffix = conversation.session_id.replace(f"conv_{agent_id}_", "")
                    logger.info(f"Reset - Found conversation {conversation_id} with session_id={conversation.session_id}")
                    logger.info(f"  session_suffix: {session_suffix}")
            
            # Reset session if memory enabled
            if agent.has_memory:
                from services.agent_cache_service import CheckpointerCacheService
                
                # Get the session using the correct session_suffix (from conversation.session_id)
                session = await self.session_service.get_user_session(
                    agent_id, 
                    user_context, 
                    session_suffix  # Use extracted suffix, not the numeric conversation_id
                )
                if session:
                    # Invalidate the checkpointer for this specific session (use async version)
                    await CheckpointerCacheService.invalidate_checkpointer_async(agent_id, session.id)
                    logger.info(f"Invalidated checkpointer for agent {agent_id}, session {session.id}")
                
                # Reset the session object (clears messages and memory)
                # Put session_suffix in user_context so reset_user_session uses correct ID
                reset_context = {**user_context, "conversation_id": session_suffix} if session_suffix else user_context
                await self.session_service.reset_user_session(agent_id, reset_context)
            
            # Clear all attached files for this user/agent session
            from services.file_management_service import FileManagementService
            file_service = FileManagementService()
            
            # Extract conversation_id for file cleanup (should match the one used when files were uploaded)
            conv_id_for_files = user_context.get("conversation_id") if user_context else None
            logger.info(f"Reset - Looking for attached files:")
            logger.info(f"  agent_id={agent_id}, user_id={user_context.get('user_id')}, conversation_id={conv_id_for_files}")
            
            # Get all attached files for this session (pass conversation_id explicitly)
            attached_files = await file_service.list_attached_files(
                agent_id, 
                user_context, 
                conversation_id=conv_id_for_files
            )
            
            # Remove each file (pass conversation_id explicitly)
            for file_data in attached_files:
                try:
                    await file_service.remove_file(
                        file_id=file_data['file_id'],
                        agent_id=agent_id,
                        user_context=user_context,
                        conversation_id=conv_id_for_files
                    )
                    logger.info(f"Removed file {file_data['filename']} during conversation reset")
                except Exception as e:
                    logger.error(f"Error removing file {file_data['file_id']} during reset: {str(e)}")
            
            # Clean up any temporary video collections for this session
            # This must happen regardless of whether agent has memory
            # (video chunks are stored separately from conversation memory)
            try:
                from services.temporary_media_service import TemporaryMediaService
                
                user_id = user_context.get('user_id', 'anonymous') if user_context else 'anonymous'
                conversation_id = user_context.get("conversation_id")
                session_key = f"{agent_id}_{user_id}_{conversation_id or 'no_conv'}"
                temp_session_id = hashlib.sha256(session_key.encode()).hexdigest()[:16]
                
                logger.info(f"Reset - Looking for temp video collection")
                logger.info(f"  agent_id={agent_id}, user_id={user_id}, conversation_id={conversation_id}")
                logger.info(f"  session_key={session_key}")
                logger.info(f"  temp_session_id={temp_session_id}")
                logger.info(f"  full collection name: temp_session_{temp_session_id}")
                
                # Get embedding service - first from silo, then fallback to app-level
                embedding_service = None
                fresh_agent = self.agent_execution_repo.get_agent_with_relationships(db, agent_id)
                if fresh_agent:
                    if fresh_agent.silo and fresh_agent.silo.embedding_service:
                        embedding_service = fresh_agent.silo.embedding_service
                    else:
                        # Fallback: use any available embedding service from the app
                        from repositories.embedding_service_repository import EmbeddingServiceRepository
                        app_embedding_services = EmbeddingServiceRepository.get_by_app_id(db, fresh_agent.app_id)
                        if app_embedding_services:
                            embedding_service = app_embedding_services[0]
                
                exists = TemporaryMediaService.temp_collection_exists(temp_session_id)
                logger.info(f"  Collection exists: {exists}")
                
                if exists:
                    success = TemporaryMediaService.cleanup_session(temp_session_id, embedding_service)
                    logger.info(f"  Cleanup result: {'SUCCESS' if success else 'FAILED'}")
                else:
                    logger.info(f"  No temp video collection found for session {temp_session_id}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up video collection: {str(cleanup_error)}")
            
            logger.info(f"Conversation reset for agent {agent_id} - cleared {len(attached_files)} files")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting agent conversation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
    
    async def get_conversation_history(
        self,
        agent_id: int,
        user_context: Dict = None,
        db: Session = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history - used by playground to load existing conversation
        
        Args:
            agent_id: ID of the agent
            user_context: User context (api_key, user_id, etc.)
            
        Returns:
            List of messages with role and content
        """
        try:
            # Get agent
            agent = self.agent_service.get_agent(db, agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Validate user has access to this agent
            await self._validate_agent_access(agent, user_context)
            
            # Get conversation history if memory enabled
            if agent.has_memory:
                # Get the session to find the session_id
                session = await self.session_service.get_user_session(agent_id, user_context)
                if session:
                    # Get history from checkpointer
                    from services.agent_cache_service import CheckpointerCacheService
                    history = await CheckpointerCacheService.get_conversation_history_async(agent_id, session.id)
                    logger.info(f"Retrieved {len(history)} messages for agent {agent_id}, session {session.id}")
                    
                    # Clean history for frontend display (handle multimodal content)
                    cleaned_history = []
                    for msg in history:
                        if not isinstance(msg, dict):
                            continue
                            
                        content = msg.get("content")
                        parsed_content = content

                        # Some backends store the content as a string representation of the list
                        if isinstance(content, str):
                            stripped_content = content.strip()
                            if stripped_content.startswith("[") and "type" in stripped_content:
                                try:
                                    parsed_content = json.loads(stripped_content)
                                except json.JSONDecodeError:
                                    try:
                                        parsed_content = ast.literal_eval(stripped_content)
                                    except (ValueError, SyntaxError):
                                        parsed_content = content
                        
                        # If content is a list (multimodal structure), extract the text for display
                        if isinstance(parsed_content, list):
                            text_parts = []
                            has_image = False
                            for item in parsed_content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))
                                    elif item.get("type") == "image_url":
                                        has_image = True
                            
                            display_text = " ".join(text_parts)
                            # If we have an image but no text (or just whitespace), add a placeholder
                            if not display_text.strip() and has_image:
                                display_text = "[Imagen adjunta]"
                                
                            # Create a copy to avoid modifying the original cache
                            clean_msg = msg.copy()
                            clean_msg["content"] = display_text
                            cleaned_history.append(clean_msg)
                        else:
                            cleaned_history.append(msg)
                            
                    return cleaned_history
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    async def _validate_agent_access(self, agent: Agent, user_context: Dict):
        """Validate user has access to the agent"""
        # TODO: Implement proper access validation
        # For now, just log the validation
        logger.info(f"Validating access for agent {agent.agent_id} with context {user_context}")
        pass
    
    async def _process_files_for_agent(self, files: List[UploadFile], agent: Agent) -> List[Dict]:
        """Process files for agent consumption using existing PDF tools"""
        processed_files = []
        
        for file in files:
            try:
                # Save file temporarily
                temp_path = await self._save_uploaded_file(file)
                
                # Process based on file type IN THREAD POOL (blocking I/O)
                loop = asyncio.get_event_loop()
                file_data = await loop.run_in_executor(
                    self._executor,
                    self._process_single_file,
                    temp_path, file.filename
                )
                
                if file_data:
                    processed_files.append(file_data)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        return processed_files
    
    def _process_single_file(self, temp_path: str, filename: str) -> Dict:
        """Synchronous file processing - called in thread pool"""
        try:
            if filename.lower().endswith('.pdf'):
                # Use existing PDF tools (blocking I/O)
                text_content = extract_text_from_pdf(temp_path)
                return {
                    "filename": filename,
                    "content": text_content,
                    "type": "pdf"
                }
            else:
                # Handle other file types (blocking I/O)
                with open(temp_path, 'r') as f:
                    content = f.read()
                return {
                    "filename": filename,
                    "content": content,
                    "type": "text"
                }
        except Exception as e:
            logger.error(f"Error in _process_single_file: {str(e)}")
            return None
    
    async def _process_pdf_with_ocr(self, agent: OCRAgent, pdf_path: str, db: Session) -> Dict[str, Any]:
        """Process PDF using OCR workflow respecting output parser/data structure"""
        # Run all OCR processing in thread pool (blocking operations: PDF parsing, LLM calls, etc.)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._process_pdf_with_ocr_sync,
            agent, pdf_path, db
        )
        return result
    
    def _process_pdf_with_ocr_sync(self, agent: OCRAgent, pdf_path: str, db: Session) -> Dict[str, Any]:
        """Synchronous OCR processing - called in thread pool"""
        try:
            # Re-load agent with all relationships
            agent = self.agent_execution_repo.get_ocr_agent_with_relationships(db, agent.agent_id)
            
            if not agent:
                raise Exception(f"Agent {agent.agent_id} not found")
            
            # Get output parser if configured - this is CRITICAL for structured output
            pydantic_class = None
            if agent.output_parser_id:
                try:
                    # Get the output parser definition using repository
                    output_parser = self.agent_execution_repo.get_output_parser_by_id(db, agent.output_parser_id)
                    
                    if output_parser and output_parser.fields:
                        logger.info(f"Found output parser: {output_parser.name} with fields: {output_parser.fields}")
                        # Create pydantic model from schema like in original
                        pydantic_class = create_model_from_json_schema(
                            output_parser.fields,
                            output_parser.name
                        )
                        logger.info(f"Output parser model created successfully: {output_parser.name} -> {pydantic_class}")
                    else:
                        logger.warning(f"Output parser {agent.output_parser_id} not found or has no fields")
                    
                except Exception as e:
                    logger.warning(f"Failed to load output parser: {str(e)}")
            
            # Check if PDF has text
            has_text = check_pdf_has_text(pdf_path)
            
            if has_text:
                # Extract text directly
                text_content = extract_text_from_pdf(pdf_path)
                logger.info(f"Extracted text from PDF: {len(text_content)} characters")
                
                # Process with text model and output parser if available
                if agent.text_system_prompt and agent.service_id and pydantic_class:
                    try:
                        text_model = get_llm(agent, is_vision=False)
                        if text_model:
                            logger.info(f"Processing text with LLM and output parser")
                            # Use the output parser to structure the data
                            structured_data = get_data_from_extracted_text(
                                text_content,
                                text_model,
                                pydantic_class,
                                agent.text_system_prompt,
                                text_content,
                                os.path.basename(pdf_path)
                            )
                            
                            logger.info(f"Structured data result: {structured_data}")
                            
                            return {
                                "method": "text_extraction_with_llm",
                                "content": structured_data,
                                "extracted_text": text_content,
                                "confidence": 0.9
                            }
                    except Exception as e:
                        logger.error(f"Error processing with LLM and output parser: {str(e)}", exc_info=True)
                
                # If no text model or output parser, return raw text
                logger.info("No text model or output parser configured, returning raw text")
                return {
                    "method": "text_extraction",
                    "content": text_content,
                    "extracted_text": text_content,
                    "confidence": 0.9
                }
            else:
                # Convert to images and process with vision
                app_config = get_app_config()
                images_dir = app_config['IMAGES_PATH']
                os.makedirs(images_dir, exist_ok=True)
                
                image_paths = convert_pdf_to_images(pdf_path, images_dir)
                logger.info(f"Converted PDF to {len(image_paths)} images")
                
                # Process images with vision model
                vision_results = []
                for i, image_path in enumerate(image_paths):
                    try:
                        base64_image = convert_image_to_base64(image_path)
                        
                        # Get vision model
                        vision_model = get_llm(agent, is_vision=True)
                        if not vision_model:
                            raise Exception("Vision model not found")
                        
                        # Extract text from image
                        vision_result = extract_text_from_image(
                            base64_image, 
                            agent.vision_system_prompt, 
                            vision_model, 
                            f"Page {i+1}"
                        )
                        vision_results.append({
                            "page": i + 1,
                            "extracted_text": vision_result
                        })
                        
                        # Clean up image file
                        try:
                            os.remove(image_path)
                        except (OSError, FileNotFoundError):
                            pass
                            
                    except Exception as e:
                        logger.warning(f"Error processing image {i+1}: {str(e)}")
                        continue
                
                # Process with text model if available and we have vision results
                if agent.text_system_prompt and agent.service_id and vision_results:
                    try:
                        text_model = get_llm(agent, is_vision=False)
                        if text_model:
                            # Format data with text model using output parser
                            formatted_result = format_data_with_text_llm(
                                vision_results, 
                                text_model, 
                                pydantic_class, 
                                agent.text_system_prompt, 
                                "", 
                                os.path.basename(pdf_path)
                            )
                            
                            # Get final structured document data
                            final_result = get_document_data_from_pages(
                                agent.text_system_prompt,
                                formatted_result,
                                pydantic_class,
                                text_model,
                                "",
                                os.path.basename(pdf_path)
                            )
                            
                            return {
                                "method": "vision_and_text",
                                "content": final_result,
                                "extracted_text": vision_results,
                                "confidence": 0.8
                            }
                    except Exception as e:
                        logger.warning(f"Error processing with text model: {str(e)}")
                
                # Return vision results directly
                return {
                    "method": "vision_only",
                    "content": vision_results,
                    "extracted_text": vision_results,
                    "confidence": 0.7
                }
                
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {str(e)}")
            raise
    
    def _execute_langchain_agent(
        self, 
        agent: Agent, 
        message: str, 
        processed_files: List[Dict], 
        search_params: Dict = None,
        session = None,
        user_context: Dict = None,
        db: Session = None,
        temp_session_id: str = None,
        video_context_message: str = ""
    ) -> str:
        """Execute agent using the new create_agent approach with full tool support"""
        try:
            import asyncio
            from tools.agentTools import create_agent, prepare_agent_config
            
            # Re-query the agent with relationships loaded using repository
            fresh_agent = self.agent_execution_repo.get_agent_with_relationships(db, agent.agent_id)
            
            if not fresh_agent:
                raise Exception("Agent not found in database")
            
            # Enhance message with file contents if files were uploaded
            enhanced_message = message
            image_files = []
            
            # Add pre-computed video context if available
            if video_context_message:
                enhanced_message = message + video_context_message
                logger.info(f"Added pre-computed video context to message")
            
            if processed_files:
                #TODO: we should move this to class initialization? It is repeated in many places.
                # Get TMP_BASE_FOLDER from config
                app_config = get_app_config()
                tmp_base_folder = app_config['TMP_BASE_FOLDER']
                
                # Separate images from text files
                text_files_msg = ""
                for file_data in processed_files:
                    if file_data.get('type') == 'image':
                        image_files.append(file_data)
                    else:
                        text_files_msg += f"\n\n--- File: {file_data['filename']} (Path: {file_data['file_path']}) ---\n{file_data['content']}\n--- End of {file_data['filename']} ---"
                
                if text_files_msg:
                    enhanced_message += "\n\nFiles base folder is: " + tmp_base_folder
                    enhanced_message += "\n\n[Attached files:]" + text_files_msg
            
            # Wrap all async operations in a single event loop to avoid conflicts
            session_id_for_cache = session.id if (fresh_agent.has_memory and session) else None
            result_text = asyncio.run(self._execute_agent_async(
                fresh_agent, enhanced_message, search_params, session_id_for_cache, user_context, image_files
            ))
            
            return result_text
                
        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else repr(e)
            logger.error(f"Error executing LangChain agent: {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Agent execution failed: {error_msg}")
    
    async def _execute_agent_async(
        self,
        fresh_agent: Agent,
        message: str,
        search_params: Dict = None,
        session_id_for_cache: str = None,
        user_context: Dict = None,
        image_files: List[Dict] = None
    ) -> str:
        """Async helper to execute agent with MCP client in same event loop"""
        from tools.agentTools import create_agent, prepare_agent_config
        from langchain_core.messages import HumanMessage
        
        mcp_client = None
        checkpointer_cm = None
        try:
            # Create the agent chain with all tools and capabilities
            # All async operations happen in the SAME event loop
            agent_chain, tracer, mcp_client, checkpointer_cm = await create_agent(
                fresh_agent, search_params, session_id_for_cache, user_context
            )
            
            # Prepare configuration with tracer
            config = prepare_agent_config(fresh_agent, tracer)
            
            # Add session-specific configuration if memory is enabled
            if fresh_agent.has_memory and session_id_for_cache:
                config["configurable"]["thread_id"] = f"thread_{fresh_agent.agent_id}_{session_id_for_cache}"
                logger.info(f"Using session-aware thread_id: {config['configurable']['thread_id']}")
            else:
                config["configurable"]["thread_id"] = f"thread_{fresh_agent.agent_id}"
            
            # Add the question to config
            config["configurable"]["question"] = message
            
            # Execute the agent in the SAME event loop as where MCP client was created
            formatted_user_message = fresh_agent.prompt_template.format(question=message)
            
            # Construct message content
            if image_files:
                content = [{"type": "text", "text": formatted_user_message}]
                
                # Get TMP_BASE_FOLDER from config
                app_config = get_app_config()
                tmp_base_folder = app_config['TMP_BASE_FOLDER']
                
                # Check for aict_base_url environment variable (Production Mode)
                aict_base_url = os.getenv('AICT_BASE_URL')
                
                for img in image_files:
                    file_path = img.get('file_path', '')
                    if not file_path:
                        logger.warning(f"Image file has no file_path: {img}")
                        continue
                        
                    # Ensure forward slashes
                    file_path = file_path.replace('\\', '/')
                    if file_path.startswith('/'):
                        file_path = file_path[1:]
                    
                    # If aict_base_url is set, use it (Production Mode)
                    if aict_base_url:
                        # Remove trailing slash if present
                        if aict_base_url.endswith('/'):
                            aict_base_url = aict_base_url[:-1]
                            
                        # Generate signed URL
                        user_email = user_context.get('email') if user_context else None
                        if user_email:
                            from utils.security import generate_signature
                            sig = generate_signature(file_path, user_email)
                            url = f"{aict_base_url}/static/{file_path}?user={user_email}&sig={sig}"
                        else:
                            # Fallback if no user context (should not happen in auth mode)
                            url = f"{aict_base_url}/static/{file_path}"
                            
                        logger.info(f"Adding image to message using public URL: {url}")
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                    else:
                        # Fallback to Base64 (Development Mode)
                        # Use Base64 for local development to avoid localhost URL issues
                        try:
                            # Construct full path
                            full_path = os.path.join(tmp_base_folder, file_path)
                            
                            if os.path.exists(full_path):
                                import base64
                                import mimetypes
                                
                                mime_type, _ = mimetypes.guess_type(full_path)
                                if not mime_type:
                                    mime_type = "image/jpeg"
                                    
                                with open(full_path, "rb") as image_file:
                                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                                    
                                data_url = f"data:{mime_type};base64,{encoded_string}"
                                logger.info(f"Adding image to message as base64 (length: {len(encoded_string)})")
                                
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": data_url}
                                })
                            else:
                                # Fallback to URL if file not found locally (should not happen)
                                url = f"http://localhost:8000/static/{file_path}"
                                logger.warning(f"Image file not found at {full_path}, falling back to URL: {url}")
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": url}
                                })
                        except Exception as e:
                            logger.error(f"Error processing image for base64: {e}")
                            # Fallback to URL
                            url = f"http://localhost:8000/static/{file_path}"
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": url}
                            })
                
                message_payload = HumanMessage(content=content)
            else:
                message_payload = HumanMessage(content=formatted_user_message)
            
            result = await agent_chain.ainvoke({"messages": [message_payload]}, config=config)
            
            # Extract the response from the result
            if isinstance(result, dict) and "messages" in result:
                # Get the last AI message
                messages = result["messages"]
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        return msg.content
                # Fallback: return the last message content
                if messages:
                    return str(messages[-1].content) if hasattr(messages[-1], 'content') else str(messages[-1])
            
            # If result is a string, return it directly
            if isinstance(result, str):
                return result
                
            # Fallback: convert to string
            return str(result)
        finally:
            # As of langchain-mcp-adapters 0.1.0, MCP client doesn't need manual cleanup
            if mcp_client:
                logger.info("MCP client will be cleaned up automatically")
            
            # Always close the checkpointer context manager
            if checkpointer_cm:
                try:
                    await checkpointer_cm.__aexit__(None, None, None)
                    logger.debug("Checkpointer context manager closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing checkpointer context manager: {e}")
    
    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location"""
        import tempfile
        
        #TODO: we should move this to class initialization? It is repeated in many places.
        # Get TMP_BASE_FOLDER from config
        app_config = get_app_config()
        tmp_base_folder = app_config['TMP_BASE_FOLDER']
        uploads_dir = os.path.join(tmp_base_folder, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Create temporary file in TMP_BASE_FOLDER/uploads
        suffix = os.path.splitext(file.filename)[1] if file.filename else ""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=uploads_dir)
        
        try:
            # Write file content
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            return temp_file.name
        finally:
            temp_file.close()
    
    def _update_request_count(self, agent: Agent, db: Session):
        """Update agent request count"""
        self.agent_execution_repo.update_agent_request_count(db, agent.agent_id) 