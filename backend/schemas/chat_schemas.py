from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

# ==================== CHAT SCHEMAS ====================

class ChatRequestSchema(BaseModel):
    """Schema for chat request"""
    message: str
    files: Optional[List[str]] = None  # File IDs
    search_params: Optional[dict] = None


class VideoReferenceSchema(BaseModel):
    """Schema for video reference with timestamp"""
    file_id: str
    filename: str
    start_time: float
    end_time: float
    start_formatted: str  # MM:SS format
    end_formatted: str    # MM:SS format
    text_preview: str     # Snippet of transcription
    video_url: str        # Signed URL to access full video
    is_agent_cited: Optional[bool] = None  # True if agent explicitly cited this timestamp


class ChatResponseSchema(BaseModel):
    """Schema for chat response"""
    response: Union[str, dict]  # Can be string or JSON object
    agent_id: int
    conversation_id: Optional[int] = None  # ID of the conversation if using multi-conversation system
    metadata: dict
    video_references: Optional[List[VideoReferenceSchema]] = None  # Relevant video segments


class ResetResponseSchema(BaseModel):
    """Schema for reset response"""
    success: bool
    message: str


class ConversationMessageSchema(BaseModel):
    """Schema for a single conversation message"""
    role: str  # 'user' or 'agent'
    content: str


class ConversationHistorySchema(BaseModel):
    """Schema for conversation history response"""
    messages: List[ConversationMessageSchema]
    agent_id: int
    has_memory: bool
