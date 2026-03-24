from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form, BackgroundTasks, Query
from typing import List, Optional
from lks_idprovider import AuthContext
from sqlalchemy.orm import Session
import json
import os
import hashlib

from services.agent_service import AgentService
from services.mcp_server_service import MCPServerService
from db.database import get_db
from schemas.agent_schemas import AgentListItemSchema, AgentDetailSchema, CreateUpdateAgentSchema, UpdatePromptSchema
from schemas.chat_schemas import ChatResponseSchema, ResetResponseSchema, ConversationHistorySchema
from services.agent_execution_service import AgentExecutionService
from services.file_management_service import FileManagementService, FileReference
from routers.internal.auth_utils import get_current_user_oauth
from routers.controls.file_size_limit import enforce_file_size_limit
from routers.controls.role_authorization import require_min_role, AppRole

from utils.logger import get_logger

logger = get_logger(__name__)

agents_router = APIRouter()

AGENT_NOT_FOUND_ERROR = "Agent not found"
INTERNAL_SERVER_ERROR = "Internal server error"

#DEPENDENCIES

def get_agent_service() -> AgentService:
    """Dependency to get AgentService instance"""
    return AgentService()


# ==================== VIDEO PROCESSING HELPERS ====================

def generate_temp_session_id(
    agent_id: int,
    user_id: int,
    conversation_id: Optional[int],
    log_prefix: str = "video"
) -> str:
    """
    Generate a deterministic temp_session_id for video chunk collection.
    
    Args:
        agent_id: Agent ID
        user_id: User ID
        conversation_id: Conversation ID (or None)
        log_prefix: Prefix for log messages
        
    Returns:
        16-character hex string for temp_session_id
    """
    session_key_str = f"{agent_id}_{user_id}_{conversation_id or 'no_conv'}"
    temp_session_id = hashlib.sha256(session_key_str.encode()).hexdigest()[:16]
    
    logger.info(f"{log_prefix} - Creating temp session:")
    logger.info(f"  agent_id={agent_id}, user_id={user_id}, conversation_id={conversation_id}")
    logger.info(f"  session_key_str={session_key_str}")
    logger.info(f"  temp_session_id={temp_session_id}")
    logger.info(f"  full collection name: temp_session_{temp_session_id}")
    
    return temp_session_id


def get_video_processing_services(
    db: Session,
    app_id: int,
    agent_id: int,
    agent_service: AgentService
) -> tuple:
    """
    Get embedding service and AI service for video processing.
    
    Args:
        db: Database session
        app_id: App ID
        agent_id: Agent ID
        agent_service: AgentService instance
        
    Returns:
        Tuple of (embedding_service, ai_service_id)
        
    Raises:
        HTTPException if services not available
    """
    from repositories.embedding_service_repository import EmbeddingServiceRepository
    from repositories.ai_service_repository import AIServiceRepository
    
    # Get agent
    agent = agent_service.get_agent(db, app_id, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get embedding service - try agent's silo first, then app-level fallback
    embedding_service = None
    if agent.silo and agent.silo.embedding_service:
        embedding_service = agent.silo.embedding_service
        logger.info(f"Using embedding service from agent's silo: {embedding_service.name}")
    else:
        app_embedding_services = EmbeddingServiceRepository.get_by_app_id(db, app_id)
        if app_embedding_services:
            embedding_service = app_embedding_services[0]
            logger.info(f"Using fallback app-level embedding service: {embedding_service.name}")
        else:
            raise HTTPException(
                status_code=400, 
                detail="No embedding service available for video processing"
            )
    
    # Get AI service for transcription (OpenAI with Whisper)
    ai_service_id = None
    ai_services = AIServiceRepository.get_by_app_id(db, app_id)
    for service in ai_services:
        if service.provider == 'OpenAI':
            ai_service_id = service.service_id
            break
    
    if not ai_service_id:
        raise HTTPException(
            status_code=400, 
            detail="No OpenAI AI service found for transcription"
        )
    
    return embedding_service, ai_service_id


#AGENT MANAGEMENT

@agents_router.get("/", 
                  summary="List agents",
                  tags=["Agents"],
                  response_model=List[AgentListItemSchema])
async def list_agents(
    app_id: int, 
    auth_context: AuthContext = Depends(get_current_user_oauth),
    role: AppRole = Depends(require_min_role("viewer")),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    List all agents for a specific app.
    """
    # App access validation would be implemented here
    
    # Get agents using service
    agents_list = agent_service.get_agents_list(db, app_id)
    
    return agents_list


@agents_router.get("/{agent_id}",
                  summary="Get agent details",
                  tags=["Agents"],
                  response_model=AgentDetailSchema)
async def get_agent(
    app_id: int, 
    agent_id: int, 
    auth_context: AuthContext = Depends(get_current_user_oauth),
    role: AppRole = Depends(require_min_role("viewer")),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Get detailed information about a specific agent plus form data for editing.
    """
    # App access validation would be implemented here
    
    # Get agent details using service
    agent_detail = agent_service.get_agent_detail(db, app_id, agent_id)
    
    if not agent_detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=AGENT_NOT_FOUND_ERROR
        )
    
    return agent_detail


@agents_router.post("/{agent_id}",
                   summary="Create or update agent",
                   tags=["Agents"],
                   response_model=AgentDetailSchema)
async def create_or_update_agent(
    app_id: int,
    agent_id: int,
    agent_data: CreateUpdateAgentSchema,
    auth_context: AuthContext = Depends(get_current_user_oauth),
    role: AppRole = Depends(require_min_role("editor")),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Create a new agent or update an existing one.
    """
    # App access validation would be implemented here
    
    # Prepare agent data
    agent_dict = {
        'agent_id': agent_id,
        'app_id': app_id,
        'name': agent_data.name,
        'description': agent_data.description,
        'system_prompt': agent_data.system_prompt,
        'prompt_template': agent_data.prompt_template,
        'type': agent_data.type,
        'is_tool': agent_data.is_tool,
        'has_memory': agent_data.has_memory,
        'service_id': agent_data.service_id,
        'silo_id': agent_data.silo_id,
        'output_parser_id': agent_data.output_parser_id,
        'temperature': agent_data.temperature,
        # OCR-specific fields
        'vision_service_id': agent_data.vision_service_id,
        'vision_system_prompt': agent_data.vision_system_prompt,
        'text_system_prompt': agent_data.text_system_prompt
    }
    
    logger.info(f"Creating/updating agent with data: {agent_dict}")
    
    # Create or update agent
    created_agent_id = agent_service.create_or_update_agent(db, agent_dict, agent_data.type)
    
    # Update tools, MCPs, and skills (always call to handle empty arrays for unselecting)
    agent_service.update_agent_tools(db, created_agent_id, agent_data.tool_ids, {})
    agent_service.update_agent_mcps(db, created_agent_id, agent_data.mcp_config_ids, {})
    agent_service.update_agent_skills(db, created_agent_id, agent_data.skill_ids, {})

    # Return updated agent (reuse the GET logic)
    return await get_agent(app_id, created_agent_id, auth_context, role, db, agent_service)


@agents_router.delete("/{agent_id}",
                     summary="Delete agent",
                     tags=["Agents"])
async def delete_agent(
    app_id: int,
    agent_id: int,
    auth_context: AuthContext = Depends(get_current_user_oauth),
    role: AppRole = Depends(require_min_role("editor")),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Delete an agent.
    """
    # App access validation would be implemented here

    # Check if agent exists
    agent = agent_service.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=AGENT_NOT_FOUND_ERROR
        )

    # Delete agent
    success = agent_service.delete_agent(db, agent_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete agent"
        )

    return {"message": "Agent deleted successfully"}


@agents_router.get("/{agent_id}/mcp-usage",
                   summary="Get MCP servers using this agent",
                   tags=["Agents"])
async def get_agent_mcp_usage(
    app_id: int,
    agent_id: int,
    auth_context: AuthContext = Depends(get_current_user_oauth),
    role: AppRole = Depends(require_min_role("viewer")),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Get list of MCP servers that use this agent.
    Used to warn users before unmarking an agent as tool or deleting it.
    """
    # Check if agent exists
    agent = agent_service.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=AGENT_NOT_FOUND_ERROR
        )

    # Get MCP servers using this agent
    servers = MCPServerService.get_mcp_servers_using_agent(db, agent_id)

    return {
        "agent_id": agent_id,
        "is_tool": agent.is_tool,
        "mcp_servers": servers,
        "used_in_mcp_servers": len(servers) > 0
    }


@agents_router.post("/{agent_id}/update-prompt",
                   summary="Update agent prompt",
                   tags=["Agents"])
async def update_agent_prompt(
    app_id: int,
    agent_id: int,
    prompt_data: UpdatePromptSchema,
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Update agent system prompt or prompt template.
    """
    # Validate prompt type
    if prompt_data.type not in ['system', 'template']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid prompt type. Must be 'system' or 'template'"
        )
    
    # Update prompt using service
    success = agent_service.update_agent_prompt(db, agent_id, prompt_data.type, prompt_data.prompt)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=AGENT_NOT_FOUND_ERROR
        )
    
    return {"message": f"{prompt_data.type.capitalize()} prompt updated successfully"}


# ==================== PLAYGROUND & ANALYTICS ====================

@agents_router.get("/{agent_id}/playground",
                  summary="Get agent playground",
                  tags=["Agents", "Playground"])
async def agent_playground(
    app_id: int, 
    agent_id: int, 
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Get agent playground interface data.
    """
    # App access validation would be implemented here
    
    playground_data = agent_service.get_agent_playground_data(db, agent_id)
    
    if not playground_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=AGENT_NOT_FOUND_ERROR
        )
    
    return playground_data


@agents_router.get("/{agent_id}/analytics",
                  summary="Get agent analytics",
                  tags=["Agents", "Analytics"])
async def agent_analytics(
    app_id: int, 
    agent_id: int, 
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Get agent analytics data (premium feature).
    """
    # App access validation would be implemented here
    # Premium feature check would be implemented here
    
    analytics_data = agent_service.get_agent_analytics(db, agent_id)
    
    if not analytics_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=AGENT_NOT_FOUND_ERROR
        )
    
    return analytics_data


# ==================== CHAT ENDPOINTS ====================

async def _save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location and return file path"""
    import tempfile
    import os
    
    # Get TMP_BASE_FOLDER from config
    from utils.config import get_app_config
    app_config = get_app_config()
    tmp_base_folder = app_config['TMP_BASE_FOLDER']
    uploads_dir = os.path.join(tmp_base_folder, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Create temporary file in TMP_BASE_FOLDER/uploads
    suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ''
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=uploads_dir) as temp_file:
        content = await upload_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    return temp_file_path


@agents_router.post("/{agent_id}/chat",
                  summary="Chat with agent",
                  tags=["Agents"],
                  response_model=ChatResponseSchema)
async def chat_with_agent(
    app_id: int,
    agent_id: int,
    request: Request,
    message: str = Form(...),
    files: List[UploadFile] = File(None),
    file_references: Optional[str] = Form(None),
    search_params: Optional[str] = Form(None),
    conversation_id: Optional[int] = Form(None),
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    _: None = Depends(enforce_file_size_limit)
):
    """
    Internal API: Chat with agent for playground (OAuth authentication)
    
    Args:
        agent_id: ID of the agent
        message: User message
        files: Optional uploaded files
        file_references: Optional JSON array of file_ids to include. If not provided, all files are included.
        search_params: Optional search parameters
        conversation_id: Optional conversation ID to continue existing conversation
    """
    try:
        # Parse search params if provided
        parsed_search_params = None
        if search_params:
            try:
                parsed_search_params = json.loads(search_params)
            except json.JSONDecodeError:
                logger.warning("Invalid search_params JSON")
        
        # Parse file_references if provided (for filtering which files to include)
        parsed_file_references = None
        if file_references:
            try:
                parsed_file_references = json.loads(file_references)
                if not isinstance(parsed_file_references, list):
                    parsed_file_references = None
            except json.JSONDecodeError:
                logger.warning("Invalid file_references JSON, ignoring")
        
        # Extract JWT token from Authorization header for MCP authentication
        auth_header = request.headers.get('Authorization', '')
        jwt_token = None
        if auth_header.startswith('Bearer '):
            jwt_token = auth_header.split(' ')[1]
            logger.debug(f"Extracted JWT token for MCP auth (length: {len(jwt_token)})")
        
        # Create user context for OAuth user
        user_context = {
            "user_id": int(auth_context.identity.id),
            "email": auth_context.identity.email,
            "oauth": True,
            "app_id": app_id,
            "token": jwt_token  # Add JWT token for MCP authentication
        }
        
        # Process files using FileManagementService for persistence
        file_service = FileManagementService()
        all_file_references = []
        uploaded_file_ids = set()  # Track newly uploaded files to avoid duplicates
        
        # Add any new files uploaded with this message
        if files:
            for upload_file in files:
                if upload_file.filename:  # Skip empty file slots
                    # Upload file to persistent storage
                    file_ref = await file_service.upload_file(
                        file=upload_file,
                        agent_id=agent_id,
                        user_context=user_context,
                        conversation_id=conversation_id
                    )
                    all_file_references.append(file_ref)
                    uploaded_file_ids.add(file_ref.file_id)
        
        # Get previously uploaded files for this session/conversation
        existing_files = await file_service.list_attached_files(
            agent_id=agent_id,
            user_context=user_context,
            conversation_id=str(conversation_id) if conversation_id else None
        )
        
        # Filter existing files if file_references was provided
        if parsed_file_references:
            requested_file_ids = set(parsed_file_references)
            existing_files = [f for f in existing_files if f['file_id'] in requested_file_ids]
            logger.info(f"Filtered to {len(existing_files)} files based on file_references")
        
        # Convert existing files to FileReference objects (avoiding duplicates)
        for file_data in existing_files:
            if file_data['file_id'] not in uploaded_file_ids:
                file_ref = FileReference(
                    file_id=file_data['file_id'],
                    filename=file_data['filename'],
                    file_type=file_data['file_type'],
                    content=file_data['content'],
                    file_path=file_data.get('file_path'),
                    temp_session_id=file_data.get('temp_session_id'),
                    temp_media_id=file_data.get('temp_media_id'),
                    processing_status=file_data.get('processing_status')
                )
                all_file_references.append(file_ref)
        
        # Use unified service layer with file references
        execution_service = AgentExecutionService(db)
        result = await execution_service.execute_agent_chat_with_file_refs(
            agent_id=agent_id,
            message=message,
            file_references=all_file_references,
            search_params=parsed_search_params,
            user_context=user_context,
            conversation_id=conversation_id,
            db=db
        )
        
        logger.info(f"Chat request processed for agent {agent_id} by user {auth_context.identity.id}")
        return ChatResponseSchema(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR)


@agents_router.post("/{agent_id}/reset",
                  summary="Reset conversation",
                  tags=["Agents"],
                  response_model=ResetResponseSchema)
async def reset_conversation(
    app_id: int,
    agent_id: int,
    conversation_id: Optional[int] = Query(None, description="Conversation ID to reset (required to properly clean up attached files)"),
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db)
):
    """
    Internal API: Reset conversation for playground (OAuth authentication)
    
    Args:
        conversation_id: Specific conversation to reset. Required to properly clean up attached files.
    """
    try:
        # Create user context for OAuth user
        user_context = {
            "user_id": int(auth_context.identity.id),
            "oauth": True,
            "app_id": app_id,
            "conversation_id": str(conversation_id) if conversation_id else None
        }
        
        # Use unified service layer
        execution_service = AgentExecutionService(db)
        success = await execution_service.reset_agent_conversation(
            agent_id=agent_id,
            user_context=user_context,
            db=db
        )
        
        if success:
            logger.info(f"Conversation reset for agent {agent_id} by user {auth_context.identity.id}")
            return ResetResponseSchema(success=True, message="Conversation reset successfully")
        else:
            return ResetResponseSchema(success=False, message="Failed to reset conversation")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in reset endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR)


@agents_router.get("/{agent_id}/conversation-history",
                  summary="Get conversation history",
                  tags=["Agents"],
                  response_model=ConversationHistorySchema)
async def get_conversation_history(
    app_id: int,
    agent_id: int,
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Internal API: Get conversation history for playground (OAuth authentication)
    """
    try:
        # Get agent to check if it has memory
        agent = agent_service.get_agent(db, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create user context for OAuth user
        user_context = {
            "user_id": int(auth_context.identity.id),
            "oauth": True,
            "app_id": app_id
        }
        
        # Use unified service layer
        execution_service = AgentExecutionService(db)
        messages = await execution_service.get_conversation_history(
            agent_id=agent_id,
            user_context=user_context,
            db=db
        )
        
        logger.info(f"Retrieved {len(messages)} messages for agent {agent_id} by user {auth_context.identity.id}")
        return ConversationHistorySchema(
            messages=messages,
            agent_id=agent_id,
            has_memory=agent.has_memory
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in conversation history endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR)


@agents_router.post("/{agent_id}/upload-file",
                  summary="Upload file for chat",
                  tags=["Agents"])
async def upload_file_for_chat(
    app_id: int,
    agent_id: int,
    file: UploadFile = File(...),
    conversation_id: Optional[int] = Form(None),
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    _: None = Depends(enforce_file_size_limit)
):
    """
    Internal API: Upload file for chat (OAuth authentication)
    
    Args:
        conversation_id: Optional conversation ID to associate the file with.
                        If provided, file will be specific to that conversation.
    """
    try:
        # Create user context for OAuth user
        user_context = {
            "user_id": int(auth_context.identity.id),
            "oauth": True,
            "app_id": app_id
        }
        
        # Use unified service layer
        file_service = FileManagementService()
        file_ref = await file_service.upload_file(
            file=file,
            agent_id=agent_id,
            user_context=user_context,
            conversation_id=conversation_id
        )
        
        logger.info(f"File uploaded for agent {agent_id} by user {auth_context.identity.id}")
        return {
            "success": True,
            "file_id": file_ref.file_id,
            "filename": file_ref.filename,
            "file_type": file_ref.file_type,
            # Visual feedback fields
            "file_size_bytes": file_ref.file_size_bytes,
            "file_size_display": FileReference.format_file_size(file_ref.file_size_bytes),
            "processing_status": file_ref.processing_status,
            "content_preview": file_ref.content_preview,
            "has_extractable_content": file_ref.has_extractable_content,
            "mime_type": file_ref.mime_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")


@agents_router.get("/{agent_id}/files",
                 summary="List attached files",
                 tags=["Agents"])
async def list_attached_files(
    app_id: int,
    agent_id: int,
    conversation_id: Optional[int] = None,
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db)
):
    """
    Internal API: List attached files for chat (OAuth authentication)
    
    Args:
        conversation_id: Optional conversation ID to filter files.
                        If provided, only files for that conversation are returned.
    """
    try:
        # Create user context for OAuth user
        user_context = {
            "user_id": int(auth_context.identity.id),
            "oauth": True,
            "app_id": app_id
        }
        
        # Use unified service layer
        file_service = FileManagementService()
        files = await file_service.list_attached_files(
            agent_id=agent_id,
            user_context=user_context,
            conversation_id=str(conversation_id) if conversation_id else None
        )
        
        # Calculate total size for visual feedback
        total_size = sum(f.get('file_size_bytes', 0) or 0 for f in files)
        
        return {
            "files": files,
            "total_size_bytes": total_size,
            "total_size_display": FileReference.format_file_size(total_size)
        }
        
    except Exception as e:
        logger.error(f"Error in list files endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list files")


@agents_router.delete("/{agent_id}/files/{file_id}",
                    summary="Remove attached file",
                    tags=["Agents"])
async def remove_attached_file(
    app_id: int,
    agent_id: int,
    file_id: str,
    conversation_id: Optional[int] = None,
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db)
):
    """
    Internal API: Remove attached file (OAuth authentication)
    
    Args:
        conversation_id: Optional conversation ID for conversation-specific files.
    """
    try:
        # Create user context for OAuth user
        user_context = {
            "user_id": int(auth_context.identity.id),
            "oauth": True,
            "app_id": app_id
        }
        
        # Use unified service layer
        file_service = FileManagementService()
        success = await file_service.remove_file(
            file_id=file_id,
            agent_id=agent_id,
            user_context=user_context,
            conversation_id=str(conversation_id) if conversation_id else None
        )
        
        if success:
            logger.info(f"File {file_id} removed for agent {agent_id} by user {auth_context.identity.id}")
            return {"success": True, "message": "File removed successfully"}
        else:
            return {"success": False, "message": "File not found or already removed"}
            
    except Exception as e:
        logger.error(f"Error in remove file endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to remove file")


@agents_router.post("/{agent_id}/youtube",
                   summary="Add YouTube video for chat",
                   tags=["Agents"])
async def add_youtube_for_chat(
    app_id: int,
    agent_id: int,
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    conversation_id: Optional[int] = Form(None),
    forced_language: Optional[str] = Form(None, description="Force transcription language (e.g., 'es', 'en', 'fr'). Leave empty for auto-detect."),
    chunk_min_duration: Optional[int] = Form(None, description="Minimum chunk duration in seconds (default: 30)"),
    chunk_max_duration: Optional[int] = Form(None, description="Maximum chunk duration in seconds (default: 120)"),
    chunk_overlap: Optional[int] = Form(None, description="Overlap between chunks in seconds (default: 5)"),
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Add YouTube video for chat: download, transcribe and create temporary silo.
    
    The video will be:
    1. Downloaded from YouTube
    2. Audio extracted and normalized
    3. Transcribed using Whisper
    4. Chunked into segments
    5. Indexed in temporary collection for the conversation
    
    When the conversation is cleared or ended, the video context will be removed.
    
    Configuration:
    - forced_language: Force transcription language (e.g., 'es', 'en', 'fr'). Leave empty for auto-detect.
    - chunk_min_duration: Minimum chunk duration in seconds (default: 30)
    - chunk_max_duration: Maximum chunk duration in seconds (default: 120)
    - chunk_overlap: Overlap between chunks in seconds (default: 5)
    
    Args:
        url: YouTube URL (youtube.com or youtu.be)
        conversation_id: Conversation ID to associate the video with
    """
    from repositories.embedding_service_repository import EmbeddingServiceRepository
    from repositories.ai_service_repository import AIServiceRepository
    from utils.config import get_app_config
    from tasks.media_tasks import process_playground_youtube_task
    import uuid
    import re
    
    try:
        # Validate YouTube URL
        youtube_pattern = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
        if not re.match(youtube_pattern, url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Create user context
        user_id = int(auth_context.identity.id)
        user_context = {
            "user_id": user_id,
            "oauth": True,
            "app_id": app_id
        }
        
        # Generate unique IDs
        file_id = str(uuid.uuid4())
        media_id = file_id[:8]
        
        # Get configuration
        app_config = get_app_config()
        tmp_base_folder = app_config['TMP_BASE_FOLDER']
        
        # Generate session ID for temp collection (based on conversation)
        file_service = FileManagementService()
        file_session_key = file_service._get_session_key(agent_id, user_context, str(conversation_id) if conversation_id else None)
        temp_session_id = generate_temp_session_id(agent_id, user_id, conversation_id, "youtube")
        
        # Get embedding and AI services for video processing
        embedding_service, ai_service_id = get_video_processing_services(db, app_id, agent_id, agent_service)
        
        # Create a file reference for the YouTube video
        file_ref = FileReference(
            file_id=file_id,
            filename=f"YouTube: {url[:50]}...",
            file_type="video",
            content="__YOUTUBE_PENDING_PROCESSING__",
            file_path=None,
            file_size_bytes=0,
            conversation_id=str(conversation_id) if conversation_id else None,
            processing_status="downloading"
        )
        
        # Initialize session if not exists and save file reference
        if file_session_key not in file_service._files:
            file_service._files[file_session_key] = {}
        file_service._files[file_session_key][file_id] = file_ref
        
        # Save to disk for persistence
        await file_service._save_youtube_file_to_disk(file_session_key, file_id, file_ref, url)
        
        logger.info(f"Created YouTube file reference: {file_id}")
        
        # Schedule background task for YouTube download and processing
        background_tasks.add_task(
            process_playground_youtube_task,
            youtube_url=url,
            session_id=temp_session_id,
            embedding_service_id=embedding_service.service_id,
            ai_service_id=ai_service_id,
            media_id=media_id,
            file_id=file_id,
            session_key=file_session_key,
            tmp_base_folder=tmp_base_folder,
            forced_language=forced_language,
            chunk_min_duration=chunk_min_duration,
            chunk_max_duration=chunk_max_duration,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Scheduled background YouTube processing for {file_id}")
        
        # Return immediately - frontend will poll for status
        return {
            "success": True,
            "message": "YouTube video download and processing started",
            "file_id": file_id,
            "filename": file_ref.filename,
            "file_type": "video",
            "processing_status": "downloading",
            "session_id": temp_session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting YouTube processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YouTube processing failed: {str(e)}")


@agents_router.post("/{agent_id}/files/{file_id}/process-video",
                   summary="Process attached video or audio",
                   tags=["Agents"])
async def process_attached_video(
    app_id: int,
    agent_id: int,
    file_id: str,
    background_tasks: BackgroundTasks,
    conversation_id: Optional[int] = None,
    forced_language: Optional[str] = Query(None, description="Force transcription language (e.g., 'es', 'en', 'fr'). Leave empty for auto-detect."),
    chunk_min_duration: Optional[int] = Query(None, description="Minimum chunk duration in seconds (default: 30)"),
    chunk_max_duration: Optional[int] = Query(None, description="Maximum chunk duration in seconds (default: 120)"),
    chunk_overlap: Optional[int] = Query(None, description="Overlap between chunks in seconds (default: 5)"),
    auth_context: AuthContext = Depends(get_current_user_oauth),
    db: Session = Depends(get_db),
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Process an attached video or audio file: transcribe and create a temporary silo.
    
    Supported formats:
    - Video: mp4, webm, mov, avi, mkv, m4v
    - Audio: mp3, wav, m4a, ogg, flac, aac, wma
    
    This endpoint schedules background processing and returns immediately.
    The frontend should poll the file status to know when processing is complete.
    
    Configuration:
    - forced_language: Force transcription language (e.g., 'es', 'en', 'fr'). Leave empty for auto-detect.
    - chunk_min_duration: Minimum chunk duration in seconds (default: 30)
    - chunk_max_duration: Maximum chunk duration in seconds (default: 120)
    - chunk_overlap: Overlap between chunks in seconds (default: 5)
    
    Args:
        conversation_id: Conversation ID to associate the temp silo with
    """
    from repositories.embedding_service_repository import EmbeddingServiceRepository
    from repositories.ai_service_repository import AIServiceRepository
    from utils.config import get_app_config
    from tasks.media_tasks import process_playground_video_task
    
    try:
        # Create user context
        user_id = int(auth_context.identity.id)
        user_context = {
            "user_id": user_id,
            "oauth": True,
            "app_id": app_id
        }
        
        # Get the file reference
        file_service = FileManagementService()
        file_ref = await file_service.get_file(
            file_id=file_id,
            agent_id=agent_id,
            user_context=user_context,
            conversation_id=str(conversation_id) if conversation_id else None
        )
        
        if not file_ref:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Verify it's a video or audio file (both support transcription)
        if file_ref.file_type not in ('video', 'audio'):
            raise HTTPException(status_code=400, detail="File is not a video or audio file")
        
        # Get video path - try content first (for __VIDEO_PENDING_PROCESSING__:path format)
        # or construct from file_path relative to TMP_BASE_FOLDER
        video_path = None
        app_config = get_app_config()
        tmp_base_folder = app_config['TMP_BASE_FOLDER']
        
        if file_ref.content and file_ref.content.startswith("__VIDEO_PENDING_PROCESSING__:"):
            # Extract absolute path from content
            video_path = file_ref.content.replace("__VIDEO_PENDING_PROCESSING__:", "")
        elif file_ref.file_path:
            # Construct absolute path from relative path
            video_path = os.path.join(tmp_base_folder, file_ref.file_path)
        
        logger.info(f"Video path resolved: {video_path}")
        
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Video file not accessible: {video_path}")
        
        # Generate session ID for temp collection (based on conversation)
        file_session_key = file_service._get_session_key(agent_id, user_context, str(conversation_id) if conversation_id else None)
        temp_session_id = generate_temp_session_id(agent_id, user_id, conversation_id, "process-video")
        
        # Get embedding and AI services for video processing
        embedding_service, ai_service_id = get_video_processing_services(db, app_id, agent_id, agent_service)
        
        media_id = file_id[:8]  # Use first 8 chars of file_id
        
        # Update status to 'processing' immediately
        await file_service.update_file_metadata(
            agent_id=agent_id,
            file_id=file_id,
            user_context=user_context,
            conversation_id=str(conversation_id) if conversation_id else None,
            updates={'processing_status': 'processing'}
        )
        logger.info(f"Set file {file_id} status to 'processing'")
        
        # Schedule background task for video processing
        background_tasks.add_task(
            process_playground_video_task,
            video_path=video_path,
            session_id=temp_session_id,
            embedding_service_id=embedding_service.service_id,
            ai_service_id=ai_service_id,
            filename=file_ref.filename,
            media_id=media_id,
            file_id=file_id,
            session_key=file_session_key,
            forced_language=forced_language,
            chunk_min_duration=chunk_min_duration,
            chunk_max_duration=chunk_max_duration,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Scheduled background video processing for {file_id}")
        
        # Return immediately - frontend will poll for status
        return {
            "success": True,
            "message": "Video processing started",
            "status": "processing",
            "session_id": temp_session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")