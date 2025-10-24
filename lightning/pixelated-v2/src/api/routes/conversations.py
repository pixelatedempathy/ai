"""Conversation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid

from src.core.models import ConversationSession, ConversationMessage, CommunicationStyle
from src.database.operations import ConversationOperations, UserOperations
from src.core.logging import get_logger

logger = get_logger("api.conversations")
router = APIRouter()


class CreateSessionRequest(BaseModel):
    user_id: str
    metadata: Dict[str, Any] = {}


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


class MessageRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}


class MessageResponse(BaseModel):
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]


class ConversationResponse(BaseModel):
    session_id: str
    user_id: str
    messages: List[MessageResponse]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


class ChatRequest(BaseModel):
    message: str
    session_id: str
    metadata: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    response: str
    confidence: float
    primary_style: str
    style_scores: Dict[str, float]
    processing_time: float
    metadata: Dict[str, Any] = {}


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new conversation session."""
    logger.info(f"Creating session for user: {request.user_id}")
    
    # Verify user exists
    user = UserOperations.get_user(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create session
    session_id = str(uuid.uuid4())
    session = ConversationSession(
        session_id=session_id,
        user_id=request.user_id,
        metadata=request.metadata
    )
    
    if ConversationOperations.create_session(session):
        logger.info(f"Session created: {session_id}")
        return SessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            metadata=session.metadata
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get("/sessions/{session_id}", response_model=ConversationResponse)
async def get_session(session_id: str):
    """Get conversation session with messages."""
    logger.info(f"Getting session: {session_id}")
    
    session = ConversationOperations.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = [
        MessageResponse(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp.isoformat(),
            metadata=msg.metadata
        )
        for msg in session.messages
    ]
    
    return ConversationResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        messages=messages,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        metadata=session.metadata
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send message and get AI response."""
    logger.info(f"Chat request for session: {request.session_id}")
    
    # Verify session exists
    session = ConversationOperations.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add user message
    user_message = ConversationMessage(
        role="user",
        content=request.message,
        metadata=request.metadata
    )
    
    message_id = ConversationOperations.add_message(request.session_id, user_message)
    if not message_id:
        raise HTTPException(status_code=500, detail="Failed to save user message")
    
    # Generate AI response using MoE system
    from src.api.services.conversation_service import ConversationService
    
    conversation_service = ConversationService(use_mock_inference=True)
    model_response = await conversation_service.generate_response(
        message=request.message,
        session_id=request.session_id,
        user_id=session.user_id
    )
    
    # Add AI message
    ai_message = ConversationMessage(
        role="assistant",
        content=model_response.content,
        metadata={
            'confidence': model_response.confidence,
            'primary_style': model_response.primary_style.value,
            'style_scores': {k.value: v for k, v in model_response.style_scores.items()},
            'processing_time': model_response.processing_time
        }
    )
    
    ai_message_id = ConversationOperations.add_message(request.session_id, ai_message)
    if not ai_message_id:
        raise HTTPException(status_code=500, detail="Failed to save AI message")
    
    logger.info(f"Chat response generated for session: {request.session_id}")
    
    return ChatResponse(
        response=model_response.content,
        confidence=model_response.confidence,
        primary_style=model_response.primary_style.value,
        style_scores={k.value: v for k, v in model_response.style_scores.items()},
        processing_time=model_response.processing_time,
        metadata=model_response.metadata
    )
