"""Core data models."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class CommunicationStyle(str, Enum):
    THERAPEUTIC = "therapeutic"
    EDUCATIONAL = "educational"
    EMPATHETIC = "empathetic"
    PRACTICAL = "practical"


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationSession(BaseModel):
    session_id: str
    user_id: str
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    content: str
    confidence: float
    primary_style: CommunicationStyle
    style_scores: Dict[CommunicationStyle, float]
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class User(BaseModel):
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingExample(BaseModel):
    text: str
    style: CommunicationStyle
    source: str
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
