"""User management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from src.core.models import User
from src.database.operations import UserOperations
from src.core.logging import get_logger

logger = get_logger("api.users")
router = APIRouter()


class CreateUserRequest(BaseModel):
    user_id: str
    preferences: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class UserResponse(BaseModel):
    user_id: str
    created_at: str
    preferences: Dict[str, Any]
    metadata: Dict[str, Any]


@router.post("/", response_model=UserResponse)
async def create_user(request: CreateUserRequest):
    """Create a new user."""
    logger.info(f"Creating user: {request.user_id}")
    
    # Check if user already exists
    existing_user = UserOperations.get_user(request.user_id)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create new user
    user = User(
        user_id=request.user_id,
        preferences=request.preferences,
        metadata=request.metadata
    )
    
    if UserOperations.create_user(user):
        logger.info(f"User created successfully: {request.user_id}")
        return UserResponse(
            user_id=user.user_id,
            created_at=user.created_at.isoformat(),
            preferences=user.preferences,
            metadata=user.metadata
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user by ID."""
    logger.info(f"Getting user: {user_id}")
    
    user = UserOperations.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=user.user_id,
        created_at=user.created_at.isoformat(),
        preferences=user.preferences,
        metadata=user.metadata
    )
