"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from src.core.logging import get_logger

logger = get_logger("api.health")
router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0"
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for load balancers."""
    # TODO: Add database connectivity check
    return {"status": "ready"}
