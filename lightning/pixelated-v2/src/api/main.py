"""FastAPI application main module."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from src.core.config import config
from src.core.logging import get_logger
from src.database.connection import initialize_pool, close_pool
from src.api.routes import conversations, health, users

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Pixelated Empathy API...")
    initialize_pool()
    logger.info("Database connection pool initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    close_pool()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Pixelated Empathy API",
        description="Therapeutic AI with Tim Fletcher communication style",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(users.router, prefix="/users", tags=["users"])
    app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )
