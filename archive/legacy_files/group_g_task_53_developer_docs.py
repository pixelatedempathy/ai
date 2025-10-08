#!/usr/bin/env python3
"""
GROUP G: TASK 53 - Write Developer Documentation
Create comprehensive technical documentation for developers.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TASK_53 - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_developer_documentation():
    """Create Task 53: Developer Documentation."""
    logger.info("📚 Creating Task 53: Developer Documentation")
    
    try:
        timestamp = datetime.now().isoformat()
        developer_docs = f'''# Pixelated Empathy AI - Developer Documentation

**Version:** 2.0.0  
**Last Updated:** {timestamp}  
**Target Audience:** Software Developers, DevOps Engineers, System Integrators

## Table of Contents

- [Getting Started](#getting-started)
- [Architecture Overview](#architecture-overview)
- [Development Environment](#development-environment)
- [API Development](#api-development)
- [Database Integration](#database-integration)
- [Authentication & Security](#authentication--security)
- [Testing Framework](#testing-framework)
- [Deployment Pipeline](#deployment-pipeline)
- [Monitoring & Logging](#monitoring--logging)
- [Contributing Guidelines](#contributing-guidelines)

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend components)
- **Docker & Docker Compose**
- **Redis** (for caching and rate limiting)
- **PostgreSQL** (for production database)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/pixelated-empathy/ai-platform.git
cd ai-platform

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/init_db.py

# Start development server
python -m uvicorn pixel_voice.api.server:app --reload --host 0.0.0.0 --port 8000
```

---

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   Core Engine   │
│   (React/Next)  │◄──►│   (FastAPI)     │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Static Files  │    │   Rate Limiter  │    │   ML Models     │
│   (CDN)         │    │   (Redis)       │    │   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Technologies

- **Backend**: FastAPI, Python 3.11+, Pydantic
- **Database**: PostgreSQL (production), SQLite (development)
- **Caching**: Redis with persistence
- **Authentication**: JWT tokens, API keys
- **Rate Limiting**: Redis-based with SlowAPI
- **Monitoring**: Prometheus metrics, structured logging
- **Deployment**: Docker, Kubernetes, Helm charts

---

## Development Environment

### Project Structure

```
pixelated-empathy-ai/
├── pixel_voice/              # Main application package
│   ├── api/                  # FastAPI application
│   │   ├── server.py         # Main server application
│   │   ├── auth.py           # Authentication logic
│   │   ├── rate_limiting.py  # Rate limiting implementation
│   │   ├── models.py         # Pydantic models
│   │   └── utils.py          # Utility functions
│   ├── core/                 # Core business logic
│   ├── models/               # ML model implementations
│   └── utils/                # Shared utilities
├── docs/                     # Documentation
├── tests/                    # Test suites
├── scripts/                  # Utility scripts
├── docker/                   # Docker configurations
└── k8s/                      # Kubernetes manifests
```

### Environment Configuration

Create `.env` file with required variables:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/pixelated_ai
DATABASE_URL_DEV=sqlite:///./dev.db

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# External Services
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# Monitoring
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

---

## API Development

### Creating New Endpoints

1. **Define Pydantic Models** (in `models.py`):

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ConversationRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    context: Optional[str] = None
    user_id: str = Field(..., min_length=1)
    
class ConversationResponse(BaseModel):
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    metadata: dict = {}
```

2. **Implement Endpoint** (in `server.py`):

```python
@app.post("/conversation", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        # Process the conversation
        result = await process_conversation(
            message=request.message,
            context=request.context,
            user_id=request.user_id
        )
        
        processing_time = time.time() - start_time
        
        return ConversationResponse(
            response=result.response,
            confidence=result.confidence,
            processing_time=processing_time,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Conversation processing failed: {{e}}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

3. **Add Authentication** (if required):

```python
from .auth import get_current_user, RequireRole, UserRole

@app.post("/admin/users")
async def create_user(
    user_data: UserCreateRequest,
    current_user: User = Depends(RequireRole(UserRole.ADMIN))
):
    # Admin-only endpoint
    pass
```

### Error Handling Best Practices

```python
from fastapi import HTTPException
from typing import Union

class APIError(Exception):
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={{
            "error": {{
                "code": exc.error_code or "GENERIC_ERROR",
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }}
        }}
    )
```

---

## Database Integration

### Database Models

Using SQLAlchemy with async support:

```python
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)
```

### Database Operations

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def create_conversation(
    db: AsyncSession,
    user_id: str,
    message: str,
    response: str,
    confidence: float
) -> Conversation:
    conversation = Conversation(
        user_id=user_id,
        message=message,
        response=response,
        confidence=confidence
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation

async def get_user_conversations(
    db: AsyncSession,
    user_id: str,
    limit: int = 50
) -> List[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()
```

---

## Authentication & Security

### JWT Implementation

```python
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({{"exp": expire}})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### API Key Authentication

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Validate API key against database
    user = await validate_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return user
```

---

## Testing Framework

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from pixel_voice.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_conversation_endpoint():
    # Mock authentication
    headers = {{"Authorization": "Bearer test_token"}}
    
    response = client.post(
        "/conversation",
        json={{
            "message": "Hello, how are you?",
            "user_id": "test_user"
        }},
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "confidence" in data
```

### Integration Tests

```python
import pytest
from httpx import AsyncClient
from pixel_voice.api.server import app

@pytest.mark.asyncio
async def test_full_conversation_flow():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 1. Authenticate
        auth_response = await ac.post("/auth/login", json={{
            "username": "test_user",
            "password": "test_password"
        }})
        token = auth_response.json()["access_token"]
        
        # 2. Create conversation
        headers = {{"Authorization": f"Bearer {{token}}"}}
        conv_response = await ac.post(
            "/conversation",
            json={{"message": "Test message", "user_id": "test_user"}},
            headers=headers
        )
        
        assert conv_response.status_code == 200
```

---

## Deployment Pipeline

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "pixel_voice.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/pixelated_ai
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
    command: uvicorn pixel_voice.api.server:app --reload --host 0.0.0.0

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: pixelated_ai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

---

## Monitoring & Logging

### Structured Logging

```python
import structlog
import logging.config

# Configure structured logging
logging.config.dictConfig({{
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {{
        "json": {{
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
        }},
    }},
    "handlers": {{
        "default": {{
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "json",
        }},
    }},
    "loggers": {{
        "": {{
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        }},
    }}
}})

logger = structlog.get_logger(__name__)

# Usage
logger.info("Processing conversation", user_id="123", message_length=50)
logger.error("Database connection failed", error=str(e), retry_count=3)
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active WebSocket connections')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Contributing Guidelines

### Code Style

- **Python**: Follow PEP 8, use Black formatter
- **Type Hints**: Required for all function signatures
- **Docstrings**: Use Google-style docstrings
- **Import Order**: Use isort for consistent imports

### Git Workflow

1. **Create Feature Branch**: `git checkout -b feature/new-endpoint`
2. **Make Changes**: Follow coding standards
3. **Write Tests**: Ensure >90% test coverage
4. **Run Linting**: `black . && isort . && flake8`
5. **Run Tests**: `pytest tests/`
6. **Create PR**: Include description and test results
7. **Code Review**: Address feedback
8. **Merge**: Squash and merge to main

### Development Commands

```bash
# Code formatting
black pixel_voice/ tests/
isort pixel_voice/ tests/

# Linting
flake8 pixel_voice/ tests/
mypy pixel_voice/

# Testing
pytest tests/ --cov=pixel_voice --cov-report=html

# Security scanning
bandit -r pixel_voice/

# Dependency checking
safety check
```

---

## Troubleshooting

### Common Issues

**Database Connection Errors**:
```bash
# Check database status
docker-compose ps db

# View database logs
docker-compose logs db

# Reset database
docker-compose down -v
docker-compose up db
```

**Redis Connection Issues**:
```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
docker-compose logs redis
```

**Authentication Problems**:
```bash
# Verify JWT secret is set
echo $JWT_SECRET_KEY

# Check token expiration
python -c "import jwt; print(jwt.decode('your_token', verify=False))"
```

### Performance Optimization

- **Database**: Use connection pooling, add indexes
- **Caching**: Implement Redis caching for frequent queries
- **API**: Use async/await for I/O operations
- **Monitoring**: Set up alerts for response time and error rates

---

## Additional Resources

- **API Documentation**: `/docs` (Swagger UI)
- **Database Schema**: `docs/database_schema.md`
- **Deployment Guide**: `docs/deployment_guide.md`
- **Security Guidelines**: `docs/security_documentation.md`

---

*Last updated: {timestamp}*
*For questions or support, contact the development team or create an issue on GitHub.*
'''
        
        # Write developer documentation
        docs_path = Path('/home/vivi/pixelated/ai/docs/developer_documentation.md')
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(docs_path, 'w') as f:
            f.write(developer_docs)
        
        logger.info("✅ Task 53: Developer Documentation created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Task 53 failed: {e}")
        return False

if __name__ == "__main__":
    success = create_developer_documentation()
    if success:
        print("✅ Task 53: Developer Documentation - COMPLETED")
    else:
        print("❌ Task 53: Developer Documentation - FAILED")
