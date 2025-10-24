"""Database operations."""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from src.core.models import (
    User, ConversationSession, ConversationMessage, 
    ModelResponse, TrainingExample, CommunicationStyle
)
from src.database.connection import get_cursor
from src.core.logging import get_logger

logger = get_logger("database.operations")


class UserOperations:
    """User database operations."""
    
    @staticmethod
    def create_user(user: User) -> bool:
        """Create a new user."""
        try:
            with get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO users (user_id, created_at, preferences, metadata)
                    VALUES (%s, %s, %s, %s)
                """, (
                    user.user_id,
                    user.created_at,
                    json.dumps(user.preferences),
                    json.dumps(user.metadata)
                ))
                logger.info(f"Created user: {user.user_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to create user {user.user_id}: {e}")
            return False
    
    @staticmethod
    def get_user(user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            with get_cursor() as cursor:
                cursor.execute("""
                    SELECT user_id, created_at, preferences, metadata
                    FROM users WHERE user_id = %s
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return User(
                        user_id=row[0],
                        created_at=row[1],
                        preferences=row[2],
                        metadata=row[3]
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None


class ConversationOperations:
    """Conversation database operations."""
    
    @staticmethod
    def create_session(session: ConversationSession) -> bool:
        """Create a new conversation session."""
        try:
            with get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO conversation_sessions (session_id, user_id, created_at, updated_at, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    session.session_id,
                    session.user_id,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.metadata)
                ))
                logger.info(f"Created session: {session.session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to create session {session.session_id}: {e}")
            return False
    
    @staticmethod
    def add_message(session_id: str, message: ConversationMessage) -> Optional[int]:
        """Add message to conversation session."""
        try:
            with get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO conversation_messages (session_id, role, content, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING message_id
                """, (
                    session_id,
                    message.role,
                    message.content,
                    message.timestamp,
                    json.dumps(message.metadata)
                ))
                
                message_id = cursor.fetchone()[0]
                logger.info(f"Added message to session {session_id}: {message_id}")
                return message_id
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return None
    
    @staticmethod
    def get_session(session_id: str) -> Optional[ConversationSession]:
        """Get conversation session with messages."""
        try:
            with get_cursor() as cursor:
                # Get session info
                cursor.execute("""
                    SELECT session_id, user_id, created_at, updated_at, metadata
                    FROM conversation_sessions WHERE session_id = %s
                """, (session_id,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    return None
                
                # Get messages
                cursor.execute("""
                    SELECT role, content, timestamp, metadata
                    FROM conversation_messages 
                    WHERE session_id = %s 
                    ORDER BY timestamp
                """, (session_id,))
                
                messages = [
                    ConversationMessage(
                        role=row[0],
                        content=row[1],
                        timestamp=row[2],
                        metadata=row[3]
                    )
                    for row in cursor.fetchall()
                ]
                
                return ConversationSession(
                    session_id=session_row[0],
                    user_id=session_row[1],
                    messages=messages,
                    created_at=session_row[2],
                    updated_at=session_row[3],
                    metadata=session_row[4]
                )
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None


class TrainingOperations:
    """Training data operations."""
    
    @staticmethod
    def add_training_example(example: TrainingExample) -> bool:
        """Add training example."""
        try:
            with get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO training_examples (text, style, source, quality_score, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    example.text,
                    example.style.value,
                    example.source,
                    example.quality_score,
                    json.dumps(example.metadata)
                ))
                logger.info(f"Added training example from {example.source}")
                return True
        except Exception as e:
            logger.error(f"Failed to add training example: {e}")
            return False
    
    @staticmethod
    def get_training_examples(style: Optional[CommunicationStyle] = None, 
                            limit: Optional[int] = None) -> List[TrainingExample]:
        """Get training examples, optionally filtered by style."""
        try:
            with get_cursor() as cursor:
                query = """
                    SELECT text, style, source, quality_score, metadata
                    FROM training_examples
                """
                params = []
                
                if style:
                    query += " WHERE style = %s"
                    params.append(style.value)
                
                query += " ORDER BY created_at DESC"
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                cursor.execute(query, params)
                
                return [
                    TrainingExample(
                        text=row[0],
                        style=CommunicationStyle(row[1]),
                        source=row[2],
                        quality_score=row[3],
                        metadata=row[4]
                    )
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to get training examples: {e}")
            return []


def initialize_database():
    """Initialize database schema."""
    try:
        with get_cursor() as cursor:
            # Read and execute schema
            schema_path = "src/database/schema.sql"
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            cursor.execute(schema_sql)
            logger.info("Database schema initialized")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False
