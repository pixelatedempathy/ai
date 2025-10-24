"""Database connection management."""

import psycopg2
import psycopg2.pool
from contextlib import contextmanager
from typing import Generator, Optional
from src.core.config import config
from src.core.logging import get_logger

logger = get_logger("database")

# Connection pool
_connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def initialize_pool():
    """Initialize database connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        try:
            _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=config.database.host,
                port=config.database.port,
                database=config.database.name,
                user=config.database.user,
                password=config.database.password
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get database connection from pool."""
    if _connection_pool is None:
        initialize_pool()
    
    conn = None
    try:
        conn = _connection_pool.getconn()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            _connection_pool.putconn(conn)


@contextmanager
def get_cursor() -> Generator[psycopg2.extensions.cursor, None, None]:
    """Get database cursor with automatic transaction handling."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise


def close_pool():
    """Close database connection pool."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed")
