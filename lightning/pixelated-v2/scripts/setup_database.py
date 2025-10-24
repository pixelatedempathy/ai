#!/usr/bin/env python3
"""Database setup script."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.operations import initialize_database
from src.database.connection import initialize_pool, close_pool
from src.core.logging import get_logger

logger = get_logger("setup")

def main():
    """Set up database."""
    try:
        logger.info("Setting up database...")
        
        # Initialize connection pool
        initialize_pool()
        
        # Initialize schema
        if initialize_database():
            logger.info("✅ Database setup complete")
        else:
            logger.error("❌ Database setup failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    finally:
        close_pool()

if __name__ == "__main__":
    main()
