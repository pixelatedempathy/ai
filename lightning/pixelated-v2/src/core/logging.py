"""Logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration."""
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if it doesn't exist
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    return logging.getLogger("pixelated")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"pixelated.{name}")


# Set up default logging
logger = setup_logging(log_file="logs/pixelated.log")
