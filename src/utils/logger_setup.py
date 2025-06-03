import logging
from loguru import logger
from pathlib import Path
import yaml
from datetime import datetime

import sys

def setup_logging(config: dict):
    """
    Set up logging configuration using loguru.
    
    Args:
        config: Dictionary containing logging configuration
    """
    try:
        # Set up log file rotation
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "hedgefund.log"
        
        # Configure loguru
        logger.remove()  # Remove default handler
        
        # Add file handler with rotation
        logger.add(
            str(log_file),
            rotation="00:00",
            retention="10 days",
            format=config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        )
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=config.get('console_format', "{time:HH:mm:ss} | {level} | {message}")
        )
        
        logger.info("Logging system initialized")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise
    
    return logger


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        loguru.Logger instance
    """
    return logger
