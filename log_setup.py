"""
Handles logging globally
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# New logging config
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "canoe_app.log")

def setup_logging(name: str = "canoe_app") -> logging.Logger:
    """
    Creates or returns a logger with consistent configuration.
    Args:
        name: The logger name (defaults to "canoe_app")
    Returns:
        A configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Only add handlers if none exist (prevents duplicate handlers)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)

        # Rotating file handler
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(module)s:%(lineno)d - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%H:%M:%S")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # Avoid duplicate logs
        logger.propagate = False

    return logger