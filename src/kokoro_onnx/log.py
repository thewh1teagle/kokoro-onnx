"""
Provide a way to enable logging by setting LOG_LEVEL environment variable
"""

import logging
import os
import colorlog

def _create_logger():
    """
    Create a logger with colorized output
    Usage: LOG_LEVEL=DEBUG python <script.py>
    """
    
    handler = colorlog.StreamHandler()
    fmt = "%(log_color)s%(levelname)-8s%(reset)s [%(filename)s:%(lineno)d] %(message)s"
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt=fmt,
            log_colors={
                "DEBUG": "blue",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )
    # Get log level from LOG_LEVEL environment variable
    log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
    logger = colorlog.getLogger(__package__)
    logger.setLevel(level=getattr(logging, log_level, logging.WARNING))
    # Setup logging to stdout
    logger.addHandler(handler)
    return logger
    
log = _create_logger()