"""Logging configuration for business entity resolution pipeline."""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "business_entity_resolution", level: Optional[str] = None) -> logging.Logger:
    """Sets up a logger with consistent formatting and handlers.

    Args:
        name: Name of the logger.
        level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
