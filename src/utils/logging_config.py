"""
Structured logging configuration with console and JSON formatting.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "spatial_hmi", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger for the HMI system."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger
