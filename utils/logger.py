"""
Logging Utility for Deepfake Detection Pipeline

Saves logs to file and optionally to console.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "deepfake_detection",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name.
        log_file: Path to log file. If None, only console logging.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
