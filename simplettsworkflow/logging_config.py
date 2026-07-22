from __future__ import annotations

import logging
import os
import sys


LOGGER_NAME = "simplettsworkflow"
DEFAULT_LOG_LEVEL = "INFO"


def configure_application_logging() -> None:
    """Make application logs visible whether started by main.py or Uvicorn CLI."""
    package_logger = logging.getLogger(LOGGER_NAME)
    if getattr(package_logger, "_simpletts_configured", False):
        return

    requested_level = os.getenv("SIMPLETTS_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, requested_level, None)
    invalid_level = not isinstance(level, int)
    if invalid_level:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False
    package_logger._simpletts_configured = True  # type: ignore[attr-defined]

    package_logger.info(
        "Application logging configured: level=%s pid=%s",
        logging.getLevelName(level),
        os.getpid(),
    )
    if invalid_level:
        package_logger.warning(
            "Invalid SIMPLETTS_LOG_LEVEL=%r; falling back to %s",
            requested_level,
            DEFAULT_LOG_LEVEL,
        )
