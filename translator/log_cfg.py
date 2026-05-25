"""Loguru logger configuration for the Translator service.

Installs a single stderr sink on every call to :func:`setup_logger`.
The container logging driver owns log retention and rotation (see
``docker/compose.yaml``).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def setup_logger(
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """Configure Loguru for the application.

    Removes any existing Loguru handlers, then attaches a single stderr
    sink. ``LOG_LEVEL`` selects the minimum level (default ``INFO``).

    Args:
        backtrace: Whether to include a full backtrace in error entries.
            Defaults to ``False``.
        diagnose: Whether to include variable values in tracebacks.
            Defaults to ``False``.
    """
    level = os.getenv("LOG_LEVEL", "INFO")

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )
