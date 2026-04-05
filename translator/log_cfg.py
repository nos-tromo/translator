"""Loguru logger configuration for the Translator service.

Configures two sinks on every call to :func:`setup_logger`:

* **stderr** — INFO-level structured log lines for live monitoring.
* **rotating file** — DEBUG-level lines with backtrace support for post-mortem
  analysis. The log path defaults to ``.log/translator.log`` relative to the
  project root and can be overridden via the ``LOG_PATH`` environment variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).parents[1].resolve()
DEFAULT_LOG_DIR: Path = PROJECT_ROOT / ".log" / "translator.log"
LOG_PATH: Path = Path(os.getenv("LOG_PATH", DEFAULT_LOG_DIR))


def setup_logger(
    encoding="utf-8",
    rotation: str = "5 MB",
    retention: int = 3,
    backtrace: bool = False,
    diagnose: bool = False,
) -> Path:
    """Configure Loguru sinks for the application.

    Removes any existing Loguru handlers, then attaches a stderr sink (INFO+)
    and a rotating file sink (DEBUG+). The log directory is created if it does
    not already exist.

    Args:
        encoding: Character encoding for the log file. Defaults to ``"utf-8"``.
        rotation: Loguru rotation policy for the file sink (e.g. ``"5 MB"`` or
            ``"1 day"``). Defaults to ``"5 MB"``.
        retention: Number of rotated log files to keep. Defaults to ``3``.
        backtrace: Whether to include a full backtrace in error entries for the
            stderr sink. Defaults to ``False``.
        diagnose: Whether to include variable values in tracebacks. Defaults to
            ``False``.

    Returns:
        The resolved path to the active log file.
    """
    log_path = LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level="INFO",
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )

    logger.add(
        sink=log_path,
        rotation=rotation,
        retention=retention,
        encoding=encoding,
        level="DEBUG",
        backtrace=True,
        diagnose=diagnose,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {line:<4} | {name} | {message}",
    )

    return log_path
