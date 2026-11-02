"""Logging configuration for the Translator service.

Installs a single Loguru stderr sink on every call to :func:`setup_logger`
and strips query strings from uvicorn's access log. The container logging
driver owns log retention and rotation (see ``docker/compose.yaml``).
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from typing_extensions import override

load_dotenv()

UVICORN_ACCESS_LOGGER = "uvicorn.access"


class StripQueryStringFilter(logging.Filter):
    """Cut the query string off the request path in uvicorn access-log records.

    uvicorn logs every request with ``record.args`` set to
    ``(client_addr, method, path_with_query, http_version, status)``. A query
    string can carry request data, so the path is truncated at its first
    ``?`` before any handler formats the record. Records of another shape
    pass through untouched, and no record is ever dropped.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the path argument of an access-log record in place.

        Args:
            record: The log record about to be handled.

        Returns:
            Always ``True``, so the record is still emitted.
        """
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            path = args[2]
            if isinstance(path, str) and "?" in path:
                record.args = (*args[:2], path.partition("?")[0], *args[3:])
        return True


def setup_logger(
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """Configure logging for the application.

    Removes any existing Loguru handlers, then attaches a single stderr
    sink. ``LOG_LEVEL`` selects the minimum level (default ``INFO``). Also
    attaches :class:`StripQueryStringFilter` to uvicorn's access logger,
    once no matter how often this runs. ``translator.main`` calls this at
    import time, which uvicorn reaches after applying its own logging
    config, so the filter is in place before the first request.

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

    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER)
    if not any(isinstance(f, StripQueryStringFilter) for f in access_logger.filters):
        access_logger.addFilter(StripQueryStringFilter())
