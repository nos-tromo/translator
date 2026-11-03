"""Tests for :mod:`translator.log_cfg` — the access-log query-string filter.

All values are synthetic. The end-to-end test serves the app with a real
uvicorn server on an ephemeral loopback port, so it checks the record shape
uvicorn actually emits rather than an assumed one.
"""

import io
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import uvicorn
from uvicorn.logging import AccessFormatter

from translator.log_cfg import UVICORN_ACCESS_LOGGER, StripQueryStringFilter, setup_logger
from translator.main import app

_CLIENT = "192.0.2.10:50000"
_SYNTHETIC_QUERY = "probe=zz-synthetic-query-4817"


def _access_record(args: tuple[object, ...] | None) -> logging.LogRecord:
    """Build a record as uvicorn's access logger would create it.

    Args:
        args: The record's ``%``-format arguments.

    Returns:
        logging.LogRecord: A record named after uvicorn's access logger.
    """
    return logging.LogRecord(
        name=UVICORN_ACCESS_LOGGER,
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=args,
        exc_info=None,
    )


def test_filter_strips_query_string() -> None:
    """Truncates the path argument at the first ``?`` and keeps the record."""
    record = _access_record((_CLIENT, "GET", f"/api/v1/languages?{_SYNTHETIC_QUERY}", "1.1", 200))
    assert StripQueryStringFilter().filter(record) is True
    assert record.args == (_CLIENT, "GET", "/api/v1/languages", "1.1", 200)


def test_filter_keeps_path_without_query() -> None:
    """Leaves a path without a query string unchanged."""
    args = (_CLIENT, "GET", "/api/v1/health", "1.1", 200)
    record = _access_record(args)
    assert StripQueryStringFilter().filter(record) is True
    assert record.args == args


@pytest.mark.parametrize(
    "args",
    [None, (_CLIENT, "GET"), (_CLIENT, "GET", 4817, "1.1", 200)],
    ids=["no-args", "short-tuple", "non-str-path"],
)
def test_filter_passes_other_records_through(args: tuple[object, ...] | None) -> None:
    """Keeps records of any other shape, unchanged.

    Args:
        args: Record arguments that do not match uvicorn's access-log shape.
    """
    record = _access_record(args)
    assert StripQueryStringFilter().filter(record) is True
    assert record.args == args


def test_access_formatter_renders_no_query_string() -> None:
    """Uvicorn's own access formatter renders the filtered path only."""
    record = _access_record((_CLIENT, "GET", f"/api/v1/languages?{_SYNTHETIC_QUERY}", "1.1", 200))
    StripQueryStringFilter().filter(record)
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(record)
    assert '"GET /api/v1/languages HTTP/1.1" 200' in line
    assert _SYNTHETIC_QUERY not in line


def test_setup_logger_attaches_filter_once() -> None:
    """Repeated setup leaves exactly one filter on uvicorn's access logger."""
    setup_logger()
    setup_logger()
    filters = logging.getLogger(UVICORN_ACCESS_LOGGER).filters
    assert sum(isinstance(f, StripQueryStringFilter) for f in filters) == 1


def test_uvicorn_access_log_omits_query_string(caplog: pytest.LogCaptureFixture) -> None:
    """A real uvicorn server logs the request path without its query string.

    Args:
        caplog: Pytest fixture capturing log records.
    """
    caplog.set_level(logging.INFO, logger=UVICORN_ACCESS_LOGGER)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                pytest.fail("uvicorn did not start within 10 s")
            time.sleep(0.05)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/api/v1/version?{_SYNTHETIC_QUERY}", timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()

    access_lines = [r.getMessage() for r in caplog.records if r.name == UVICORN_ACCESS_LOGGER]
    assert any('"GET /api/v1/version HTTP/1.1" 200' in line for line in access_lines)
    assert not any(_SYNTHETIC_QUERY in line for line in access_lines)


def test_setup_logger_moves_stdout_handlers_to_stderr() -> None:
    """Stdlib handlers on stdout end up on stderr; handlers on other streams keep theirs."""
    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER)
    on_stdout = logging.StreamHandler(sys.stdout)
    elsewhere = logging.StreamHandler(io.StringIO())
    other_stream = elsewhere.stream
    for handler in (on_stdout, elsewhere):
        access_logger.addHandler(handler)
    try:
        setup_logger()
    finally:
        for handler in (on_stdout, elsewhere):
            access_logger.removeHandler(handler)
    assert on_stdout.stream is sys.stderr
    assert elsewhere.stream is other_stream


def test_uvicorn_default_logging_leaves_stdout_clear(tmp_path: Path) -> None:
    """Under uvicorn's default logging config, the app's import moves uvicorn's logs off stdout.

    stdout carries audit events only. uvicorn applies its logging config
    before it imports the app; the child interpreter does the same, then
    logs one access line and one server line.

    Args:
        tmp_path: Empty working directory for the child process.
    """
    script = (
        "import logging, logging.config\n"
        "from uvicorn.config import LOGGING_CONFIG\n"
        "logging.config.dictConfig(LOGGING_CONFIG)\n"
        "import translator.main\n"
        "logging.getLogger('uvicorn.access').info("
        "'%s - \"%s %s HTTP/%s\" %d', '192.0.2.10:50000', 'GET', '/api/v1/health', '1.1', 200)\n"
        "logging.getLogger('uvicorn.error').info('zz-synthetic-server-line')\n"
    )
    env = {
        **os.environ,
        "OPENAI_API_BASE": "http://test-host:11434/v1",
        "TEXT_MODEL": "test-model",
        "TRANSLATOR_EDGE_SECRET": "",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert '"GET /api/v1/health HTTP/1.1" 200' in result.stderr
    assert "zz-synthetic-server-line" in result.stderr
