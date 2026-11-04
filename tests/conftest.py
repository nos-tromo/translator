"""Shared pytest fixtures and session-level configuration for the translator tests.

``OPENAI_API_BASE`` and ``TEXT_MODEL`` are injected into the environment at
import time — before any test module is collected — because
:mod:`translator.main` instantiates a :class:`~translator.engine.Translator` at
module scope, and the constructor requires both variables to be present.
"""

import os

# Must be set before translator.main is imported, as Translator() is
# instantiated at module level and requires OPENAI_API_BASE + TEXT_MODEL.
os.environ.setdefault("OPENAI_API_BASE", "http://test-host:11434/v1")
os.environ.setdefault("TEXT_MODEL", "test-model")

import io
import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from typing_extensions import override

from translator.audit import EMITTER_STATE_KEY, OUTPUT_ENV, REQUIRE_PURPOSE_ENV, SERVICE, get_required_actions
from translator.identity import DEFAULT_IDENTITY_ENV, EDGE_SECRET_ENV, get_default_identity, get_edge_secret
from translator.main import app
from translator.utils.audit_event import AuditEmitter, verify_chain

_PER_PROCESS_SETTINGS = (get_edge_secret, get_default_identity, get_required_actions)


@pytest.fixture(autouse=True)
def _no_config_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test without identity or audit settings.

    ``translator.log_cfg`` loads the repo's ``.env`` at import, so a
    developer's own values would otherwise decide which callers count as
    authenticated and which actions need a stated purpose. Tests that need
    a setting set it with the fixtures below.

    Args:
        monkeypatch: Pytest fixture that restores the environment afterwards.

    Yields:
        None: Control to the test.
    """
    for name in (EDGE_SECRET_ENV, DEFAULT_IDENTITY_ENV, OUTPUT_ENV, REQUIRE_PURPOSE_ENV):
        monkeypatch.delenv(name, raising=False)
    for setting in _PER_PROCESS_SETTINGS:
        setting.cache_clear()
    yield
    for setting in _PER_PROCESS_SETTINGS:
        setting.cache_clear()


class CapturedAuditStream(io.StringIO):
    """An in-memory audit stream for one test.

    Attributes:
        failing: While set, every write fails as a broken output would.
    """

    failing: bool = False

    @override
    def write(self, s: str, /) -> int:
        """Keep the text, or fail while :attr:`failing` is set.

        Args:
            s: The text to write.

        Returns:
            int: The number of characters written.

        Raises:
            OSError: While :attr:`failing` is set.
        """
        if self.failing:
            raise OSError("synthetic audit stream failure")
        return super().write(s)

    def events(self) -> list[dict[str, Any]]:
        """Return every event written so far.

        Returns:
            list[dict[str, Any]]: The parsed lines, in order.
        """
        return [json.loads(line) for line in self.getvalue().splitlines()]

    def operation_events(self) -> list[dict[str, Any]]:
        """Return the events of audited operations, without the chain's lifecycle events.

        Returns:
            list[dict[str, Any]]: The events, in order.
        """
        return [event for event in self.events() if event["action"] != "lifecycle"]


@pytest.fixture
def audit_stream() -> CapturedAuditStream:
    """Return the stream one test's audit events go to.

    Returns:
        CapturedAuditStream: An empty in-memory stream.
    """
    return CapturedAuditStream()


@pytest.fixture
def audit_emitter(audit_stream: CapturedAuditStream, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuditEmitter]:
    """Install a started emitter writing to ``audit_stream`` as the app's emitter for one test.

    Afterwards the chain is closed and verified, so every test that emits
    ends with an intact chain.

    Args:
        audit_stream: The stream the events go to.
        monkeypatch: Pytest fixture that removes the emitter from the app afterwards.

    Yields:
        AuditEmitter: The installed emitter.
    """
    emitter = AuditEmitter(SERVICE, "0+test", stream=audit_stream, heartbeat_interval_s=0)
    emitter.start()
    monkeypatch.setattr(app.state, EMITTER_STATE_KEY, emitter, raising=False)
    yield emitter
    audit_stream.failing = False
    emitter.close()
    report = verify_chain(audit_stream.events())
    assert report.ok, report.problems


@pytest.fixture
def client(audit_emitter: AuditEmitter) -> TestClient:
    """Return a Starlette TestClient wrapping the FastAPI application.

    The app's audit emitter is the ``audit_emitter`` fixture's; the lifespan
    does not run.

    Args:
        audit_emitter: The emitter installed for this test.

    Returns:
        TestClient: Synchronous test client with the full ASGI app mounted.
    """
    return TestClient(app)


@pytest.fixture
def set_edge_secret(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str | None], None]]:
    """Set or unset ``TRANSLATOR_EDGE_SECRET`` for one test.

    The backend reads the secret once per process, so the cached value is
    dropped on every change and again after the test — with the default
    identity, which counts only while no secret is set.

    Args:
        monkeypatch: Pytest fixture that restores the environment afterwards.

    Yields:
        Callable[[str | None], None]: Sets the secret, or unsets it for ``None``.
    """

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(EDGE_SECRET_ENV, raising=False)
        else:
            monkeypatch.setenv(EDGE_SECRET_ENV, value)
        get_edge_secret.cache_clear()
        get_default_identity.cache_clear()

    yield _set
    get_edge_secret.cache_clear()
    get_default_identity.cache_clear()


@pytest.fixture
def set_default_identity(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str | None], None]]:
    """Set or unset ``TRANSLATOR_DEFAULT_IDENTITY`` for one test.

    Read once per process like the edge secret, so the cached value is
    dropped on every change and again after the test.

    Args:
        monkeypatch: Pytest fixture that restores the environment afterwards.

    Yields:
        Callable[[str | None], None]: Sets the identity, or unsets it for ``None``.
    """

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(DEFAULT_IDENTITY_ENV, raising=False)
        else:
            monkeypatch.setenv(DEFAULT_IDENTITY_ENV, value)
        get_default_identity.cache_clear()

    yield _set
    get_default_identity.cache_clear()


@pytest.fixture
def set_required_actions(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str | None], None]]:
    """Set or unset ``AUDIT_REQUIRE_PURPOSE`` for one test.

    Read once per process, so the cached value is dropped on every change
    and again after the test.

    Args:
        monkeypatch: Pytest fixture that restores the environment afterwards.

    Yields:
        Callable[[str | None], None]: Sets the value, or unsets it for ``None``.
    """

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(REQUIRE_PURPOSE_ENV, raising=False)
        else:
            monkeypatch.setenv(REQUIRE_PURPOSE_ENV, value)
        get_required_actions.cache_clear()

    yield _set
    get_required_actions.cache_clear()


@pytest.fixture
def loguru_warnings() -> Iterator[list[str]]:
    """Capture what Loguru logs at WARNING and above during one test.

    Yields:
        list[str]: The captured messages, in the order they were logged.
    """
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)
