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

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from translator.identity import DEFAULT_IDENTITY_ENV, EDGE_SECRET_ENV, get_default_identity, get_edge_secret
from translator.main import app


@pytest.fixture(autouse=True)
def _no_identity_config_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test without an edge secret or a default identity.

    ``translator.log_cfg`` loads the repo's ``.env`` at import, so a
    developer's own values would otherwise decide which callers count as
    authenticated. Tests that need either set it with the fixtures below.

    Args:
        monkeypatch: Pytest fixture that restores the environment afterwards.

    Yields:
        None: Control to the test.
    """
    for name in (EDGE_SECRET_ENV, DEFAULT_IDENTITY_ENV):
        monkeypatch.delenv(name, raising=False)
    get_edge_secret.cache_clear()
    get_default_identity.cache_clear()
    yield
    get_edge_secret.cache_clear()
    get_default_identity.cache_clear()


@pytest.fixture
def client() -> TestClient:
    """Return a Starlette TestClient wrapping the FastAPI application.

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
def loguru_warnings() -> Iterator[list[str]]:
    """Capture what Loguru logs at WARNING and above during one test.

    Yields:
        list[str]: The captured messages, in the order they were logged.
    """
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)
