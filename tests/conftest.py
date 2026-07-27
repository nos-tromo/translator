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

import pytest
from fastapi.testclient import TestClient

from translator.main import app


@pytest.fixture
def client() -> TestClient:
    """Return a Starlette TestClient wrapping the FastAPI application.

    Returns:
        TestClient: Synchronous test client with the full ASGI app mounted.
    """
    return TestClient(app)
