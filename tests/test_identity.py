"""Tests for :mod:`translator.identity` — the edge secret and identity resolution.

The ``GET /whoami`` tests cover the request paths end to end; these cover the
secret's startup checks and the edge cases of the comparison. All values are
synthetic.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import Request

from translator.identity import (
    EDGE_SECRET_ENV,
    MIN_EDGE_SECRET_LENGTH,
    Identity,
    get_edge_secret,
    resolve_identity,
)

_SECRET = "zz-synthetic-edge-secret-0123456789abcdef"
_SHORT_SECRET = "zz-synthetic-short-secret"


def _request(headers: list[tuple[bytes, bytes]]) -> Request:
    """Build a ``GET /api/v1/whoami`` request carrying exactly these headers.

    Args:
        headers: Header name/value pairs as raw bytes, as they arrive on the wire.

    Returns:
        Request: The request as Starlette hands it to an endpoint.
    """
    return Request({"type": "http", "method": "GET", "path": "/api/v1/whoami", "headers": headers})


# ── get_edge_secret ────────────────────────────────────────────────────────────


def test_unset_secret_returns_none_and_warns_once(
    set_edge_secret: Callable[[str | None], None], loguru_warnings: list[str]
) -> None:
    """An unset secret selects the legacy behaviour and logs one warning, not one per call.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
        loguru_warnings: Fixture capturing Loguru warnings.
    """
    set_edge_secret(None)
    assert get_edge_secret() is None
    assert get_edge_secret() is None
    assert len(loguru_warnings) == 1
    assert EDGE_SECRET_ENV in loguru_warnings[0]
    assert "identity-contract.md" in loguru_warnings[0]


def test_blank_secret_counts_as_unset(set_edge_secret: Callable[[str | None], None]) -> None:
    """A whitespace-only secret is treated as unset.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
    """
    set_edge_secret("   ")
    assert get_edge_secret() is None


def test_secret_surrounding_whitespace_is_dropped(set_edge_secret: Callable[[str | None], None]) -> None:
    """Whitespace around the secret is not part of it.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
    """
    set_edge_secret(f"  {_SECRET}\n")
    assert get_edge_secret() == _SECRET


def test_secret_of_minimum_length_is_accepted(set_edge_secret: Callable[[str | None], None]) -> None:
    """A secret of exactly the minimum length is accepted.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
    """
    secret = "z" * MIN_EDGE_SECRET_LENGTH
    set_edge_secret(secret)
    assert get_edge_secret() == secret


@pytest.mark.parametrize("secret", [_SHORT_SECRET, "z" * (MIN_EDGE_SECRET_LENGTH - 1)], ids=["short", "one-under"])
def test_short_secret_raises_without_echoing_it(set_edge_secret: Callable[[str | None], None], secret: str) -> None:
    """A secret below the minimum length is refused, and the error does not quote it.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
        secret: The too-short secret.
    """
    set_edge_secret(secret)
    with pytest.raises(ValueError, match=f"{EDGE_SECRET_ENV} must be at least {MIN_EDGE_SECRET_LENGTH}") as excinfo:
        get_edge_secret()
    assert secret not in str(excinfo.value)


def test_short_secret_stops_app_startup(tmp_path: Path) -> None:
    """Importing the app with a too-short secret fails, as uvicorn's import of it would.

    Runs in a fresh interpreter from an empty directory, with every variable
    the import needs set explicitly, so this process's cached secret and any
    local ``.env`` play no part.

    Args:
        tmp_path: Empty working directory for the child process.
    """
    env = {
        **os.environ,
        "OPENAI_API_BASE": "http://test-host:11434/v1",
        "TEXT_MODEL": "test-model",
        EDGE_SECRET_ENV: _SHORT_SECRET,
    }
    result = subprocess.run(
        [sys.executable, "-c", "import translator.main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert f"{EDGE_SECRET_ENV} must be at least {MIN_EDGE_SECRET_LENGTH} characters long" in result.stderr
    assert _SHORT_SECRET not in result.stderr


# ── resolve_identity ───────────────────────────────────────────────────────────


def test_non_ascii_secret_header_is_a_mismatch_not_an_error(set_edge_secret: Callable[[str | None], None]) -> None:
    """A non-ASCII ``X-Edge-Secret`` value is rejected like any wrong value.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
    """
    set_edge_secret(_SECRET)
    request = _request([(b"x-edge-secret", b"\xff" * 40), (b"x-auth-user", b"alex")])
    assert resolve_identity(request) == Identity()


def test_non_ascii_secret_matches_its_utf8_bytes(set_edge_secret: Callable[[str | None], None]) -> None:
    """A secret with non-ASCII characters matches when sent as its UTF-8 bytes.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
    """
    secret = "zz-synthetic-edge-secret-äöü-0123456789"
    set_edge_secret(secret)
    request = _request(
        [(b"x-edge-secret", secret.encode()), (b"x-auth-user", b"alex"), (b"x-auth-name", b"Alex Example")]
    )
    assert resolve_identity(request) == Identity(username="alex", display_name="Alex Example")


def test_unverified_request_without_identity_headers_logs_nothing(
    set_edge_secret: Callable[[str | None], None], loguru_warnings: list[str]
) -> None:
    """A request with neither secret nor identity headers is not worth a warning.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
        loguru_warnings: Fixture capturing Loguru warnings.
    """
    set_edge_secret(_SECRET)
    assert resolve_identity(_request([])) == Identity()
    assert loguru_warnings == []
