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
from fastapi import HTTPException, Request

from translator.identity import (
    DEFAULT_IDENTITY_ENV,
    EDGE_SECRET_ENV,
    MIN_EDGE_SECRET_LENGTH,
    Identity,
    Principal,
    get_default_identity,
    get_edge_secret,
    resolve_identity,
    resolve_principal,
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


# ── resolve_principal ──────────────────────────────────────────────────────────

_DEV_IDENTITY = "zz-dev-user"


def _translate_request(headers: list[tuple[bytes, bytes]]) -> Request:
    """Build a ``POST /api/v1/translate`` request carrying exactly these headers.

    Args:
        headers: Header name/value pairs as raw bytes, as they arrive on the wire.

    Returns:
        Request: The request as Starlette hands it to a dependency.
    """
    return Request({"type": "http", "method": "POST", "path": "/api/v1/translate", "headers": headers})


_ALEX = Principal(name="alex", from_gateway=True)
_DEV = Principal(name=_DEV_IDENTITY, from_gateway=False)
_VERIFIED = [(b"x-edge-secret", _SECRET.encode()), (b"x-auth-user", b"alex")]
_WRONG_SECRET = [(b"x-edge-secret", b"zz-synthetic-wrong-secret"), (b"x-auth-user", b"alex")]
_NO_SECRET = [(b"x-auth-user", b"alex")]


@pytest.mark.parametrize(
    ("secret", "default_identity", "headers", "expected"),
    [
        # No edge secret: identity headers are trusted as sent.
        (None, None, [], None),
        (None, None, [(b"x-auth-user", b"")], None),
        (None, None, _NO_SECRET, _ALEX),
        (None, _DEV_IDENTITY, [], _DEV),
        (None, _DEV_IDENTITY, _NO_SECRET, _ALEX),
        # Edge secret set: only edge-verified headers count, and the default identity never does.
        (_SECRET, None, [], None),
        (_SECRET, None, _VERIFIED, _ALEX),
        (_SECRET, None, _WRONG_SECRET, None),
        (_SECRET, None, _NO_SECRET, None),
        (_SECRET, _DEV_IDENTITY, [], None),
        (_SECRET, _DEV_IDENTITY, _VERIFIED, _ALEX),
        (_SECRET, _DEV_IDENTITY, _WRONG_SECRET, None),
        (_SECRET, _DEV_IDENTITY, _NO_SECRET, None),
    ],
    ids=[
        "no-secret/no-default/no-header",
        "no-secret/no-default/empty-header",
        "no-secret/no-default/header",
        "no-secret/default/no-header",
        "no-secret/default/header-wins",
        "secret/no-default/no-header",
        "secret/no-default/verified",
        "secret/no-default/forged-wrong-secret",
        "secret/no-default/forged-no-secret",
        "secret/default/no-header",
        "secret/default/verified",
        "secret/default/forged-wrong-secret",
        "secret/default/forged-no-secret",
    ],
)
def test_principal_for_every_combination(
    set_edge_secret: Callable[[str | None], None],
    set_default_identity: Callable[[str | None], None],
    secret: str | None,
    default_identity: str | None,
    headers: list[tuple[bytes, bytes]],
    expected: Principal | None,
) -> None:
    """Edge secret (set or not) x default identity (set or not) x identity headers, incl. forged ones.

    The default identity stands in only while no edge secret is set; with
    one set, a request without an edge-verified identity always gets 401.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
        set_default_identity: Fixture that sets or unsets the default identity.
        secret: The configured edge secret, or ``None``.
        default_identity: The configured default identity, or ``None``.
        headers: The request's headers.
        expected: The resolved principal, or ``None`` for a 401.
    """
    set_edge_secret(secret)
    set_default_identity(default_identity)
    request = _translate_request(headers)
    if expected is None:
        with pytest.raises(HTTPException) as excinfo:
            resolve_principal(request)
        assert excinfo.value.status_code == 401
    else:
        assert resolve_principal(request) == expected


def test_blank_default_identity_counts_as_unset(set_default_identity: Callable[[str | None], None]) -> None:
    """A whitespace-only default identity is treated as unset.

    Args:
        set_default_identity: Fixture that sets or unsets the default identity.
    """
    set_default_identity("   ")
    assert get_default_identity() is None


def test_default_identity_is_trimmed_and_announced_once(
    set_default_identity: Callable[[str | None], None], loguru_warnings: list[str]
) -> None:
    """A default identity in effect is trimmed, and one warning names the variable but not the account.

    Args:
        set_default_identity: Fixture that sets or unsets the default identity.
        loguru_warnings: Fixture capturing Loguru warnings.
    """
    set_default_identity(f"  {_DEV_IDENTITY}\n")
    assert get_default_identity() == _DEV_IDENTITY
    assert get_default_identity() == _DEV_IDENTITY
    announced = [message for message in loguru_warnings if DEFAULT_IDENTITY_ENV in message]
    assert len(announced) == 1
    assert f"{DEFAULT_IDENTITY_ENV} is set" in announced[0]
    assert _DEV_IDENTITY not in announced[0]


def test_default_identity_is_ignored_with_one_warning_while_the_edge_secret_is_set(
    set_edge_secret: Callable[[str | None], None],
    set_default_identity: Callable[[str | None], None],
    loguru_warnings: list[str],
) -> None:
    """With both set, the default identity is not in effect, and one warning says why.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.
        set_default_identity: Fixture that sets or unsets the default identity.
        loguru_warnings: Fixture capturing Loguru warnings.
    """
    set_edge_secret(_SECRET)
    set_default_identity(_DEV_IDENTITY)
    assert get_default_identity() is None
    assert get_default_identity() is None
    assert len(loguru_warnings) == 1
    assert f"{DEFAULT_IDENTITY_ENV} is ignored because {EDGE_SECRET_ENV} is set" in loguru_warnings[0]
    for value in (_DEV_IDENTITY, _SECRET):
        assert value not in loguru_warnings[0]
