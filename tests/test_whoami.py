"""Integration tests for the ``GET /whoami`` endpoint in :mod:`translator.main`.

The identity headers count only as far as
:func:`translator.identity.resolve_identity` trusts them, so each test that
sends them sets ``TRANSLATOR_EDGE_SECRET`` explicitly. All values are synthetic.
"""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from translator.identity import EDGE_SECRET_HEADER

_SECRET = "zz-synthetic-edge-secret-0123456789abcdef"
_WRONG_SECRET = "zz-synthetic-wrong-secret-0123456789abcde"
_IDENTITY = {"X-Auth-User": "alex", "X-Auth-Name": "Alex Example"}
_TRUSTED: dict[str, str | None] = {"username": "alex", "display_name": "Alex Example"}
_NULLS: dict[str, str | None] = {"username": None, "display_name": None}


def test_whoami_without_edge_secret_echoes_headers(
    client: TestClient, set_edge_secret: Callable[[str | None], None]
) -> None:
    """With no edge secret configured, GET /api/v1/whoami echoes the identity headers.

    Args:
        client: TestClient provided by the ``client`` fixture.
        set_edge_secret: Fixture that sets or unsets the edge secret.
    """
    set_edge_secret(None)
    resp = client.get("/api/v1/whoami", headers=_IDENTITY)
    assert resp.status_code == 200
    assert resp.json() == _TRUSTED


def test_whoami_returns_nulls_when_headers_absent(client: TestClient) -> None:
    """GET /api/v1/whoami returns nulls outside the gateway (dev), not a 401.

    translator has no fail-closed principal seam — this endpoint is
    decorative header-echo only, so a 200 with nulls is correct.
    """
    resp = client.get("/api/v1/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"username": None, "display_name": None}


def test_whoami_trusts_headers_with_valid_edge_secret(
    client: TestClient, set_edge_secret: Callable[[str | None], None]
) -> None:
    """With the edge secret set, a request carrying it gets its identity echoed.

    Args:
        client: TestClient provided by the ``client`` fixture.
        set_edge_secret: Fixture that sets or unsets the edge secret.
    """
    set_edge_secret(_SECRET)
    resp = client.get("/api/v1/whoami", headers={**_IDENTITY, EDGE_SECRET_HEADER: _SECRET})
    assert resp.status_code == 200
    assert resp.json() == _TRUSTED


@pytest.mark.parametrize("sent", [None, _WRONG_SECRET], ids=["missing", "wrong"])
def test_whoami_ignores_headers_without_valid_edge_secret(
    client: TestClient, set_edge_secret: Callable[[str | None], None], sent: str | None
) -> None:
    """With the edge secret set, identity headers without it are ignored.

    The answer is the same as for a request without identity headers.

    Args:
        client: TestClient provided by the ``client`` fixture.
        set_edge_secret: Fixture that sets or unsets the edge secret.
        sent: The ``X-Edge-Secret`` value sent, or ``None`` to omit the header.
    """
    set_edge_secret(_SECRET)
    headers = _IDENTITY if sent is None else {**_IDENTITY, EDGE_SECRET_HEADER: sent}
    resp = client.get("/api/v1/whoami", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == _NULLS


def test_ignored_identity_warning_names_only_method_and_path(
    client: TestClient, set_edge_secret: Callable[[str | None], None], loguru_warnings: list[str]
) -> None:
    """The warning for ignored identity headers holds no secret, header value or query.

    Args:
        client: TestClient provided by the ``client`` fixture.
        set_edge_secret: Fixture that sets or unsets the edge secret.
        loguru_warnings: Fixture capturing Loguru warnings.
    """
    user, name, query = "zz-synthetic-user-4817", "Zz Synthetic Name 4817", "probe=zz-synthetic-query-4817"
    set_edge_secret(_SECRET)
    client.get(
        f"/api/v1/whoami?{query}",
        headers={"X-Auth-User": user, "X-Auth-Name": name, EDGE_SECRET_HEADER: _WRONG_SECRET},
    )
    logged = "".join(loguru_warnings)
    assert "Ignored identity headers" in logged
    assert "GET /api/v1/whoami" in logged
    for value in (_SECRET, _WRONG_SECRET, user, name, query):
        assert value not in logged
