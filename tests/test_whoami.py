"""Integration test for the ``GET /whoami`` endpoint in :mod:`translator.main`."""

from fastapi.testclient import TestClient


def test_whoami_returns_headers_when_present(client: TestClient) -> None:
    """GET /api/v1/whoami echoes the trusted identity headers when present."""
    resp = client.get(
        "/api/v1/whoami",
        headers={"X-Auth-User": "alex", "X-Auth-Name": "Alex Example"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"username": "alex", "display_name": "Alex Example"}


def test_whoami_returns_nulls_when_headers_absent(client: TestClient) -> None:
    """GET /api/v1/whoami returns nulls outside the gateway (dev), not a 401.

    translator has no fail-closed principal seam — this endpoint is
    decorative header-echo only, so a 200 with nulls is correct.
    """
    resp = client.get("/api/v1/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"username": None, "display_name": None}
