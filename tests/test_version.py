"""Integration test for the ``GET /version`` endpoint in :mod:`translator.main`."""

from fastapi.testclient import TestClient


def test_version_returns_app_version(client: TestClient) -> None:
    """GET /api/v1/version returns a non-empty version string."""
    resp = client.get("/api/v1/version")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["version"], str) and body["version"]
