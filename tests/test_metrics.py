"""Integration test for the ``GET /metrics`` Prometheus endpoint in :mod:`translator.main`."""

from fastapi.testclient import TestClient


def test_metrics_returns_prometheus_exposition(client: TestClient) -> None:
    """GET /metrics responds 200 with Prometheus exposition-format text.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
