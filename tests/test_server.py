from __future__ import annotations

from fastapi.testclient import TestClient

from agentvcr import __version__
from agentvcr.config import Settings
from agentvcr.core.store import SCHEMA_VERSION
from agentvcr.server.app import create_app


def test_healthz(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__
        assert body["mode"] == "record"
        assert body["schema_version"] == SCHEMA_VERSION
        assert body["upstreams"]["openai"].startswith("https://")
    assert settings.db_path.exists()
