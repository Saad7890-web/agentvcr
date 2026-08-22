from __future__ import annotations

from pathlib import Path

import pytest

from agentvcr.config import Settings
from agentvcr.core.store import Store

#: Fake upstream every proxy test mocks with respx.
UPSTREAM = "https://upstream.test/v1"


@pytest.fixture()
def store() -> Store:
    with Store.open(None) as s:
        yield s


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "agentvcr.db")


@pytest.fixture()
def proxy(tmp_path: Path):
    """A TestClient over the proxy app, plus the store it writes to."""
    from dataclasses import dataclass

    from fastapi.testclient import TestClient

    from agentvcr.server.app import create_app

    settings = Settings(
        db_path=tmp_path / "agentvcr.db",
        upstreams={"openai": UPSTREAM, "anthropic": "https://anthropic.test"},
    )
    opened = Store.open(settings.db_path)

    @dataclass
    class Harness:
        client: object
        store: Store
        settings: Settings

    # Dialed on the loopback rather than TestClient's default `testserver`: the control
    # API refuses a host it was not bound to, which is what stops a rebound DNS name
    # from reaching it (agentvcr.server.api).
    with TestClient(create_app(settings, store=opened), base_url="http://127.0.0.1:8484") as client:
        yield Harness(client=client, store=opened, settings=settings)
    opened.close()
