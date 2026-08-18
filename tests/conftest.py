from __future__ import annotations

from pathlib import Path

import pytest

from agentvcr.config import Settings
from agentvcr.core.store import Store


@pytest.fixture()
def store() -> Store:
    with Store.open(None) as s:
        yield s


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "agentvcr.db")
