"""FastAPI application factory: proxy routes + control API + bundled static UI.

Phase 0 wires only ``/healthz``; later phases mount ``server.proxy`` (record/replay/
fork), ``server.api`` (the UI's REST surface) and the built ``ui/`` bundle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .. import __version__
from ..config import Settings, load_settings
from ..core.store import Store


def create_app(settings: Settings | None = None, *, store: Store | None = None) -> FastAPI:
    """Build the ASGI app. Pass ``store`` to bind an already-open database (tests)."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_store = store is None
        app.state.store = store or Store.open(settings.db_path)
        try:
            yield
        finally:
            if owns_store:
                app.state.store.close()
            app.state.store = None

    app = FastAPI(
        title="agentvcr",
        version=__version__,
        summary="Record, replay, fork and diff agent runs at the LLM boundary.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict[str, Any]:
        current: Store | None = getattr(app.state, "store", None)
        return {
            "status": "ok",
            "version": __version__,
            "mode": settings.mode,
            "db": str(settings.db_path),
            "schema_version": current.schema_version if current else None,
            "upstreams": settings.upstreams,
        }

    return app
