"""FastAPI application factory: proxy routes + control API + bundled static UI.

Phase 1 wires ``/healthz`` and the record/passthrough proxy. Later phases mount
``server.api`` (the UI's REST surface) and the built ``ui/`` bundle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from .. import __version__
from ..config import Settings, load_settings
from ..core.recorder import RunRouter
from ..core.store import Store
from . import proxy


def create_app(settings: Settings | None = None, *, store: Store | None = None) -> FastAPI:
    """Build the ASGI app. Pass ``store`` to bind an already-open database (tests)."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_store = store is None
        app.state.store = store or Store.open(settings.db_path)
        app.state.http = httpx.AsyncClient(timeout=proxy.UPSTREAM_TIMEOUT, follow_redirects=True)
        app.state.run_router = RunRouter(app.state.store, idle_timeout_s=settings.idle_timeout_s)
        try:
            yield
        finally:
            # Runs grouped heuristically have no other end signal; shutdown is theirs.
            app.state.run_router.close_all()
            await app.state.http.aclose()
            if owns_store:
                app.state.store.close()
            app.state.store = None
            app.state.run_router = None

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

    # Registered last so the catch-all provider mounts cannot shadow named routes.
    app.include_router(proxy.build_router(settings))
    return app
