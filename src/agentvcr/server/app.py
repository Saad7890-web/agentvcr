"""FastAPI application factory: proxy routes + control API + bundled static UI.

One process serves all three, on one port: the proxy an agent points its ``base_url``
at, the ``/api`` the web UI reads, and ``/ui`` itself. The UI is a static bundle built
from ``ui/`` and shipped inside the wheel, so ``pip install agentvcr`` needs no Node —
and a checkout that has not built it says so instead of 404ing.

Route order matters. The proxy mounts a catch-all under each provider prefix and under
``/r/<run>/…``, so it is registered last and cannot shadow ``/healthz``, ``/api`` or
``/ui``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Settings, load_settings
from ..core.recorder import RunRouter
from ..core.store import Store
from . import api, proxy
from .jobs import JobRunner

#: Where the built web UI lands. Written by ``npm --prefix ui run build`` and packaged
#: into the wheel as an artifact; never committed (PLAN.md phase 5).
UI_DIST = Path(__file__).resolve().parent.parent / "ui_dist"

UI_NOT_BUILT = """<!doctype html>
<html><head><title>agentvcr — UI not built</title>
<style>body{font:16px/1.6 system-ui,sans-serif;margin:4rem auto;max-width:40rem;padding:0 1rem}
code{background:#eee;padding:.15em .4em;border-radius:4px}</style></head>
<body><h1>The agentvcr UI is not built</h1>
<p>This looks like a source checkout. The UI ships prebuilt in the wheel; from a
checkout, build it once:</p>
<pre><code>npm --prefix ui install
npm --prefix ui run build</code></pre>
<p>Then reload this page. The API it talks to is up either way — try
<a href="/api/runs">/api/runs</a> — and every command
(<code>agentvcr runs</code>, <code>show</code>, <code>fork</code>, <code>diff</code>)
works without it.</p></body></html>"""


def create_app(settings: Settings | None = None, *, store: Store | None = None) -> FastAPI:
    """Build the ASGI app. Pass ``store`` to bind an already-open database (tests)."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_store = store is None
        app.state.store = store or Store.open(settings.db_path)
        app.state.http = httpx.AsyncClient(timeout=proxy.UPSTREAM_TIMEOUT, follow_redirects=True)
        app.state.run_router = RunRouter(app.state.store, idle_timeout_s=settings.idle_timeout_s)
        app.state.jobs = JobRunner()
        try:
            yield
        finally:
            # An agent the UI started must not outlive the server that started it.
            app.state.jobs.stop_all()
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

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse("/ui/")

    app.include_router(api.router)
    _mount_ui(app)
    # Registered last so the catch-all provider mounts cannot shadow named routes.
    app.include_router(proxy.build_router(settings))
    return app


class _Bundle(StaticFiles):
    """The built UI, with the one cache rule a hashed bundle needs.

    Asset filenames carry a content hash, so they can be cached forever. ``index.html``
    is the one file whose URL never changes — and a browser that reuses a stale copy
    after an upgrade asks for asset files that are no longer on disk, which is a blank
    page rather than an old one. So it is served ``no-cache``: revalidate, every time.
    """

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[no-untyped-def]
        response = super().file_response(full_path, stat_result, scope, status_code)
        immutable = "/assets/" in str(full_path).replace("\\", "/")
        response.headers["cache-control"] = (
            "public, max-age=31536000, immutable" if immutable else "no-cache"
        )
        return response


def _mount_ui(app: FastAPI) -> None:
    """Serve the built UI at ``/ui``, or explain how to build it."""
    if (UI_DIST / "index.html").is_file():
        app.mount("/ui", _Bundle(directory=UI_DIST, html=True), name="ui")
        return

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/{_path:path}", include_in_schema=False)
    async def ui_not_built(_path: str = "") -> HTMLResponse:
        return HTMLResponse(UI_NOT_BUILT, status_code=501)
