"""The proxy routes: ``/openai/v1/*`` and ``/anthropic/*``.

Each provider is mounted under a prefix that mirrors its upstream base URL, so the
only change an agent makes is its ``base_url``. Paths a provider declares as
``recorded_paths`` are captured as steps; everything else under the mount is forwarded
verbatim (``/v1/models``, and so on).

A request carries its run in one of two ways (DESIGN.md §4). The ``X-AgentVCR-Run``
header is the explicit form for clients that can set default headers. Because most
agents cannot, ``agentvcr run`` instead points the SDK at ``/r/<run-id>/openai/v1`` —
the run travels in the base URL, so an unmodified client carries it for free. Failing
both, the grouping heuristic in :mod:`agentvcr.core.recorder` chains the call onto a
run by message prefix.

Phase 1 implements ``record`` and ``passthrough``; ``replay`` and ``fork`` answer with
a structured 501 until Phases 2 and 4 land.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import Settings
from ..core.recorder import RunRouter, record_step
from ..core.store import Store, utcnow
from ..providers import Provider, get_provider

RUN_HEADER = "x-agentvcr-run"
MODE_HEADER = "x-agentvcr-mode"

MODE_RECORD = "record"
MODE_PASSTHROUGH = "passthrough"
_UNIMPLEMENTED_MODES = {"replay": 2, "fork": 4}

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]

#: Never forwarded upstream: hop-by-hop headers, headers httpx recomputes, and our own.
_HEADERS_NOT_FORWARDED = frozenset(
    {"host", "content-length", "connection", "transfer-encoding", "accept-encoding", "keep-alive"}
)
#: Never passed back to the client: httpx already decoded the body for us.
_HEADERS_NOT_RETURNED = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
)

UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)


def build_router(settings: Settings) -> APIRouter:
    """Mount every registered provider, with and without a run-scoped prefix."""
    router = APIRouter()
    for name in settings.upstreams:
        provider = get_provider(name)
        if provider is None:  # pragma: no cover - Settings.validate rejects these
            continue
        handler = _make_handler(provider)
        mount = provider.mount_path
        router.add_api_route(
            f"{mount}/{{subpath:path}}", handler, methods=_PROXY_METHODS, include_in_schema=False
        )
        router.add_api_route(
            f"/r/{{run_id}}{mount}/{{subpath:path}}",
            handler,
            methods=_PROXY_METHODS,
            include_in_schema=False,
        )
    return router


def _make_handler(provider: Provider):
    async def handle(request: Request, subpath: str = "", run_id: str | None = None) -> Response:
        return await _dispatch(request, provider, subpath, run_id)

    handle.__name__ = f"proxy_{provider.name}"
    return handle


# --------------------------------------------------------------------------- dispatch


async def _dispatch(
    request: Request, provider: Provider, subpath: str, path_run_id: str | None
) -> Response:
    settings: Settings = request.app.state.settings
    store: Store = request.app.state.store
    client: httpx.AsyncClient = request.app.state.http

    explicit_run_id = path_run_id or request.headers.get(RUN_HEADER)
    mode = _resolve_mode(request, store, settings, explicit_run_id)

    unimplemented_phase = _UNIMPLEMENTED_MODES.get(mode)
    if unimplemented_phase is not None:
        return _error(
            501,
            "mode_not_implemented",
            f"mode {mode!r} lands in Phase {unimplemented_phase} (PLAN.md); "
            f"this build implements 'record' and 'passthrough'",
        )

    body = await request.body()
    upstream = settings.upstream_for(provider.name).rstrip("/") + "/" + subpath.lstrip("/")
    is_recorded_path = request.method == "POST" and f"/{subpath.lstrip('/')}" in tuple(
        provider.recorded_paths
    )

    parsed: dict[str, Any] | None = None
    if is_recorded_path and mode == MODE_RECORD:
        parsed = _parse_json_object(body)

    if parsed is None:
        return await _forward(request, client, upstream, body, recorded=False)

    router: RunRouter = request.app.state.run_router
    run = router.resolve(
        provider=provider,
        body=parsed,
        mode=mode,
        run_id=explicit_run_id,
        upstream_url=settings.upstream_for(provider.name),
    )
    if provider.is_streaming(parsed):
        return await _forward_streaming_recorded(
            request, client, upstream, body, provider=provider, run_id=run.id, parsed=parsed
        )
    return await _forward_recorded(
        request, client, upstream, body, provider=provider, run_id=run.id, parsed=parsed
    )


def _resolve_mode(request: Request, store: Store, settings: Settings, run_id: str | None) -> str:
    """Header wins, then the run's own mode, then the server default (DESIGN.md §3)."""
    header = request.headers.get(MODE_HEADER)
    if header:
        return header.strip().lower()
    if run_id:
        run = store.get_run(run_id)
        if run is not None:
            return run.mode
    return settings.mode


# ---------------------------------------------------------------------------- proxying


async def _forward(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    *,
    recorded: bool,
) -> Response:
    """Plain proxy: forward verbatim, return verbatim, persist nothing."""
    try:
        upstream = await client.request(
            request.method,
            url,
            headers=_forward_headers(request.headers),
            params=dict(request.query_params),
            content=body or None,
        )
    except httpx.HTTPError as exc:
        return _upstream_unreachable(url, exc)
    headers = _response_headers(upstream.headers)
    headers["X-AgentVCR-Recorded"] = "true" if recorded else "false"
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)


async def _forward_recorded(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    *,
    provider: Provider,
    run_id: str,
    parsed: dict[str, Any],
) -> Response:
    started_at, started = utcnow(), time.perf_counter()
    try:
        upstream = await client.post(
            url,
            headers=_forward_headers(request.headers),
            params=dict(request.query_params),
            content=body,
        )
    except httpx.HTTPError as exc:
        return _upstream_unreachable(url, exc)
    latency_ms = int((time.perf_counter() - started) * 1000)

    step = record_step(
        request.app.state.store,
        run_id=run_id,
        provider=provider,
        request_headers=request.headers,
        request_body=parsed,
        response=_parse_json_object(upstream.content),
        status_code=upstream.status_code,
        latency_ms=latency_ms,
        started_at=started_at,
    )
    headers = _response_headers(upstream.headers)
    headers.update(
        {"X-AgentVCR-Recorded": "true", "X-AgentVCR-Run": run_id, "X-AgentVCR-Step": str(step.idx)}
    )
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)


async def _forward_streaming_recorded(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    *,
    provider: Provider,
    run_id: str,
    parsed: dict[str, Any],
) -> Response:
    """Stream SSE through to the client while accumulating the final message."""
    started_at, started = utcnow(), time.perf_counter()
    upstream_request = client.build_request(
        "POST",
        url,
        headers=_forward_headers(request.headers),
        params=dict(request.query_params),
        content=body,
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return _upstream_unreachable(url, exc)

    settings: Settings = request.app.state.settings
    store: Store = request.app.state.store

    if upstream.status_code >= 400:
        # An error response is not an SSE stream; drain it and record it as a step.
        error_body = await upstream.aread()
        await upstream.aclose()
        record_step(
            store,
            run_id=run_id,
            provider=provider,
            request_headers=request.headers,
            request_body=parsed,
            response=_parse_json_object(error_body),
            status_code=upstream.status_code,
            latency_ms=int((time.perf_counter() - started) * 1000),
            started_at=started_at,
        )
        headers = _response_headers(upstream.headers)
        headers.update({"X-AgentVCR-Recorded": "true", "X-AgentVCR-Run": run_id})
        return Response(content=error_body, status_code=upstream.status_code, headers=headers)

    async def relay() -> AsyncIterator[bytes]:
        chunks: list[bytes] = []
        try:
            # Raw iteration: the forwarded request asks for identity encoding, so what
            # arrives on the wire is what we store and what the client receives.
            async for chunk in upstream.aiter_raw():
                chunks.append(chunk)
                yield chunk
        finally:
            await upstream.aclose()
            # A client that disconnects mid-stream still leaves a partial tape, which
            # is more useful than nothing when the point is to debug a failed run.
            record_step(
                store,
                run_id=run_id,
                provider=provider,
                request_headers=request.headers,
                request_body=parsed,
                response=provider.accumulate_stream(chunks),
                status_code=upstream.status_code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                chunks=chunks if settings.record_chunks else None,
                started_at=started_at,
            )

    headers = _response_headers(upstream.headers)
    headers.update({"X-AgentVCR-Recorded": "true", "X-AgentVCR-Run": run_id})
    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


# ----------------------------------------------------------------------------- helpers


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Client headers, credentials included — they go upstream but never to disk."""
    out = {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HEADERS_NOT_FORWARDED and not name.lower().startswith("x-agentvcr-")
    }
    # Identity encoding keeps the recorded bytes and the client's bytes identical.
    out["accept-encoding"] = "identity"
    return out


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {n: v for n, v in headers.items() if n.lower() not in _HEADERS_NOT_RETURNED}


def _parse_json_object(body: bytes) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"type": code, "message": message, "by": "agentvcr"}}
    )


def _upstream_unreachable(url: str, exc: httpx.HTTPError) -> JSONResponse:
    # The call never reached the provider, so there is no response to record.
    return _error(502, "upstream_unreachable", f"{type(exc).__name__} contacting {url}: {exc}")
