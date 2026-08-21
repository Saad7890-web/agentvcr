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

All four modes are served from here: ``record`` forwards and captures, ``replay``
answers from a tape without a network, ``fork`` does both — tape first, then live on a
child run — and ``passthrough`` is a plain proxy that persists nothing.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import MODES, Settings
from ..core import forker, replayer
from ..core.models import STATUS_DIVERGED, Run
from ..core.recorder import RunRouter, record_step
from ..core.store import Store, utcnow
from ..providers import Provider, get_provider

RUN_HEADER = "x-agentvcr-run"
MODE_HEADER = "x-agentvcr-mode"

MODE_RECORD = "record"
MODE_REPLAY = "replay"
MODE_FORK = "fork"
MODE_PASSTHROUGH = "passthrough"
#: Modes the tape can be captured or served for; the rest is a plain proxy.
_TAPE_MODES = (MODE_RECORD, MODE_REPLAY, MODE_FORK)

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

    if mode not in MODES:
        # Silently proxying an unknown mode would look like recording and record
        # nothing, so a typo in X-AgentVCR-Mode is an error, not a default.
        return _error(400, "unknown_mode", f"unknown mode {mode!r}; expected one of {list(MODES)}")
    body = await request.body()
    upstream = settings.upstream_for(provider.name).rstrip("/") + "/" + subpath.lstrip("/")
    is_recorded_path = request.method == "POST" and f"/{subpath.lstrip('/')}" in tuple(
        provider.recorded_paths
    )

    parsed: dict[str, Any] | None = None
    if is_recorded_path and mode in _TAPE_MODES:
        parsed = _parse_json_object(body)

    if parsed is None:
        if mode == MODE_REPLAY:
            # Replay never contacts an upstream, not even for the calls it cannot serve.
            return _error(
                501,
                "not_replayable",
                f"{request.method} {request.url.path} is not recorded, and replay mode "
                f"has no upstream to forward it to",
            )
        # A fork does have an upstream — it is a live run that starts from a tape — so
        # a path the tape never covered (`GET /models`) is forwarded, as when recording.
        return await _forward(request, client, upstream, body, recorded=False)

    router: RunRouter = request.app.state.run_router
    run = router.resolve(
        provider=provider,
        body=parsed,
        mode=mode,
        run_id=explicit_run_id,
        upstream_url=settings.upstream_for(provider.name),
    )
    if mode == MODE_REPLAY:
        return await _replay(
            request, client, upstream, body, provider=provider, run=run, parsed=parsed
        )
    if mode == MODE_FORK:
        return await _fork(
            request, client, upstream, body, provider=provider, run=run, parsed=parsed
        )
    return await _go_live(
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


async def _go_live(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    *,
    provider: Provider,
    run_id: str,
    parsed: dict[str, Any],
) -> Response:
    """Make the call for real and record it — what recording does, and what a replay
    or a fork falls through to once it leaves the tape."""
    if provider.is_streaming(parsed):
        return await _forward_streaming_recorded(
            request, client, url, body, provider=provider, run_id=run_id, parsed=parsed
        )
    return await _forward_recorded(
        request, client, url, body, provider=provider, run_id=run_id, parsed=parsed
    )


# ----------------------------------------------------------------------------- replay


async def _replay(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    *,
    provider: Provider,
    run: Run,
    parsed: dict[str, Any],
) -> Response:
    """Answer from the tape — no upstream, no tokens (DESIGN.md §5).

    The replay is recorded as a run of its own, so the served step, its position and
    any divergence belong to the replay rather than to the recording it came from.
    """
    store: Store = request.app.state.store
    settings: Settings = request.app.state.settings
    router: RunRouter = request.app.state.run_router

    try:
        replayer.assert_replayable(run)
        session = router.replay_run_for(
            run, provider=provider, upstream_url=settings.upstream_for(provider.name)
        )
        served = replayer.next_from_tape(
            store,
            replay_run=session,
            provider=provider,
            body=parsed,
            policy=settings.mismatch_policy,
        )
    except replayer.ReplayError as exc:
        return _replay_error(exc)

    if served is None:
        # `live-on-miss` policy: the tape stops here, the branch continues for real.
        return await _go_live(
            request, client, url, body, provider=provider, run_id=session.id, parsed=parsed
        )
    return _serve_from_tape(
        request, provider=provider, session=session, parsed=parsed, served=served
    )


# ------------------------------------------------------------------------------- fork


async def _fork(
    request: Request,
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    *,
    provider: Provider,
    run: Run,
    parsed: dict[str, Any],
) -> Response:
    """Replay the prefix, serve the edit, then go live on the fork (DESIGN.md §6).

    Unlike a replay, a fork is *not* given a session run of its own: it already is one.
    ``agentvcr fork`` creates the child run before the agent starts, because the edits
    that define where the branch leaves the tape have to be stored somewhere first.
    """
    store: Store = request.app.state.store
    settings: Settings = request.app.state.settings

    try:
        fork_plan = forker.plan(store, run)
        served = forker.next_step(
            store,
            fork_run=run,
            fork_plan=fork_plan,
            provider=provider,
            body=parsed,
            policy=settings.mismatch_policy,
        )
    except replayer.ReplayError as exc:
        return _replay_error(exc)

    if served is not None:
        return _serve_from_tape(
            request, provider=provider, session=run, parsed=parsed, served=served
        )

    # Live from here on, with the edit applied to every outbound request — each one
    # carries the whole conversation, so patching only the first would let the real
    # tool result back in on the next turn.
    patched, applied = forker.patch_outbound(fork_plan, provider, parsed)
    if patched is not parsed:
        body = json.dumps(patched, ensure_ascii=False).encode()
    response = await _go_live(
        request, client, url, body, provider=provider, run_id=run.id, parsed=patched
    )
    if applied:
        response.headers["X-AgentVCR-Patched"] = ", ".join(applied)
    return response


# ----------------------------------------------------------------------- serving a step


def _serve_from_tape(
    request: Request,
    *,
    provider: Provider,
    session: Run,
    parsed: dict[str, Any],
    served: replayer.Served,
) -> Response:
    """Answer from ``served`` and record the answer as a step of ``session``.

    The step recorded holds the request the agent actually sent and the response it was
    given, so a replay or a fork can be diffed against the run it came from.
    """
    store: Store = request.app.state.store
    started_at, started = utcnow(), time.perf_counter()
    streaming = provider.is_streaming(parsed)
    content, media_type = replayer.response_body(served.step, provider, streaming=streaming)
    status_code = served.step.status_code or 200

    step = record_step(
        store,
        run_id=session.id,
        provider=provider,
        request_headers=request.headers,
        request_body=parsed,
        response=served.step.response,
        status_code=status_code,
        latency_ms=int((time.perf_counter() - started) * 1000),
        started_at=started_at,
        diverged=served.diverged,
    )
    if served.diverged:
        store.update_run(session.id, status=STATUS_DIVERGED)

    headers = {
        "Content-Type": media_type,
        "X-AgentVCR-Recorded": "false",
        "X-AgentVCR-Replayed": "true",
        "X-AgentVCR-Run": session.id,
        "X-AgentVCR-Tape": served.step.run_id,
        "X-AgentVCR-Step": str(step.idx),
        "X-AgentVCR-Diverged": "true" if served.diverged else "false",
        "X-AgentVCR-Edited": "true" if served.edited else "false",
    }
    return Response(content=content, status_code=status_code, headers=headers)


def _replay_error(exc: replayer.ReplayError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"type": exc.code, "message": exc.message, "by": "agentvcr", **exc.details}
        },
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
