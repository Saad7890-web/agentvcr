"""Control REST API consumed by the web UI (``/api/*``).

Everything the UI does is here: list runs and their lineage, read one run's timeline,
open a step, fork a step with an edit, re-run the fork, and diff two runs. It is the
same machinery the CLI drives — :mod:`agentvcr.core` — with JSON on the outside.

**Why this is guarded.** The proxy binds to localhost, but *any* page the user's
browser visits can send requests to ``localhost:8484``. This API hands out whole
recorded prompts and can start a process on the user's machine, so it refuses anything
that is not this UI talking to its own server:

* **Origin.** A browser attaches it to every cross-origin request, and to same-origin
  writes. Present and not our own origin means someone else's page is calling; that is
  a 403. Non-browser clients (curl, tests) send none and are let through — they are not
  the threat model, since a program that can run curl can read the tape file directly.
* **Host.** DNS rebinding beats an Origin check by making the attacker's page *be* this
  origin, so the host we were dialed on must be a loopback name. A server the user
  deliberately exposed with ``--host`` cannot make that check and does not pretend to.

A read-only page is not a safe page here: ``POST rerun`` spawns the argv stored on a
run, which is why the guard covers the whole router rather than the writes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings
from ..core import differ, forker, launch
from ..core.models import STATUS_DIVERGED, Edit, Run, RunStats, Step, ToolCall
from ..core.store import Store
from ..providers import Provider, get_provider
from .jobs import Job, JobRunner

#: Host names that mean "this machine". A server bound to one of these can insist it
#: was dialed by one of these; a server the user pointed at the world cannot.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: How much of a message or a result the run/step listings inline. The full value is
#: one request away, in the step inspector.
PREVIEW_CHARS = 400


# ------------------------------------------------------------------------------- guard


def _hostname(host_header: str) -> str:
    """The host half of a ``Host`` header, brackets kept so ``[::1]`` stays itself."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host.partition("]")[0] + "]"
    return host.rsplit(":", 1)[0] if ":" in host else host


def guard(request: Request) -> None:
    """Refuse calls that did not come from this server's own UI (see the module docs)."""
    settings: Settings = request.app.state.settings
    host_header = request.headers.get("host", "")

    if settings.host in LOOPBACK_HOSTS and _hostname(host_header) not in LOOPBACK_HOSTS:
        raise HTTPException(
            403,
            f"refusing an /api request addressed to {host_header!r}: this server answers "
            f"on the loopback, and a name that merely resolves there is how DNS rebinding "
            f"turns someone else's page into this origin",
        )

    origin = request.headers.get("origin")
    # Compared on host and port only: a reverse proxy may terminate TLS in front of us,
    # and the scheme is not what is being defended here — the address is.
    if origin and urlsplit(origin).netloc.lower() != host_header.lower():
        raise HTTPException(
            403,
            f"refusing a cross-origin /api request from {origin!r}; the agentvcr UI is "
            f"served by this same server, and no other page has business reading tapes "
            f"or starting agents",
        )


router = APIRouter(prefix="/api", tags=["api"], dependencies=[Depends(guard)])


# ------------------------------------------------------------------------ request bodies


class EditIn(BaseModel):
    """One edit, in the shape :class:`agentvcr.core.forker.EditSpec` resolves."""

    kind: str = Field(description="response | tool_result | request_patch")
    value: Any = Field(default=None, description="The new response, result or message content.")
    target: str | None = Field(
        default=None, description="Tool name for tool_result; message index for request_patch."
    )


class ForkIn(BaseModel):
    at: int = Field(description="Step to branch at, numbered as the timeline shows it.")
    edits: list[EditIn] = Field(default_factory=list)
    name: str | None = None


# ------------------------------------------------------------------------- serialization


def _state(request: Request) -> tuple[Store, Settings, JobRunner]:
    return request.app.state.store, request.app.state.settings, request.app.state.jobs


def _provider_for(run: Run) -> Provider:
    """The wire format this run speaks. An unknown one is refused rather than guessed:
    reading an Anthropic tape with the OpenAI provider finds no text and no tool calls,
    which renders as an empty run instead of an error."""
    provider = get_provider(run.provider or "openai")
    if provider is None:
        raise HTTPException(
            400, f"run {run.id} was recorded in a format this build does not know: {run.provider}"
        )
    return provider


def _safe(call: Any, payload: dict[str, Any]) -> Any:
    """A provider whose renderer is not written yet must not break a listing."""
    try:
        return call(payload)
    except NotImplementedError:  # pragma: no cover - every shipped provider implements these
        return None


def _label(run: Run) -> str:
    if run.name:
        return run.name
    if run.parent_run_id:
        return f"fork of {run.parent_run_id}"
    return f"replay of {run.replay_of}" if run.replay_of else "-"


def run_summary(store: Store, run: Run, *, stats: RunStats | None = None) -> dict[str, Any]:
    """One run as the list and the header show it — counts, lineage, and what it allows."""
    stats = stats if stats is not None else store.run_stats(run.id)
    try:
        plan = rerun_plan(store, run)
        rerun: dict[str, Any] = {"available": True, "mode": plan.mode, "command": plan.command}
    except RerunUnavailable as exc:
        rerun = {"available": False, "reason": str(exc)}
    return {
        **run.as_dict(),
        "label": _label(run),
        "stats": stats.as_dict(),
        "diverged": stats.diverged or run.status == STATUS_DIVERGED,
        "rerun": rerun,
    }


def step_summary(provider: Provider, step: Step) -> dict[str, Any]:
    """A step as the timeline shows it: what it cost, and what the model said."""
    response = step.response or {}
    calls = _safe(provider.extract_tool_calls, response) or []
    text = _safe(provider.assistant_text, response)
    error = response.get("error") if not step.ok else None
    return {
        "idx": step.idx,
        "model": step.model,
        "status_code": step.status_code,
        "ok": step.ok,
        "latency_ms": step.latency_ms,
        "tokens": step.total_tokens,
        "usage": step.usage,
        "diverged": step.diverged,
        "started_at": step.started_at,
        "fingerprint": step.fingerprint,
        "has_chunks": step.response_chunks is not None,
        "messages": len(provider.messages_of(_body(step))),
        "text": _clip(text),
        "tool_calls": [
            {
                "tool_name": call.get("tool_name"),
                "tool_call_id": call.get("tool_call_id"),
                "args": call.get("args"),
            }
            for call in calls
        ],
        "error": error if isinstance(error, dict) else None,
    }


def tool_call_json(call: ToolCall) -> dict[str, Any]:
    return {**call.as_dict(), "result_preview": _clip(call.result)}


def edit_json(edit: Edit) -> dict[str, Any]:
    return {**edit.as_dict(), "description": forker.describe(edit)}


def _body(step: Step) -> dict[str, Any]:
    body = step.request.get("body")
    return body if isinstance(body, dict) else {}


def _clip(value: Any) -> str | None:
    """A value as one readable line, short enough to sit in a table."""
    if value is None:
        return None
    text = value if isinstance(value, str) else _json_line(value)
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


def _json_line(value: Any) -> str:
    return " ".join(json.dumps(value, ensure_ascii=False, default=str).split())


# ------------------------------------------------------------------------------- re-run


class RerunUnavailable(Exception):
    """This run cannot be re-run, with the reason a user can act on."""


@dataclass(frozen=True)
class RerunPlan:
    """How re-running a run would work: which mode, which argv, from where."""

    mode: str
    command: list[str]
    cwd: str | None
    #: Tape a replay would be created against. ``None`` for a fork, which already exists.
    tape: str | None


def rerun_plan(store: Store, run: Run) -> RerunPlan:
    """What ``POST /rerun`` would do, or why it cannot (DESIGN.md §10).

    A fork is re-run *as itself* — it is where the edits live — and only once, because
    its position on the tape is how many steps it has recorded (PLAN.md phase 4). Any
    other run is re-run as a fresh replay of its tape, which costs nothing.
    """
    parent = store.get_run(run.parent_run_id) if run.parent_run_id else None
    command = run.command or (parent.command if parent else None)
    if not command:
        raise RerunUnavailable(
            "agentvcr does not know how to start this agent: the run was not launched "
            "with `agentvcr run`, so no command was stored with it"
        )
    cwd = run.meta.get("cwd") or (parent.meta.get("cwd") if parent else None)

    if run.parent_run_id:
        recorded = store.count_steps(run.id)
        if recorded:
            raise RerunUnavailable(
                f"this fork already ran and holds {recorded} step(s); re-running it "
                f"would resume in the middle of its own branch. Fork again from step "
                f"{run.fork_step} instead"
            )
        return RerunPlan(mode="fork", command=list(command), cwd=cwd, tape=None)

    tape = run.replay_of or run.id
    if not store.count_steps(tape):
        raise RerunUnavailable("this run recorded no steps, so there is nothing to replay")
    return RerunPlan(mode="replay", command=list(command), cwd=cwd, tape=tape)


# ------------------------------------------------------------------------------- routes


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    """What server the UI is talking to, and where its tape lives."""
    store, settings, _ = _state(request)
    return {
        "version": __version__,
        "mode": settings.mode,
        "db": str(settings.db_path),
        "schema_version": store.schema_version,
        "upstreams": settings.upstreams,
        "base_url": launch.local_base(settings),
        "mismatch_policy": settings.mismatch_policy,
    }


@router.get("/runs")
async def list_runs(request: Request, limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Every run, newest first. Lineage is in the rows; the UI nests them."""
    store, _, _ = _state(request)
    runs = store.list_runs(limit=limit)
    return {"runs": [run_summary(store, run) for run in runs]}


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str) -> dict[str, Any]:
    """One run's timeline: its steps, the tool runs between them, and its edits."""
    store, _, jobs = _state(request)
    run = _require_run(store, run_id)
    provider = _provider_for(run)
    steps = store.list_steps(run.id)
    parent = store.get_run(run.parent_run_id) if run.parent_run_id else None
    job = jobs.running_for(run.id)
    return {
        "run": run_summary(store, run),
        "steps": [step_summary(provider, step) for step in steps],
        "tool_calls": [tool_call_json(call) for call in store.list_tool_calls(run.id)],
        "edits": [edit_json(edit) for edit in store.list_edits(run.id)],
        "parent": run_summary(store, parent) if parent else None,
        "children": [run_summary(store, child) for child in store.list_runs(parent_run_id=run.id)],
        "job": job.as_dict() if job else None,
    }


@router.get("/runs/{run_id}/steps/{idx}")
async def get_step(request: Request, run_id: str, idx: int) -> dict[str, Any]:
    """One step in full: the conversation it sent, and the response it got back."""
    store, _, _ = _state(request)
    run = _require_run(store, run_id)
    step = store.get_step(run.id, idx)
    if step is None:
        raise HTTPException(404, f"run {run_id} has no step {idx}")
    provider = _provider_for(run)
    body = _body(step)
    settings_fields = {k: v for k, v in provider.normalize(body).items() if k != "messages"}
    return {
        "step": {
            **step_summary(provider, step),
            "request": {"headers": step.request.get("headers") or {}, "body": body},
            "response": step.response,
            "conversation": provider.messages_of(body),
            "request_settings": settings_fields,
            "full_text": _safe(provider.assistant_text, step.response or {}),
        },
        "tool_calls": [
            tool_call_json(call)
            for call in store.list_tool_calls(run.id)
            if call.after_step_idx == idx
        ],
    }


@router.post("/runs/{run_id}/fork", status_code=201)
async def fork_run(request: Request, run_id: str, payload: ForkIn) -> dict[str, Any]:
    """Branch a run at a step, with an edit. The edit is what defines the fork point."""
    store, _, _ = _state(request)
    tape = _require_run(store, run_id)
    specs = [
        forker.EditSpec(kind=edit.kind, value=edit.value, target=edit.target)
        for edit in payload.edits
    ]
    try:
        child = forker.create(
            store,
            tape=tape,
            at=payload.at,
            edits=specs,
            provider=_provider_for(tape),
            name=payload.name,
        )
    except forker.CannotFork as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "fork": run_summary(store, child),
        "edits": [edit_json(edit) for edit in store.list_edits(child.id)],
        "command": forker.rerun_command(child, tape.command),
    }


@router.post("/runs/{run_id}/rerun", status_code=202)
async def rerun(request: Request, run_id: str) -> dict[str, Any]:
    """Re-run this run's agent: a fork branches, anything else replays.

    The command spawned is the one stored on the run when it was recorded — never
    anything from this request (:mod:`agentvcr.server.jobs`).
    """
    store, settings, jobs = _state(request)
    run = _require_run(store, run_id)
    if jobs.running_for(run.id) is not None:
        raise HTTPException(409, f"run {run.id} is already being re-run")
    try:
        plan = rerun_plan(store, run)
    except RerunUnavailable as exc:
        raise HTTPException(400, str(exc)) from exc

    if plan.mode == "fork":
        target = run
        # The fork learns its argv the first time something re-runs it, exactly as the
        # CLI records it — a fork is created before anyone knows what will be re-run.
        store.update_run(target.id, command=plan.command)
    else:
        target = store.create_run(
            mode="replay",
            name=run.name,
            command=plan.command,
            replay_of=plan.tape,
            provider=run.provider,
            upstream_url=run.upstream_url,
            meta={"cwd": plan.cwd} if plan.cwd else None,
        )

    job = jobs.start(
        run_id=target.id,
        mode=plan.mode,
        command=plan.command,
        env={
            **os.environ,
            **launch.agent_env(launch.local_base(settings), run_id=target.id, mode=plan.mode),
        },
        cwd=plan.cwd,
    )
    return {"job": job.as_dict(), "run": run_summary(store, store.get_run(target.id) or target)}


@router.get("/jobs")
async def list_jobs(request: Request) -> dict[str, Any]:
    _, _, jobs = _state(request)
    return {"jobs": [job.as_dict() for job in jobs.list()]}


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict[str, Any]:
    """A re-run in progress: its output so far, and how it ended."""
    store, _, jobs = _state(request)
    job = _require_job(jobs, job_id)
    run = store.get_run(job.run_id)
    return {"job": job.as_dict(), "run": run_summary(store, run) if run else None}


@router.post("/jobs/{job_id}/stop")
async def stop_job(request: Request, job_id: str) -> dict[str, Any]:
    """Stop a re-run. Its run keeps whatever steps the agent got as far as recording."""
    _, _, jobs = _state(request)
    return {"job": jobs.stop(_require_job(jobs, job_id)).as_dict()}


@router.get("/diff")
async def diff(request: Request, a: str, b: str) -> dict[str, Any]:
    """Compare two runs step by step — behavior, not bytes (DESIGN.md §7)."""
    store, _, _ = _state(request)
    left, right = _require_run(store, a), _require_run(store, b)
    try:
        result = differ.diff_runs(store, left, right)
    except differ.IncomparableRuns as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "left": run_summary(store, left),
        "right": run_summary(store, right),
        "diff": result.as_dict(),
    }


# ------------------------------------------------------------------------------ helpers


def _require_run(store: Store, run_id: str) -> Run:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no such run: {run_id}")
    return run


def _require_job(jobs: JobRunner, job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id} (jobs live only as long as the server)")
    return job
