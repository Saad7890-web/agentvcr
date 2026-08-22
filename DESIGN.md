# agentvcr — System Design

> **VCR for AI agents.** A drop-in proxy (change only your `base_url`) that records every
> LLM call and tool call from any agent run into a local file — then deterministically
> replays the run for free, forks it at any step, lets you edit a prompt or a tool
> result, re-runs from that exact point, and diffs two runs side by side.

- **Package / CLI / repo name:** `agentvcr` (free on PyPI and npm as of 2026-08-17)
- **Language:** Python ≥ 3.10 (FastAPI + httpx + SQLite), bundled static web UI (Vite + React)
- **License:** Apache-2.0 (open-core friendly)

---

## 1. Core idea and constraints

The agent's code changes in exactly one place:

```python
client = OpenAI(base_url="http://localhost:8484/openai/v1")  # was api.openai.com
# or
client = Anthropic(base_url="http://localhost:8484/anthropic")  # was api.anthropic.com
```

Everything else — recording, replay, forking, editing, diffing — happens in the proxy
and its local store. **No SDK, no framework integration, no decorators.** This is the
wedge: LangGraph's time-travel is locked to LangGraph; agentvcr works with anything
that speaks the OpenAI or Anthropic wire format (which includes Groq, Gemini,
OpenRouter, Ollama, vLLM…).

Hard constraints the design honors:

- **Zero-token replays.** Replay mode never contacts an upstream API.
- **Never store secrets.** `Authorization` / `x-api-key` headers are forwarded upstream
  but redacted before anything touches disk.
- **Local-first.** One SQLite file per project (`.agentvcr/agentvcr.db`). No accounts,
  no network calls except to the LLM provider the user already uses.

## 2. The key insight: tool calls are visible at the LLM boundary

Agents execute tools *client-side*, so a proxy never sees a tool run directly. But it
doesn't need to:

- The **tool call** (name + arguments) appears in the assistant response of LLM call *N*.
- The **tool result** appears as a tool-result message embedded in the request of LLM
  call *N+1* (the agent appends it to the conversation before asking the LLM what to
  do next).

By diffing consecutive request message lists within a run, the proxy reconstructs a
complete interleaved timeline — *LLM step, tool step, LLM step, …* — with tool names,
arguments, results, and which LLM step triggered them. **Zero client instrumentation.**
Tool steps are extracted at record time into their own table so the UI and differ can
treat them as first-class.

Corollary and honest limitation: in replay/fork modes the agent still *executes* its
tools locally (we can't stub client-side code from a proxy). Replay stays deterministic
at the LLM boundary regardless — recorded responses are served by position, not by
what the possibly-drifted client sent — but tools with external side effects will
re-fire. This is documented behavior; an optional tool-stubbing shim is a post-MVP
add-on for users who want it.

## 3. Modes

| Mode | Upstream called? | Behavior |
|---|---|---|
| `record` | yes | Transparent passthrough; every request/response persisted as a step in a run. |
| `replay` | **no** | Serves recorded responses positionally. Free, offline, deterministic. Recorded as a run of its own (§4). |
| `fork` | after fork point | Replays the tape up to the fork/edit point, then goes live against the real API (recording the new branch as a child run). |
| `passthrough` | yes | Pure proxy, no recording (escape hatch). |

Mode is resolved per request: the `X-AgentVCR-Mode` header wins, then the mode stored on
the resolved run (which is how `agentvcr run --mode replay` takes effect without the agent
sending anything), then the proxy's default setting.

**Mount convention.** A provider's path prefix mirrors its upstream base URL, so swapping
`base_url` is the whole change and no path rewriting is needed: `/openai/v1` stands in for
`https://api.openai.com/v1`, `/anthropic` for `https://api.anthropic.com`. Everything after
the prefix is forwarded verbatim.

## 4. Runs, steps, and how requests join a run

A **run** is one agent execution; a **step** is one LLM call within it (tool steps are
derived, see §2). Two ways a request is assigned to a run:

1. **Explicit (reliable):** two forms, because most agents cannot set headers.
   - **In the base URL:** the proxy also mounts every provider under `/r/<run-id>/…`, so
     `http://localhost:8484/r/01H…/openai/v1` pins the run without the client knowing.
     This is what `agentvcr run -- python agent.py` uses: it creates the run, exports
     `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` pointing at that prefix (plus
     `AGENTVCR_RUN`/`AGENTVCR_MODE`), and **stores the command line** (this powers
     one-click re-run from the UI later). The agent needs no change at all.
   - **In a header:** `X-AgentVCR-Run: <id>`, for clients that support `default_headers`.
2. **Heuristic (zero-config):** an incoming request whose message list extends the
   message prefix of an active run's last step is chained onto that run; otherwise a new
   run starts. An idle timeout closes runs. Good enough for the drop-in demo; the
   explicit path is what docs recommend for CI.

**A replay is a run of its own.** Replaying tape *R* creates a new run with
`replay_of = R`, and the requests the agent sends plus the responses served back are
recorded onto it exactly as a recording would be. Three things fall out of this:

- **The tape is never written to.** Divergence is a property of *this replay*, not of
  the recording — marking the original would corrupt the very thing being replayed.
- **The position needs no session counter.** The *N*th call is answered with tape step
  *N* where *N* is simply how many steps the replay run has recorded so far, which
  survives a proxy restart and cannot drift.
- **A run can be diffed against its own replay** (§7), which is how a replay proves it
  reproduced the recording rather than merely not crashing.

Replay lineage is deliberately separate from fork lineage (`parent_run_id`): a fork
*branches away* from a tape and generates new content, a replay *re-derives* one.

`agentvcr run --mode replay --run <tape>` creates that replay run up front and points
the agent at it. A client that instead points straight at a recorded run and asks for
replay mode by header gets one created on first call, for as long as the session stays
warm — same zero-config ergonomics as the grouping heuristic, and just as advisory.

## 5. Replay matching

- **Primary: positional.** The *N*th LLM call of the session gets the *N*th recorded
  response. This is what makes replay robust to harmless nondeterminism (timestamps in
  prompts, etc.).
- **Fingerprint check:** each recorded step stores a hash of the normalized request
  (model + messages + tools, volatile headers stripped). On replay, mismatch between
  the incoming request's fingerprint and the tape triggers a policy:
  - `warn` (default): serve the positional response, mark the step as *diverged*.
  - `strict`: return a structured 409 error (for CI).
  - `live-on-miss`: fall through to the real API from that point, i.e. auto-fork.
- **Streaming:** if the client asked for `stream: true`, the proxy prefers the **raw
  recorded chunk sequence** — replaying the provider's own bytes, byte-for-byte. Only
  when no chunk tape was kept (`record_chunks = false`) is the stream synthesized from
  the stored final message: faithful in content, but the ids and chunk boundaries are
  ours, so byte-identity is a property of the raw tape, not of replay in general.
  Recorded streams are always accumulated into a final message at record time as well,
  so one tape answers both a streaming and a non-streaming client.
- **Errors and retries.** A non-2xx upstream response is recorded as a step like any
  other, with its status code. Both SDKs retry 429/500 by default, so one logical call
  can produce two steps — and because the failure is on the tape, positional replay
  reproduces the same 429-then-success sequence the client already knows how to handle.
  Recording only successes would desynchronize every later step. A request that never
  reaches the provider (connection refused, DNS failure) is *not* recorded: there is no
  response to serve back, and the proxy answers 502. The cost of this fidelity is that
  a replay reproduces the *client's* retry loop too, backoff sleeps included — and that
  the tape is coupled to the client's retry configuration, since an agent replayed with
  a different `max_retries` consumes the tape at a different rate. A policy to skip
  recorded error steps on replay is post-MVP.
- **Concurrency note (post-MVP):** parallel LLM calls (multi-agent fan-out) break pure
  positional order → matching falls back to fingerprint-first with position as
  tiebreak. Flagged for milestone 7, not the MVP.

  This is measured, not assumed. `examples/framework-check/` records and replays real
  framework agents — LangGraph and the OpenAI Agents SDK — against a scripted upstream
  that is then killed. Both sequential agents replay perfectly, with no fingerprint
  divergence at all. A LangGraph graph whose branches
  fan out in one superstep replays *nondeterministically* — over five runs the replay
  raced the other way twice, and each branch then received the other's answer. Both
  times the fingerprint check caught it: every step flagged, the replay run marked
  `diverged`. A replay of a fan-out agent may be wrong; it is never quietly wrong, and
  that is the property that makes shipping the MVP without fan-out support defensible.

## 6. Fork & edit semantics

**An edit defines the fork point.** For a fork of run R at step *k* (numbered as
`agentvcr show` prints it, from 0), *where* the branch leaves the tape follows from
*what* was edited — it is not a separate setting:

- **Edited assistant response at step k** (`--edit-response FILE`): steps `0 … k-1` are
  served from the tape, step *k* is answered with the edited response, and `k+1 …` are
  live. The agent genuinely reacts to the edit.
- **Edited tool result of step k** (`--edit-tool-result NAME=FILE`): the tool result
  physically lives inside request `k+1`, and during the replayed prefix the proxy
  ignores request bodies — so the edit is applied as an **outbound request patch**.
  Steps `0 … k` replay from the tape, and from the first live call onward the proxy
  rewrites the matching tool-result message in the outgoing request body before
  forwarding upstream.
- **Edited prompt** (`--edit-message I=FILE`): the same outbound patch, pointed at a
  message instead of a result. It rewrites request *k* itself, so steps `0 … k-1`
  replay and the branch goes live *at* *k*, carrying the new message.
- The fork is stored as a child run (`parent_run_id`, `fork_step`, edits) — so the diff
  view can show exactly what changed and lineage forms a tree.

Two consequences of "every request carries the whole conversation" shape the rest:

- **The patch is re-applied to every live call**, not just the first. Patching only the
  first would hand the model the real tool result again on the very next turn.
- **Edits of different kinds at one step are refused, at creation.** Each kind puts the
  branch in a different place, so mixing them is a contradiction rather than a merge: an
  edited response at *k* replaces the very tool calls a tool-result edit names, leaving
  that edit silently inert. Several tool results at one step are the exception — a step
  can call more than one tool.

This yields the 30-second demo: *agent fails at step 9 → open step 6 → edit the tool
result → re-run → agent passes.* Re-running executes the user's own agent process with
`AGENTVCR_MODE=fork AGENTVCR_RUN=<fork-id>`; if the original run was launched via
`agentvcr run -- …`, the stored command makes this a single button in the UI.

## 7. Diffing

Align two runs (LCS over step fingerprints, positional fallback within a stretch where
nothing matched), then per aligned step: message-level diff of requests, response text
and tool-call diffs, tool name/args/result diffs, and HTTP status.

Two decisions shape what comes out:

- **Comparison is semantic, not byte-level.** Any two live calls to one model differ in
  their response id, their timestamp and usually their token counts. Reporting those
  leaves no signal for the question actually being asked — *did the agent decide
  something different?* — so they are not compared.
- **A changed request message is reported once**, at the step that introduced it. Every
  request carries the whole conversation, so a single edited system prompt is present
  in all forty steps that follow it; reporting it forty times buries the rest. The
  first occurrence is also, by construction, the step the runs diverge at.

Within one step, changes are ordered most-explanatory first: a tool that returned
something else *causes* the request message carrying it to differ, and saying so is
more use than saying a message changed.

Three consumers:

- `agentvcr diff <runA> <runB>` — colored terminal summary: *"runs diverge at step 6
  (tool `search_flights` returned different results)"*. Exits 1 on a difference, like
  `diff(1)`.
- UI side-by-side view with per-step drill-down.
- The CI feature that exit code is for ("your prompt change altered the agent's
  decisions in 12 of 40 recorded runs").

## 8. Storage schema (SQLite)

`.agentvcr/agentvcr.db` in the project directory (overridable via `--db` /
`AGENTVCR_DB`). Plain `sqlite3` with a tiny migration runner — no ORM.

**Known growth characteristic.** `steps.request_json` holds the whole request, and
every request contains the entire conversation so far, so a run's storage is quadratic
in its length: a 100-step run over a 50k-token context stores that history 100 times.
Acceptable for the runs people actually debug, and the shape stays honest and trivially
diffable. If it starts to hurt, the fix is content-hashing message bodies into a shared
table (or storing per-step message deltas) behind the same `Step` dataclass — which is
also the question the export format (`.vcr.json`) has to answer.

```sql
runs(
  id TEXT PRIMARY KEY,            -- short ulid
  name TEXT,                      -- human label, defaults to timestamp
  created_at TEXT,
  mode TEXT,                      -- record | replay | fork
  parent_run_id TEXT,             -- fork lineage
  fork_step INTEGER,
  replay_of TEXT,                 -- replay lineage: the tape this run replays (§4)
  command TEXT,                   -- argv if launched via `agentvcr run`
  provider TEXT, upstream_url TEXT,
  status TEXT,                    -- active | completed | diverged
  meta_json TEXT
)

steps(
  id INTEGER PRIMARY KEY,
  run_id TEXT, idx INTEGER,       -- position within run
  request_json TEXT,              -- redacted headers + body
  response_json TEXT,             -- final accumulated response
  response_chunks BLOB,           -- optional raw SSE tape
  fingerprint TEXT,               -- normalized-request hash
  model TEXT, usage_json TEXT, latency_ms INTEGER,
  diverged INTEGER DEFAULT 0,     -- set on a *replay* step that drifted from its tape
  status_code INTEGER,            -- upstream HTTP status; errors are steps too (§5)
  started_at TEXT
)

tool_calls(                       -- materialized at record time from §2
  id INTEGER PRIMARY KEY,
  run_id TEXT, after_step_idx INTEGER,
  tool_name TEXT, args_json TEXT, result_json TEXT, tool_call_id TEXT
)

edits(
  id INTEGER PRIMARY KEY,
  run_id TEXT,                    -- the fork run the edit belongs to
  step_idx INTEGER,
  kind TEXT,                      -- response | tool_result | request_patch
  patch_json TEXT                 -- edited content / JSON patch
)
```

## 9. Components & repo layout

```
agentvcr/
├── pyproject.toml                # hatchling; deps: fastapi, uvicorn, httpx, typer
├── src/agentvcr/
│   ├── cli.py                    # typer: serve, run, runs, show, fork, diff, ui
│   ├── config.py                 # upstream map, mode defaults, db path, presets
│   ├── server/
│   │   ├── app.py                # FastAPI factory: proxy + control API + static UI
│   │   ├── proxy.py              # /openai/*, /anthropic/* routes, SSE plumbing
│   │   ├── api.py                # REST for the UI: runs, steps, edits, fork, rerun
│   │   └── jobs.py               # re-runs the UI started: spawn, watch, tail output
│   ├── core/
│   │   ├── store.py              # SQLite layer + migrations
│   │   ├── models.py             # Run / Step / ToolCall / Edit dataclasses
│   │   ├── recorder.py           # capture, redaction, run-grouping heuristic
│   │   ├── replayer.py           # positional matching, policies, SSE synthesis
│   │   ├── forker.py             # fork creation, edit application, request patching
│   │   ├── tools.py              # tool-call extraction from message diffs
│   │   ├── differ.py             # run alignment + step diffs
│   │   └── launch.py             # the env that points an agent at a run (/r/<id>/…)
│   └── providers/
│       ├── base.py               # Provider interface: normalize, fingerprint,
│       │                         #   extract/patch_tool_result, synthesize_stream
│       ├── openai_chat.py        # /v1/chat/completions (covers Groq, Gemini-compat,
│       │                         #   OpenRouter, Ollama, vLLM)
│       └── anthropic_messages.py # /v1/messages
├── ui/                           # Vite + React source → built into src/agentvcr/ui_dist/
├── examples/                     # openai-agents-sdk/, langgraph/, crewai/, plain-loop/
├── tests/                        # incl. golden record→replay round-trip tests
└── .github/workflows/ci.yml
```

**Provider abstraction is the load-bearing interface.** Recording/replay/fork logic is
format-agnostic; each provider module only knows how to normalize a body, fingerprint
it, pull tool calls/results out of messages, put an edited result *back* into one, and
synthesize SSE chunks. Adding `/v1/responses` or a new provider later touches only
`providers/`. Reading a tool result and rewriting one are the same knowledge in both
directions, so they live together and are contract-tested against each other.

**Upstream routing:** path prefix picks the wire format; the actual upstream host is
config (`agentvcr.toml` or flags), with presets so free-tier demos are one flag:
`agentvcr serve --preset groq` (Groq's OpenAI-compatible endpoint) or `--preset gemini`.
Replays need no upstream at all — the whole product is demoable on free tiers, and
replays cost zero tokens by construction.

## 10. Web UI (local, bundled)

Served by the same process at `http://localhost:8484/ui`, over a REST control API at
`/api`. Views:

1. **Run list** — table with status, model, steps, tokens, and fork/replay lineage: a
   branch is nested under the run it came from.
2. **Timeline** — the hero view: interleaved LLM/tool steps; failed/diverged steps flagged.
3. **Step inspector** — pretty-rendered messages, raw JSON toggle, usage/latency.
4. **Edit → fork** — edit a response, a tool result or a prompt message in place,
   prefilled with what was recorded → creates the fork run and shows the re-run command,
   with a *Run it now* button when the original command is stored.
5. **Diff** — pick two runs, side-by-side with divergence markers.

Static build committed nowhere — built in CI and bundled into the wheel, so
`pip install agentvcr` ships the UI with no Node required at install time. A source
checkout that has not run the build gets a page saying how, rather than a 404.

**Re-running is the UI's only side effect on the machine**, and it is deliberately
narrow: `POST /api/runs/<id>/rerun` spawns the argv that `agentvcr run` stored on that
run, as a list, with the fork/replay environment. Nothing in the request decides what
gets executed — only *which run* to re-run. A fork is re-run as itself and only once
(its position on the tape is how many steps it has recorded, §4); anything else is
re-run as a fresh replay of its tape, which costs nothing.

## 11. Security & privacy

- Binds to `127.0.0.1` by default; `--host` opt-in for containers.
- Header redaction list (`authorization`, `x-api-key`, `api-key`, cookies) applied
  before persistence; regex-based body redaction as a follow-up.
- Recordings may contain sensitive prompt data — README ships a recommended
  `.gitignore` entry for `.agentvcr/`.
- **The control API is guarded against the browser it is served to.** Binding to
  localhost keeps other machines out, but every page the user visits can send requests
  to `localhost:8484` — and `/api` hands out whole recorded prompts and can start a
  process. So it refuses a request whose `Origin` is not this server's own, and a
  request addressed to a `Host` this server was never bound to, which is what DNS
  rebinding looks like from the inside. A request with no `Origin` (curl, the CLI) is
  allowed: it is not a browser, and a program that can run curl can read the tape file
  anyway. The proxy routes are not guarded — an agent is exactly the non-browser client
  the check has to let through.

## 12. Explicit non-goals (MVP)

- Not an observability platform (no dashboards-of-averages, no OTel backend — though an
  OTel *exporter* is a fine later add-on).
- Not an eval framework, not an MCP gateway (saturated adjacent categories).
- No tool stubbing / client-side interception in MVP (§2 limitation, documented).
- No hosted anything — that's the open-core monetization layer, later.
