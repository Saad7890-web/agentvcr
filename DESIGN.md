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
| `replay` | **no** | Serves recorded responses positionally. Free, offline, deterministic. |
| `fork` | after fork point | Replays the tape up to the fork/edit point, then goes live against the real API (recording the new branch as a child run). |
| `passthrough` | yes | Pure proxy, no recording (escape hatch). |

Mode is selected per-run via the `X-AgentVCR-Mode` header, the `AGENTVCR_MODE` env var
(read by the CLI wrapper), or the proxy's default setting.

## 4. Runs, steps, and how requests join a run

A **run** is one agent execution; a **step** is one LLM call within it (tool steps are
derived, see §2). Two ways a request is assigned to a run:

1. **Explicit (reliable):** client sends `X-AgentVCR-Run: <id>` (both OpenAI and
   Anthropic SDKs support `default_headers`), or the user launches via the CLI wrapper —
   `agentvcr run -- python agent.py` — which creates the run, exports
   `AGENTVCR_RUN`/`AGENTVCR_MODE`, and **stores the command line** (this powers one-click
   re-run from the UI later).
2. **Heuristic (zero-config):** an incoming request whose message list extends the
   message prefix of an active run's last step is chained onto that run; otherwise a new
   run starts. An idle timeout closes runs. Good enough for the drop-in demo; the
   explicit path is what docs recommend for CI.

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
- **Streaming:** if the client asked for `stream: true`, the proxy re-emits the recorded
  response as SSE chunks (synthesized from the stored final message; optionally the raw
  recorded chunk sequence). Recorded streams are always accumulated into a final
  message at record time so both stream and non-stream replay work from one tape.
- **Concurrency note (post-MVP):** parallel LLM calls (multi-agent fan-out) break pure
  positional order → matching falls back to fingerprint-first with position as
  tiebreak. Flagged for milestone 7, not the MVP.

## 6. Fork & edit semantics

**An edit defines the fork point.** For a fork of run R at step *k* with optional edits:

- Steps `1 … k-1`: served from the tape (agent deterministically retraces its path).
- **Edited assistant response at step k:** the proxy serves the *edited* response at
  step k, then goes live for `k+1 …`. The agent genuinely reacts to the edit.
- **Edited tool result of step k:** the tool result physically lives inside request
  `k+1`, and during the replayed prefix the proxy ignores request bodies — so the edit
  is applied as an **outbound request patch**: steps `1 … k` replay from tape, and from
  the first live call onward the proxy rewrites the matching tool-result message in the
  outgoing request body before forwarding upstream. Same mechanism handles system/user
  prompt edits.
- The fork is stored as a child run (`parent_run_id`, `fork_step`, edits) — so the diff
  view can show exactly what changed and lineage forms a tree.

This yields the 30-second demo: *agent fails at step 9 → open step 6 → edit the tool
result → re-run → agent passes.* Re-running executes the user's own agent process with
`AGENTVCR_MODE=fork AGENTVCR_RUN=<fork-id>`; if the original run was launched via
`agentvcr run -- …`, the stored command makes this a single button in the UI.

## 7. Diffing

Align two runs (LCS over step fingerprints, positional fallback), then per aligned
step: message-level diff of requests, text/JSON diff of responses, and tool
name/args/result diffs. Two consumers:

- `agentvcr diff <runA> <runB>` — colored terminal summary: *"runs diverge at step 6
  (tool `search_flights` returned different results)"*.
- UI side-by-side view with per-step drill-down.

This is also the engine behind the eventual CI feature ("your prompt change altered
the agent's decisions in 12 of 40 recorded runs").

## 8. Storage schema (SQLite)

`.agentvcr/agentvcr.db` in the project directory (overridable via `--db` /
`AGENTVCR_DB`). Plain `sqlite3` with a tiny migration runner — no ORM.

```sql
runs(
  id TEXT PRIMARY KEY,            -- short ulid
  name TEXT,                      -- human label, defaults to timestamp
  created_at TEXT,
  mode TEXT,                      -- record | replay | fork
  parent_run_id TEXT,             -- fork lineage
  fork_step INTEGER,
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
  diverged INTEGER DEFAULT 0,
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
│   │   └── api.py                # REST for the UI: runs, steps, edits, fork, rerun
│   ├── core/
│   │   ├── store.py              # SQLite layer + migrations
│   │   ├── models.py             # Run / Step / ToolCall / Edit dataclasses
│   │   ├── recorder.py           # capture, redaction, run-grouping heuristic
│   │   ├── replayer.py           # positional matching, policies, SSE synthesis
│   │   ├── forker.py             # fork creation, edit application, request patching
│   │   ├── tools.py              # tool-call extraction from message diffs
│   │   └── differ.py             # run alignment + step diffs
│   └── providers/
│       ├── base.py               # Provider interface: normalize, fingerprint,
│       │                         #   extract_tool_calls, synthesize_stream
│       ├── openai_chat.py        # /v1/chat/completions (covers Groq, Gemini-compat,
│       │                         #   OpenRouter, Ollama, vLLM)
│       └── anthropic_messages.py # /v1/messages
├── ui/                           # Vite + React source → built into server/static/
├── examples/                     # openai-agents-sdk/, langgraph/, crewai/, plain-loop/
├── tests/                        # incl. golden record→replay round-trip tests
└── .github/workflows/ci.yml
```

**Provider abstraction is the load-bearing interface.** Recording/replay/fork logic is
format-agnostic; each provider module only knows how to normalize a body, fingerprint
it, pull tool calls/results out of messages, and synthesize SSE chunks. Adding
`/v1/responses` or a new provider later touches only `providers/`.

**Upstream routing:** path prefix picks the wire format; the actual upstream host is
config (`agentvcr.toml` or flags), with presets so free-tier demos are one flag:
`agentvcr serve --preset groq` (Groq's OpenAI-compatible endpoint) or `--preset gemini`.
Replays need no upstream at all — the whole product is demoable on free tiers, and
replays cost zero tokens by construction.

## 10. Web UI (local, bundled)

Served by the same process at `http://localhost:8484/ui`. Views:

1. **Run list** — table with status, model, steps, tokens, cost estimate, fork lineage tree.
2. **Timeline** — the hero view: interleaved LLM/tool steps; failed/diverged steps flagged.
3. **Step inspector** — pretty-rendered messages, raw JSON toggle, usage/latency.
4. **Edit → fork** — edit a response or tool result in place → creates the fork run and
   shows the re-run command (or a *Re-run* button when the original command is stored).
5. **Diff** — pick two runs, side-by-side with divergence markers.

Static build committed nowhere — built in CI and bundled into the wheel, so
`pip install agentvcr` ships the UI with no Node required at install time.

## 11. Security & privacy

- Binds to `127.0.0.1` by default; `--host` opt-in for containers.
- Header redaction list (`authorization`, `x-api-key`, `api-key`, cookies) applied
  before persistence; regex-based body redaction as a follow-up.
- Recordings may contain sensitive prompt data — README ships a recommended
  `.gitignore` entry for `.agentvcr/`.

## 12. Explicit non-goals (MVP)

- Not an observability platform (no dashboards-of-averages, no OTel backend — though an
  OTel *exporter* is a fine later add-on).
- Not an eval framework, not an MCP gateway (saturated adjacent categories).
- No tool stubbing / client-side interception in MVP (§2 limitation, documented).
- No hosted anything — that's the open-core monetization layer, later.
