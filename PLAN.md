# agentvcr — Implementation Plan

Step-by-step build order. Each phase ends with something runnable and a concrete
acceptance check — no phase depends on a later one. Target: MVP (phases 0–5) in
3–4 weeks, launch in week 4–5.

See `DESIGN.md` for the architecture these phases implement.

---

## Phase 0 — Scaffold (day 1–2)

Repo skeleton and plumbing so every later phase is just filling in modules.

- [x] `pyproject.toml` (hatchling, `requires-python >= 3.10`), deps: `fastapi`,
      `uvicorn`, `httpx`, `typer`; dev deps: `pytest`, `pytest-asyncio`, `respx`, `ruff`
- [x] `src/agentvcr/` package layout from DESIGN.md §9, with empty modules + docstrings
- [x] `config.py`: settings from flags/env/`agentvcr.toml` (port, db path, upstream map,
      mode default, mismatch policy); presets for `openai`, `anthropic`, `groq`, `gemini`
- [x] `core/store.py`: SQLite open/migrate (schema from DESIGN.md §8), `core/models.py`
      dataclasses
- [x] `cli.py`: `agentvcr serve` boots FastAPI app with a `/healthz` route
- [x] CI: ruff + pytest on push; Apache-2.0 LICENSE; README stub with the one-liner

**Done when:** `uvx --from . agentvcr serve` starts and `/healthz` answers; CI is green.
**Done.**

## Phase 1 — Record mode, OpenAI format (week 1)

The proxy earns its keep: transparent passthrough that records.

- [x] `server/proxy.py`: `POST /openai/v1/chat/completions` → forward to configured
      upstream with client's auth headers; stream SSE through to the client while
      accumulating the final message (`providers/openai_chat.py` does the accumulation)
- [x] Generic passthrough for other `/openai/v1/*` paths (e.g. `/models`) — proxied,
      not recorded
- [x] `core/recorder.py`: persist step (redacted request, final response, optional raw
      chunks, fingerprint, model, usage, latency); header redaction list
- [x] Run assignment: `X-AgentVCR-Run` header → run; else grouping heuristic
      (message-prefix chaining + idle timeout) per DESIGN.md §4
- [x] `agentvcr run [--name X] -- <cmd>`: creates run, exports `AGENTVCR_RUN` /
      `AGENTVCR_MODE` / `OPENAI_BASE_URL`, stores argv on the run
- [x] `agentvcr runs` (list) and `agentvcr show <run>` (step table) — plain text is fine
- [x] Tests with `respx`-mocked upstream: non-stream and stream recording, redaction,
      run grouping

Folded in while building (decisions from the Phase 0 design review):

- [x] `passthrough` mode — it was in DESIGN.md §3 and `config.MODES` but no phase owned it
- [x] Errors recorded as steps, so an SDK retry sequence stays on the tape (DESIGN.md §5)
- [x] Run id carried in the base URL (`/r/<id>/…`), the only explicit mechanism that works
      for agents that cannot set headers — this is what makes `agentvcr run` zero-change
- [x] Store: serialize writes behind a lock (one connection, concurrent proxy calls) and
      refuse a tape written by a newer schema

**Done when:** a scripted tool-loop agent (`examples/plain-loop/`, OpenAI SDK pointed
at the proxy, Groq free tier as upstream) runs unmodified except `base_url`, and
`agentvcr show` displays its calls. **No API key ever appears in the DB** (test asserts).
**Done** — `tests/test_record_roundtrip.py` is the automated form of this check.

## Phase 2 — Replay mode (week 1–2)

The zero-token payoff; this is what makes the project real.

- [x] `core/replayer.py`: positional matching, fingerprint comparison, policies
      `warn` / `strict` / `live-on-miss` (DESIGN.md §5); mark diverged steps
- [x] SSE replay: raw recorded chunks when they were kept, synthesis from the stored
      final message otherwise; one tape answers streaming and non-streaming clients
- [x] Structured error when the tape runs out of steps
- [x] `agentvcr run --mode replay --run <id> -- <cmd>` wires it together
- [x] **Golden round-trip test:** record the example agent against a mocked upstream →
      replay with the network fully disabled → transcript and final answer identical

Folded in while building (decisions from the Phase 1 design review):

- [x] **A replay is a run of its own** (`replay_of`, DESIGN.md §4). Without it there was
      nowhere to put divergence but the tape itself, and nothing to diff a recording
      against in Phase 3
- [x] Replay never contacts an upstream, not even for paths it cannot serve (an
      unrecorded `GET /models` under replay is a structured 501, not a forward)
- [x] An unknown `X-AgentVCR-Mode` is a 400 — it used to fall through to an unrecorded
      passthrough, which looks exactly like recording that silently kept nothing
- [x] Run grouping: an exact repeat of a request chains onto its run only when the last
      step *failed* (an SDK retry). Two copies of one agent starting from the same
      prompt used to land on a single tape

**Done when:** the Phase 1 example replays offline (`--mode replay`) with zero upstream
requests, reproducing the recorded transcript step for step — byte-identically where a
raw chunk tape exists. **Done** — `tests/test_replay.py` replays with respx registering
no routes at all, so any upstream call raises rather than quietly succeeding.

## Phase 3 — Anthropic format + tool timeline + diff (week 2)

- [x] **Framework reality check, first thing.** Positional matching assumes strictly
      sequential LLM calls, and a framework is where that assumption breaks. Pulled
      ahead of Phase 6 so it cannot ambush launch week: `examples/framework-check/`
      records and replays a real LangGraph agent against a scripted upstream that is
      then killed. **Sequential agents replay perfectly, with zero divergence; parallel
      fan-out replays nondeterministically (raced the other way in 2 of 5 runs) and is
      flagged on every step when it does** (see that README, and DESIGN.md §5)
- [ ] Same check for the OpenAI Agents SDK (`check.py --agent …`, no new harness needed)
- [ ] `providers/anthropic_messages.py`: `POST /anthropic/v1/messages`, its SSE event
      accumulation, fingerprinting, record + replay parity with OpenAI (shared tests
      parameterized over providers)
- [ ] `core/tools.py`: extract tool calls/results by diffing consecutive request
      message lists (DESIGN.md §2); materialize into `tool_calls` at record time;
      `agentvcr show` now renders the interleaved LLM/tool timeline
- [ ] `core/differ.py` + `agentvcr diff <a> <b>`: LCS alignment over fingerprints,
      per-step message/response/tool diffs, "diverges at step N" summary

**Done when:** an Anthropic-SDK example records & replays; `show` displays tool
name/args/result rows; `diff` of a run against its own replay reports no divergence,
and against a tweaked re-record pinpoints the diverging step.

## Phase 4 — Fork & edit (week 3)

The killer feature, per DESIGN.md §6.

- [ ] `core/forker.py`: create child run (copy prefix, set `parent_run_id`/`fork_step`),
      store edits
- [ ] Fork mode in the proxy: replay prefix → serve edited response at the edit step →
      go live and record the new branch; outbound request patching for tool-result and
      prompt edits
- [ ] `agentvcr fork <run> --at N [--edit-response file.json | --edit-tool-result name=file.json]`
      → prints the fork id and the exact re-run command
- [ ] Tests: fork with edited assistant response changes the branch; fork with edited
      tool result patches the outbound request (assert on mocked upstream's received
      body); lineage recorded

**Done when:** the demo script works end-to-end headlessly: example agent "fails" at
step 9 → `agentvcr fork <run> --at 6 --edit-tool-result …` → re-run → passes, and
`agentvcr diff` shows exactly where the branch diverged.

## Phase 5 — Web UI (week 3–4)

- [ ] `ui/` Vite + React; build output bundled into the wheel (CI builds it; no Node
      needed at install time)
- [ ] `server/api.py`: REST for runs/steps/tool-calls/diff/edits/fork + `POST rerun`
      (spawns the stored command with fork env — only for runs launched via
      `agentvcr run`)
- [ ] Views in order of demo value: run list → timeline → step inspector →
      edit-modal-creates-fork (+ Re-run button) → diff side-by-side. **Ship the first
      four**; the side-by-side diff can wait if the week runs out, since `agentvcr diff`
      already covers it in the terminal
- [ ] CI: the wheel job needs Node to build `ui/` — today it only installs Python
- [ ] `agentvcr ui` opens the browser

**Done when:** the 30-second GIF is recordable entirely in the UI: open failing run →
click step 6 → edit tool result → Re-run → watch the forked run pass → open diff.

**Fallback, decided up front:** the launch GIF must not be blocked on the UI. The
headless demo script from Phase 4 is recordable in a terminal and tells the same story;
if Phase 5 slips, the GIF ships from there and the UI lands in v0.2.

## Phase 6 — Polish & launch (week 4–5)

- [ ] `examples/`: OpenAI Agents SDK, LangGraph, CrewAI — each a ≤50-line agent with a
      README showing the one-line `base_url` change (the first two were already proven
      to record and replay in Phase 3; this is polish, not discovery)
- [ ] README: hero GIF, quickstart (`uvx agentvcr serve` + three commands), honest
      limitations section (tool re-execution, concurrency), `.gitignore` note for
      `.agentvcr/`
- [ ] Package QA: `pip install agentvcr` from TestPyPI on a clean machine; version
      pinning; `--help` text pass
- [ ] Publish to PyPI; tag v0.1.0
- [ ] Show HN + r/LocalLLaMA posts (lead with the GIF); cross-post examples to the
      frameworks' discussion boards

**Done when:** a stranger can go from `pip install agentvcr` to their first replay in
under 5 minutes using only the README.

---

## Post-MVP backlog (only after launch feedback)

Ordered by expected pull, not effort:

1. **CI regression replays** — GitHub Action: replay N recorded runs in `strict` mode
   against a changed prompt/codebase; report "decisions changed in 12 of 40 runs"
   (the differ already does the work). First paid-tier candidate.
2. **Concurrency-safe matching** — fingerprint-first matching for parallel/multi-agent
   fan-out (DESIGN.md §5 note).
3. **OpenAI Responses API** (`/v1/responses`) provider module.
4. **Body redaction rules** (regex/JSONPath) for sensitive prompt data.
5. **Optional tool-stubbing shim** — tiny client wrapper for users who need tools
   frozen during replay, not just LLM calls.
6. **Export/import tapes** (single-file `.vcr.json`) → foundation for shareable run
   links, i.e. the hosted open-core product.

## Standing decisions (so we don't relitigate)

- Name **agentvcr**; Python; SQLite, no ORM; Apache-2.0; provider logic isolated in
  `providers/`; proxy binds localhost by default; secrets never persisted.
- Every phase lands with tests + a runnable example — the golden record→replay
  round-trip test from Phase 2 stays green forever; it is the product's contract.
