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
- [x] Same check for the OpenAI Agents SDK (`check.py --agent …`, no new harness needed).
      **Records and replays clean**: two steps, zero fingerprint drift, zero network
      calls with the upstream process dead. Two of its defaults had to be turned off in
      the agent, and both are worth knowing about: it speaks the Responses API unless
      told otherwise (agentvcr does not, yet — post-MVP item 3), and its tracing
      uploads to api.openai.com, which would make "the replay reached the network zero
      times" false for a reason unrelated to replay
- [x] `providers/anthropic_messages.py`: `POST /anthropic/v1/messages`, its SSE event
      accumulation, fingerprinting, record + replay parity with OpenAI (shared tests
      parameterized over providers, `tests/test_providers.py`)
- [x] `core/tools.py`: extract tool calls/results by diffing consecutive request
      message lists (DESIGN.md §2); materialize into `tool_calls` at record time;
      `agentvcr show` now renders the interleaved LLM/tool timeline
- [x] `core/differ.py` + `agentvcr diff <a> <b>`: LCS alignment over fingerprints,
      per-step message/response/tool diffs, "diverges at step N" summary

Folded in while building (decisions from the Phase 2 design review, and what the second
wire format turned up):

- [x] **A diff compares behavior, not bytes.** Two live calls to one model differ in
      their response id, timestamp and token counts; reporting those leaves no signal
      for the question actually being asked. Steps are compared on request messages,
      response text, tool calls, tool results and status
- [x] **A changed request message is reported once**, at the step that introduced it.
      Every request carries the whole conversation, so one edited system prompt
      otherwise reappears in all forty steps after it and buries everything else
- [x] `agentvcr diff` exits 1 on a difference, like `diff(1)` — that is what lets a
      strict replay in CI gate on the runs still matching, which is post-MVP item 1
- [x] A run created by `agentvcr run` **learns its provider from its first call**. The
      run exists before the agent has spoken, so the wire format is unknown at creation
      — and `show` reads it to know how to render the run. Invisible while OpenAI was
      the only recording format and also the default
- [x] A stream cut mid-tool-call **keeps its partial arguments**. Anthropic applies a
      tool call's JSON when the block closes, so a client that disconnects first used
      to leave a call recorded with empty arguments — which reads as a call made with
      none, a different bug from the one being debugged

**Done when:** an Anthropic-SDK example records & replays; `show` displays tool
name/args/result rows; `diff` of a run against its own replay reports no divergence,
and against a tweaked re-record pinpoints the diverging step.
**Done** — `examples/plain-loop/agent_anthropic.py` is the same agent as `agent.py` in
the other wire format; `tests/test_anthropic_roundtrip.py` and `tests/test_differ.py`
are the automated form of the check.

## Phase 4 — Fork & edit (week 3)

The killer feature, per DESIGN.md §6.

- [x] `core/forker.py`: create child run (set `parent_run_id` / `fork_step`, **no copied
      steps** — see below), store edits
- [x] Generalize tape resolution in `core/replayer.py`: a fork finds its tape through
      `parent_run_id` the way a replay finds one through `replay_of`
- [x] Fork mode in the proxy: replay prefix → serve edited response at the edit step →
      go live and record the new branch; outbound request patching for tool-result and
      prompt edits
- [x] `agentvcr fork <run> --at N [--edit-response file.json | --edit-tool-result name=file.json]`
      → prints the fork id and the exact re-run command
- [x] Tests: fork with edited assistant response changes the branch; fork with edited
      tool result patches the outbound request (assert on mocked upstream's received
      body); lineage recorded

**A fork run accumulates its own steps; it never starts with a copy of the prefix.**
Position on a tape is how many steps the run being served has recorded so far
(`core/replayer.py`, DESIGN.md §4) — a fork pre-loaded with *k* prefix rows would have
its *first* call answered with tape step *k*, and every step of the prefix replay would
be off by *k*. The prefix is *replayed onto* the child exactly as a replay run
accumulates it, so `fork_step` records where the branch leaves the tape, not how many
rows were pre-inserted.

Folded in while building (decisions from the Phase 3 design review):

- [x] **`patch_tool_result` is a provider method**, the inverse of `extract_tool_results`.
      Rewriting a result is as format-specific as reading one — a `role: "tool"` message
      in one wire format, a `tool_result` block inside a user message in the other — and
      the whole point of `providers/` is that `core/` never learns the difference. The
      contract test asserts the two halves agree: what the patch puts in is what
      extraction reads back
- [x] **The edit is re-applied to every live call, not just the first.** Every request
      carries the whole conversation, so patching only the first one would hand the real
      tool result back to the model on the very next turn
- [x] **Edits of different kinds at one step are refused.** Each kind puts the branch in
      a different place, so mixing them is a contradiction rather than a merge: an edited
      response at step *k* replaces the very tool calls a tool-result edit names, which
      would leave that edit silently inert. Several tool results at one step are the
      exception — a step can call more than one tool
- [x] A `--edit-message I=FILE` prompt edit, since DESIGN.md §6 promised the outbound
      patch mechanism covered system/user prompts and nothing exercised it
- [x] **Re-running a fork that already ran is refused.** Position is how many steps the
      fork has recorded, so a second run would resume in the middle of its own branch
      rather than start it again. The error says how to make a fresh fork instead
- [x] `Store.update_run` serializes `command` like it already did `meta` — the fork
      learns its argv when it is re-run, which is the first time anything updated that
      column rather than writing it at creation
- [x] A fork **does** forward paths the tape never covered (`GET /models`), where a
      replay refuses. A fork is a live run that starts from a tape; it has an upstream
      and credentials by definition
- [x] CI runs `examples/fork-demo/demo.py`. It is the launch GIF's fallback script and
      the only check that goes through the real CLI, a real socket and a real SDK, so it
      cannot be left to rot between now and launch week

**Done when:** the demo script works end-to-end headlessly: example agent "fails" at
step 9 → `agentvcr fork <run> --at 6 --edit-tool-result …` → re-run → passes, and
`agentvcr diff` shows exactly where the branch diverged.
**Done** — `examples/fork-demo/demo.py` is that script, against a scripted upstream that
*reads the tool result*, so the different ending is caused by the edit rather than by a
canned sequence advancing. `tests/test_fork.py` is the fast in-process form of it.

## Phase 5 — Web UI (week 3–4)

- [x] `ui/` Vite + React; build output bundled into the wheel (CI builds it; no Node
      needed at install time)
- [x] `server/api.py`: REST for runs/steps/tool-calls/diff/edits/fork + `POST rerun`
      (spawns the stored command with fork env — only for runs launched via
      `agentvcr run`)
- [x] Views in order of demo value: run list → timeline → step inspector →
      edit-modal-creates-fork (+ Re-run button) → diff side-by-side. **Ship the first
      four**; the side-by-side diff can wait if the week runs out, since `agentvcr diff`
      already covers it in the terminal. **All five shipped** — the diff view is the
      differ's own output in two columns, which was an afternoon, not a week
- [x] Refuse cross-origin calls to `/api/*` (`Origin` check, or a token minted into the
      URL `agentvcr ui` opens), **in the same commit that adds the routes**. Any page the
      user's browser visits can reach `localhost:8484`; the tapes hold whole prompts and
      `POST rerun` spawns a stored command line
- [x] CI: the wheel job needs Node to build `ui/` — today it only installs Python
- [x] `agentvcr ui` opens the browser

Folded in while building (decisions from the Phase 4 design review):

- [x] **The guard is `Origin` *and* `Host`, and no token.** The plan offered either; the
      Origin check won because it needs no state — a token has to be minted into a file
      both processes can read, and it then leaks into shell history and `Referer`. But
      Origin alone loses to DNS rebinding, where the attacker's page *becomes* this
      origin, so a server bound to the loopback also insists it was dialed on one. A
      request with no `Origin` at all (curl, the tests) is allowed through deliberately:
      a program that can run curl can read the tape file directly, so refusing it would
      buy nothing and break every non-browser client
- [x] **`POST rerun` runs the argv stored on the run, and nothing from the request.**
      The browser can choose *which run* to re-run; what that means was decided when the
      run was recorded. It is spawned as an argv list, never through a shell
- [x] **Re-running a recording replays it; re-running a fork branches it — once.** A
      fork's position on the tape is how many steps it has recorded (phase 4), so the
      API refuses a second run of one exactly as the CLI does, and says to fork again
- [x] **A run listing reads narrow columns** (`Store.run_stats`). Every step's request
      holds the whole conversation up to it (DESIGN.md §8), so building a list of counts
      out of full steps reads every tape in the database. `agentvcr runs` had the same
      bug and now shares the fix
- [x] **`agentvcr run` records its working directory** on the run. The Re-run button
      starts the agent where it was first started, rather than wherever `agentvcr serve`
      happens to have been run from
- [x] The run-scoped base URL (`/r/<id>/openai/v1`) now has two launchers — the CLI and
      the Re-run button — so its shape moved to `core/launch.py` rather than being
      spelled out in both
- [x] **A checkout with no built bundle says so.** `/ui` answers with the two commands
      that build it instead of a 404, because a wheel ships it prebuilt and a contributor
      hitting a blank page has no way to tell those two situations apart
- [x] Re-run jobs live in memory and die with the server, and an agent the UI started is
      stopped when the server stops. What the run produced is on the tape either way

**Done when:** the 30-second GIF is recordable entirely in the UI: open failing run →
click step 6 → edit tool result → Re-run → watch the forked run pass → open diff.
**Done** — driven end to end in a real browser against a real server: the recorded run
gives up, the step-0 tool result is edited in the modal (prefilled with what was
recorded), *Run it now* re-runs the agent, step 0 replays in 0ms while step 1 goes live,
the branch answers "The cheapest is B6918 at $289", and the diff names step 0 as where
they parted. `tests/test_api.py` is the automated form of that path.

**Fallback, decided up front:** the launch GIF must not be blocked on the UI. The
headless demo script from Phase 4 is recordable in a terminal and tells the same story;
if Phase 5 slips, the GIF ships from there and the UI lands in v0.2. **Not needed.**

## Phase 6 — Polish & launch (week 4–5)

- [x] `examples/`: OpenAI Agents SDK, LangGraph, CrewAI — each a ≤50-line agent with a
      README showing the one-line `base_url` change (the first two were already proven
      to record and replay in Phase 3; this is polish, not discovery).
      **CrewAI was the one that was not polish**, since nothing had ever pointed it at
      the proxy — and it records and replays clean on the first try (crewai 1.15.17,
      2026-08-24: two steps, zero fingerprint drift, zero network calls with the model
      process dead). What it needed was the same class of thing as the Agents SDK's
      tracing: **two uploads turned off** (`CREWAI_DISABLE_TELEMETRY` before the import,
      `Crew(tracing=False)`), neither about replay, both of which would otherwise keep a
      supposedly offline run talking to the network.
      **Each example is verified by `framework-check/check.py --agent <path>`** rather
      than by hand — the harness already records, kills the upstream, replays and
      compares, so the examples get a no-API-key runner and cannot rot silently. The
      LangGraph one is written on `langchain.agents.create_agent`, not the prebuilt the
      Phase 3 probe used, which LangGraph 1.0 deprecated; it replays identically.
      **CrewAI requires `openai<3` and this repository develops against `openai>=3`.**
      That is not a conflict to resolve — it is the "a proxy is not a library" claim
      showing up as a fact, so the crewai README says so where a reader is most likely
      to be worried about it
- [ ] README: hero GIF, quickstart (`uvx agentvcr serve` + three commands), honest
      limitations section (tool re-execution, concurrency), `.gitignore` note for
      `.agentvcr/`
- [x] `agentvcr rm <run>…` (with `--before <date>`): nothing in phases 0–5 deletes a
      tape and storage is quadratic in run length (DESIGN.md §8), so a first user who
      records a 200-step run has no way out but `rm -rf .agentvcr/`. The schema already
      cascades, so this is small — it just has to exist before launch.
      **Deleting a run that others were derived from is refused without `--recursive`**:
      `ON DELETE SET NULL` would leave a fork in place with its lineage erased and its
      prefix unreplayable, which is a worse outcome than the error. And the delete is
      followed by a `VACUUM` — SQLite keeps freed pages for reuse, so without it the
      200-step run is gone and the file is exactly as large as it was
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
7. **Skip recorded error steps on replay** — a policy that serves past a recorded
   429/500 instead of reproducing the client's retry loop, backoff sleeps included, and
   that decouples a tape from the `max_retries` the recording client happened to use
   (DESIGN.md §5 promises this policy and had nowhere to point).

## Standing decisions (so we don't relitigate)

- Name **agentvcr**; Python; SQLite, no ORM; Apache-2.0; provider logic isolated in
  `providers/`; proxy binds localhost by default; secrets never persisted.
- Every phase lands with tests + a runnable example — the golden record→replay
  round-trip test from Phase 2 stays green forever; it is the product's contract.
