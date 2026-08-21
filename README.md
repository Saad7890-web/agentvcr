# agentvcr

**VCR for AI agents.** A drop-in proxy — change only your `base_url` — that records
every LLM call and tool call from any agent run into a local file, then deterministically
replays the run for free, forks it at any step, lets you edit a prompt or a tool result,
re-runs from that exact point, and diffs two runs side by side.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8484/openai/v1")  # was api.openai.com
```

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8484/anthropic")  # was api.anthropic.com
```

No SDK, no decorators, no framework integration. Anything that speaks the OpenAI or
Anthropic wire format works — including Groq, Gemini, OpenRouter, Ollama and vLLM.

## Status

Early development. **Record, replay, fork and diff work** (phases 1–4), in both the
OpenAI and Anthropic wire formats: the proxy forwards calls to any compatible upstream,
streaming included, writes every step to a local SQLite tape, reconstructs the tool
timeline from those steps alone, replays the tape offline for free without ever
contacting an upstream, and branches a run at any step with an edited response, tool
result or prompt. The web UI lands in phase 5 — see [`PLAN.md`](PLAN.md), and
[`DESIGN.md`](DESIGN.md) for the architecture.

## Quickstart

```bash
pip install -e '.[dev]'        # from a checkout; the PyPI release lands at v0.1.0

agentvcr serve --preset groq                        # terminal 1
agentvcr run --name flights -- python agent.py      # terminal 2

agentvcr runs
agentvcr show <run-id>
```

```
STEP  MODEL                   HTTP  LATENCY   TOKENS  RESPONSE
0     llama-3.1-8b-instant    200   641ms     412     → search_flights({"origin": "SFO", ...})
      ↳ search_flights        tool  -         -       {"flights": [{"flight": "B6918", "price": 289}]}
1     llama-3.1-8b-instant    200   388ms     503     The cheapest SFO→JFK flight is B6918 at $289.
```

The proxy never saw that tool run — agents execute tools locally. It reconstructed it
from the LLM boundary alone: the call is in step 0's response, the result is in step 1's
request. No instrumentation, no SDK, nothing added to your agent.

`agentvcr run` needs no change to your agent at all: it creates the run and exports
`OPENAI_BASE_URL=http://127.0.0.1:8484/r/<run-id>/openai/v1`, so the run id rides along
in the base URL. Without the wrapper, change one line instead:

```python
client = OpenAI(base_url="http://localhost:8484/openai/v1")
```

A complete example lives in [`examples/plain-loop/`](examples/plain-loop/).

## Replay

Replaying a recorded run costs nothing and needs no network:

```bash
agentvcr run --mode replay --run <run-id> -- python agent.py
```

Your agent runs again, unchanged, and every LLM call is answered from the tape by
position — the *N*th call gets the *N*th recorded response. The replay is recorded as a
run of its own, linked to the tape it came from, so the original recording is never
written to and the two can be diffed later.

Each request's fingerprint is also compared against the tape. A mismatch means the
agent has drifted from what was recorded, and what happens then is
`mismatch_policy`:

| Policy | On a mismatch |
|---|---|
| `warn` (default) | serve the recorded response anyway, flag the step and the run as diverged |
| `strict` | refuse with a `409` — the setting for CI |
| `live-on-miss` | stop replaying and go live from that step onward (auto-fork) |

Known limits: replay is deterministic at the LLM boundary, but your **tools still
execute locally** — a tool with side effects will re-fire. And agents that issue LLM
calls *concurrently* replay in whatever order the race lands, which shows up as
divergence rather than as a silent wrong answer. Both are measured in
[`examples/framework-check/`](examples/framework-check/).

## Fork & edit

Replay answers *what did the agent do?* A fork answers **what would it have done?**

```bash
agentvcr fork <run-id> --at 6 --edit-tool-result search_flights=flights.json
agentvcr run --mode fork --run <fork-id> -- python agent.py
```

Steps before the edit replay off the tape for free; from the edit onward the agent makes
real calls, recorded onto the fork. Your agent is not touched — the edit is applied at
the proxy, so the model reacts to it while the agent's own code, tools and prompt stay
exactly as they were.

**The edit is what defines the fork point.** Three kinds, and each puts the branch in
the place its own mechanics require:

| Flag | What happens |
|---|---|
| `--edit-response FILE` | steps `0…N-1` replay, step `N` is answered with your response, `N+1…` are live |
| `--edit-tool-result NAME=FILE` | steps `0…N` replay; the result rides inside request `N+1`, so it is rewritten on the way upstream |
| `--edit-message I=FILE` | a prompt edit: steps `0…N-1` replay, and request `N` goes live carrying the new message `I` |

A fork is a run of its own — `parent_run_id`, `fork_step` and its edits — so lineage
forms a tree and `agentvcr diff <run> <fork>` names exactly where the branch left the
tape. It starts empty and collects its steps as it runs: the prefix is *replayed onto*
it, never copied into it.

[`examples/fork-demo/`](examples/fork-demo/) runs the whole story end to end with no API
key: an agent gives up because its search returned nothing, one tool result is edited,
and the same agent books a flight.

## Diff

Two runs, compared step by step:

```bash
agentvcr diff <run-a> <run-b>
```

```
diff 01K9WQ2M7X4B2Q → 01K9WQ8N3P1D7F
  a  01K9WQ2M7X4B2Q      record     completed  2 step(s)  flights
  b  01K9WQ8N3P1D7F      record     completed  2 step(s)  flights-after-prompt-edit

  step 0  tool search_flights returned different results
    - {"flights": [{"flight": "B6918", "price": 289}]}
    + {"flights": []}

runs diverge at step 0 (tool search_flights returned different results); 1/2 step(s) identical
```

Steps are aligned by fingerprint, so a run that gained or lost a call still lines up
around the change instead of reporting everything after it as different. The comparison
is about **behavior, not bytes**: request messages, response text, tool calls, tool
results and HTTP status — never the response ids, timestamps and token counts that
differ between any two live calls.

`diff` exits 1 when the runs differ, like `diff(1)`. A recording diffed against its own
replay should always come back identical — that is what proves the replay reproduced
the run rather than merely not crashing.

## Commands

| Command | What it does |
|---|---|
| `agentvcr serve` | Start the proxy (`--preset groq\|gemini\|openai\|anthropic`, `--port`, `--db`) |
| `agentvcr run -- <cmd>` | Create a run, point the child at the proxy, record it, store its argv |
| `agentvcr run --mode replay --run <id> -- <cmd>` | Replay a tape offline; no upstream, no tokens |
| `agentvcr fork <run> --at N` | Branch a run at step N with an edit; prints the command to re-run it |
| `agentvcr run --mode fork --run <id> -- <cmd>` | Re-run the agent against a fork: replay the prefix, then go live |
| `agentvcr runs` | List recorded runs, newest first |
| `agentvcr show <run>` | Interleaved LLM/tool timeline for one run (`--json` for the machine-readable form) |
| `agentvcr diff <a> <b>` | Diff two runs step by step; exits 1 when they differ |

## Development

```bash
pytest && ruff check . && ruff format --check .
```

## Configuration

Settings resolve as **defaults < `agentvcr.toml` < environment < CLI flags**.

```toml
# agentvcr.toml — the nearest one walking up from the working directory is used
[agentvcr]
port = 8484
mode = "record"              # record | replay | fork | passthrough
db_path = ".agentvcr/agentvcr.db"
preset = "groq"              # openai | anthropic | groq | gemini
mismatch_policy = "warn"     # warn | strict | live-on-miss

[agentvcr.upstreams]
openai = "http://localhost:11434/v1"
```

Environment equivalents: `AGENTVCR_PORT`, `AGENTVCR_DB`, `AGENTVCR_MODE`,
`AGENTVCR_PRESET`, `AGENTVCR_UPSTREAM_OPENAI`, `AGENTVCR_UPSTREAM_ANTHROPIC`,
`AGENTVCR_MISMATCH_POLICY`, `AGENTVCR_HOST`, `AGENTVCR_IDLE_TIMEOUT`,
`AGENTVCR_RECORD_CHUNKS`.

## Privacy

The proxy binds to `127.0.0.1` by default. `Authorization`, `x-api-key` and cookie
headers are forwarded upstream but **never written to disk** — a test asserts the key
appears nowhere in the database file. Recordings do contain your prompts, so add
`.agentvcr/` to your `.gitignore`.

## License

Apache-2.0
