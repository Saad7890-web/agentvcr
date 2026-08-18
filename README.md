# agentvcr

**VCR for AI agents.** A drop-in proxy — change only your `base_url` — that records
every LLM call and tool call from any agent run into a local file, then deterministically
replays the run for free, forks it at any step, lets you edit a prompt or a tool result,
re-runs from that exact point, and diffs two runs side by side.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8484/openai/v1")  # was api.openai.com
```

No SDK, no decorators, no framework integration. Anything that speaks the OpenAI or
Anthropic wire format works — including Groq, Gemini, OpenRouter, Ollama and vLLM.

## Status

Early development. **Record mode works** (phase 1): the proxy forwards OpenAI-format
calls to any compatible upstream, streaming included, and writes every step to a local
SQLite tape you can list and inspect. Replay, fork, diff and the web UI land in phases
2–5 — see [`PLAN.md`](PLAN.md), and [`DESIGN.md`](DESIGN.md) for the architecture.

`/anthropic/*` is mounted but forwards unrecorded until phase 3.

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
1     llama-3.1-8b-instant    200   388ms     503     The cheapest SFO→JFK flight is B6918 at $289.
```

`agentvcr run` needs no change to your agent at all: it creates the run and exports
`OPENAI_BASE_URL=http://127.0.0.1:8484/r/<run-id>/openai/v1`, so the run id rides along
in the base URL. Without the wrapper, change one line instead:

```python
client = OpenAI(base_url="http://localhost:8484/openai/v1")
```

A complete example lives in [`examples/plain-loop/`](examples/plain-loop/).

## Commands

| Command | What it does |
|---|---|
| `agentvcr serve` | Start the proxy (`--preset groq\|gemini\|openai\|anthropic`, `--port`, `--db`) |
| `agentvcr run -- <cmd>` | Create a run, point the child at the proxy, record it, store its argv |
| `agentvcr runs` | List recorded runs, newest first |
| `agentvcr show <run>` | Step table for one run (`--json` for the machine-readable form) |

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
