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

Early development. Phase 0 (scaffold) is in place: config, SQLite store, CLI and a
health endpoint. Record, replay, fork, diff and the web UI land in phases 1–5 — see
[`PLAN.md`](PLAN.md), and [`DESIGN.md`](DESIGN.md) for the architecture.

## Quickstart (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

agentvcr serve --preset groq      # proxy on http://127.0.0.1:8484
curl -s localhost:8484/healthz

pytest && ruff check .
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
`AGENTVCR_MISMATCH_POLICY`, `AGENTVCR_HOST`, `AGENTVCR_IDLE_TIMEOUT`.

## Privacy

The proxy binds to `127.0.0.1` by default. `Authorization`, `x-api-key` and cookie
headers are forwarded upstream but **never written to disk**. Recordings do contain your
prompts, so add `.agentvcr/` to your `.gitignore`.

## License

Apache-2.0
