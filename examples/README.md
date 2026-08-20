# examples

Each example is a small agent that runs **unmodified except for its `base_url`**,
pointed at a local `agentvcr serve`.

- [`plain-loop/`](plain-loop/) — a scripted tool loop, once on the OpenAI SDK and once
  on the Anthropic one. The record/replay contract, and the smallest thing that proves
  the drop-in claim.
- [`framework-check/`](framework-check/) — not a demo: the harness that records and
  replays real LangGraph and OpenAI Agents SDK agents against a scripted upstream that
  is then killed, to find out where positional replay breaks. It found one place.

Planned (PLAN.md phase 6): `openai-agents-sdk/`, `langgraph/`, `crewai/` — one-line
`base_url` change each. The first two are already proven to record and replay by
`framework-check/`; phase 6 is the polish.
