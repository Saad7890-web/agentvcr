# examples

Each example is a small agent that runs **unmodified except for its `base_url`**,
pointed at a local `agentvcr serve`.

- [`plain-loop/`](plain-loop/) — a scripted tool loop on the OpenAI SDK. The record/replay
  contract, and the smallest thing that proves the drop-in claim.

Planned (PLAN.md phase 6): `openai-agents-sdk/`, `langgraph/`, `crewai/` — one-line
`base_url` change each.
