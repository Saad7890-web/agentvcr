# examples

Each example is a small agent that runs **unmodified except for its `base_url`**,
pointed at a local `agentvcr serve`.

- [`plain-loop/`](plain-loop/) — a scripted tool loop, once on the OpenAI SDK and once
  on the Anthropic one. The record/replay contract, and the smallest thing that proves
  the drop-in claim.
- [`fork-demo/`](fork-demo/) — the whole story in one command and no API key: an agent
  gives up because a tool came back empty, one tool result is edited, and the same
  agent books a flight. Record → fork → re-run → diff.
- [`framework-check/`](framework-check/) — not a demo: the harness that records and
  replays real agents against a scripted upstream that is then killed, to find out where
  positional replay breaks. It found one place.

The other three are one agent written three times — one tool, one question, the
same two steps on the tape — so the diff between them is the framework and nothing else:

- [`langgraph/`](langgraph/) — `create_agent`, and the fan-out caveat.
- [`openai-agents-sdk/`](openai-agents-sdk/) — and the two SDK defaults to change.
- [`crewai/`](crewai/) — and the two uploads to turn off.

Each records and replays with the model process dead, verified by `framework-check/`,
which is also how to run any of them with no API key at all.
