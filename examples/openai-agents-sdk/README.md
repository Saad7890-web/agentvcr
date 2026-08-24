# openai-agents-sdk

An OpenAI Agents SDK agent — one agent, one tool, ~50 lines. It records and replays
through agentvcr with no agentvcr import, and no change to the agent itself.

## Record it

```bash
pip install openai-agents
export GROQ_API_KEY=...                          # any OpenAI-compatible provider works
export OPENAI_API_KEY=$GROQ_API_KEY
export MODEL=llama-3.1-8b-instant

agentvcr serve --preset groq                     # terminal 1
agentvcr run --name flights -- python agent.py   # terminal 2
```

## The three SDK lines

This is the one framework where the base URL is not a single argument, because the SDK
builds its own client rather than reading `OPENAI_BASE_URL`:

```python
set_default_openai_client(AsyncOpenAI(base_url=os.environ["OPENAI_BASE_URL"]), use_for_tracing=False)
set_default_openai_api("chat_completions")   # the SDK defaults to /v1/responses
set_tracing_disabled(True)                   # tracing uploads to api.openai.com
```

Two of those are SDK defaults worth knowing about whether or not you use agentvcr:

- it speaks the **Responses API** (`/v1/responses`) unless told otherwise. agentvcr does
  not record that wire format yet — DESIGN.md §12, post-MVP item 3 — so this is the one
  framework default that currently has to change;
- its **tracing uploads to api.openai.com**, which needs a real key and would keep the
  process talking to OpenAI during a replay that is otherwise entirely offline.

## Replay it

```bash
agentvcr show <run-id>
agentvcr run --mode replay --run <run-id> -- python agent.py
agentvcr diff <run-id> <replay-id>     # identical — 2 step(s) aligned, no differences
```

```
STEP  MODEL                   HTTP  LATENCY   TOKENS  RESPONSE
0     fake-model              200   9ms       70      → search_flights({"origin": "SFO", "destination": "JFK"})
      ↳ search_flights        tool  -         -       {'flights': [{'flight': 'UA512', 'price': 312}, ...
1     fake-model              200   2ms       102     The cheapest is B6918 at $289.
```

The tool row is reconstructed: `search_flights` ran inside the `Runner`, in this
process, and the proxy only ever saw the two LLM calls around it (DESIGN.md §2).

Sequential runs like this one replay with zero divergence and zero network calls, with
the upstream process dead — measured by [`../framework-check/`](../framework-check/),
which also documents where that stops being true (concurrent fan-out).

## No API key

The check harness plays the model, so the whole loop runs offline — from the
repository root:

```bash
python examples/framework-check/check.py --agent examples/openai-agents-sdk/agent.py
```
