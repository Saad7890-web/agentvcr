# crewai

A CrewAI crew — one agent, one tool, one task, ~50 lines. Recorded and replayed through
agentvcr with no agentvcr import: the crew's `LLM` gets a `base_url`, and that is the
whole integration.

## Record it

```bash
pip install crewai
export GROQ_API_KEY=...                          # any OpenAI-compatible provider works
export OPENAI_API_KEY=$GROQ_API_KEY
export MODEL=llama-3.1-8b-instant

agentvcr serve --preset groq                     # terminal 1
agentvcr run --name flights -- python agent.py   # terminal 2
```

`agentvcr run` exports `OPENAI_BASE_URL`, which the example passes straight to `LLM`.
Without the wrapper, one line changes:

```diff
-llm = LLM(model="openai/llama-3.1-8b-instant")
+llm = LLM(model="openai/llama-3.1-8b-instant", base_url="http://127.0.0.1:8484/openai/v1")
```

Keep the `openai/` prefix on the model name whatever provider is behind the proxy: it is
how CrewAI picks the wire format to speak, and agentvcr's OpenAI endpoint is what is
listening. Which model actually answers is the proxy's business, not the crew's.

## Two uploads to turn off

Neither is about replay, and both would otherwise keep the process talking to the
network during a run that is meant to be entirely offline:

```python
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")   # before importing crewai
crew = Crew(agents=[agent], tasks=[task], tracing=False)    # uploads to crewai.com
```

`OTEL_SDK_DISABLED=true` does the same as the first one. The environment variable is
read as CrewAI's telemetry is set up, which is why it is set above the import rather
than in `main()`.

## Replay it

```bash
agentvcr show <run-id>
agentvcr run --mode replay --run <run-id> -- python agent.py
agentvcr diff <run-id> <replay-id>     # identical — 2 step(s) aligned, no differences
```

```
STEP  MODEL                   HTTP  LATENCY   TOKENS  RESPONSE
0     fake-model              200   7ms       70      → search_flights({"origin": "SFO", "destination": "JFK"})
      ↳ search_flights        tool  -         -       {'flights': [{'flight': 'UA512', 'price': 312}, ...
1     fake-model              200   2ms       102     The cheapest is B6918 at $289.
```

The tool row is reconstructed. `search_flights` ran inside the crew, in this process,
and the proxy saw only the two LLM calls around it (DESIGN.md §2).

Checked on 2026-08-24 against crewai 1.15.17: a crew that calls one tool and answers
records two steps and replays them with the model process killed — same answer, zero
fingerprint drift, zero calls to the network.

## agentvcr is not in your dependency graph

CrewAI requires `openai<3`; this repository develops against `openai>=3`. That is not a
conflict to resolve, because the two never meet: agentvcr is an HTTP proxy in its own
process, and nothing it ships is ever imported by the agent. Install it wherever you
like — `uv tool install agentvcr`, a separate virtualenv, `uvx agentvcr serve` — and
point `base_url` at it.

## No API key

The framework-check harness plays the model, so the whole loop runs offline — from
the repository root:

```bash
python examples/framework-check/check.py --agent examples/crewai/agent.py
```
