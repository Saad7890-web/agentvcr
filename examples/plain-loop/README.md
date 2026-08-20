# plain-loop

A ~50-line agent: one tool, a `while` loop, the official OpenAI SDK. No framework, no
agentvcr import — it is here to prove the drop-in claim.

`agent_anthropic.py` is the same agent in the Anthropic wire format. The two files
differ only in the SDK and the message shape it speaks; the proxy records, replays and
diffs both through the same recorder, replayer and tool extractor.

## Record it

```bash
pip install openai
export GROQ_API_KEY=...                       # any OpenAI-compatible provider works
export OPENAI_API_KEY=$GROQ_API_KEY

agentvcr serve --preset groq                  # terminal 1
agentvcr run --name flights -- python agent.py   # terminal 2
```

`agentvcr run` creates the run, points `OPENAI_BASE_URL` at
`http://127.0.0.1:8484/r/<run-id>/openai/v1`, and stores the argv — so the agent runs
with **zero** changes. Without the wrapper, change one line instead:

```diff
-client = OpenAI()
+client = OpenAI(base_url="http://127.0.0.1:8484/openai/v1")
```

Then look at what happened:

```bash
agentvcr runs
agentvcr show <run-id>
```

```
run 01K9WQ2M7X4B2Q  mode=record  status=completed
  command python agent.py

STEP  MODEL               HTTP    LATENCY    TOKENS     RESPONSE
0     llama-3.1-8b-inst   200     641ms      412        → search_flights({"origin": "SFO", ...})
      ↳ search_flights    tool    -          -          {"flights": [{"flight": "B6918", "price": 289}]}
1     llama-3.1-8b-inst   200     388ms      503        The cheapest flight from SFO to JFK is B6918 at $289.
```

The tool row is not something the agent reported. `search_flights` ran inside the
`while` loop, in this process, and the proxy never saw it — it reconstructed the run
from the two LLM calls around it (DESIGN.md §2).

## Replay it

```bash
agentvcr run --mode replay --run <run-id> -- python agent.py
```

The agent runs again, unchanged, with every LLM call answered from the tape. No
upstream, no tokens, no network. Then check that the replay really reproduced the
recording:

```bash
agentvcr diff <run-id> <replay-id>     # identical — 2 step(s) aligned, no differences
```

## The Anthropic one

Identical, with `pip install anthropic`, `ANTHROPIC_API_KEY` and:

```bash
agentvcr serve --preset anthropic
agentvcr run --name flights -- python agent_anthropic.py
```

Or without the wrapper:

```diff
-client = Anthropic()
+client = Anthropic(base_url="http://127.0.0.1:8484/anthropic")
```
