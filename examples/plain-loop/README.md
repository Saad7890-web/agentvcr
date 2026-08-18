# plain-loop

A ~50-line agent: one tool, a `while` loop, the official OpenAI SDK. No framework, no
agentvcr import — it is here to prove the drop-in claim.

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
1     llama-3.1-8b-inst   200     388ms      503        The cheapest flight from SFO to JFK is B6918 at $289.
```

Replaying this tape offline for free is Phase 2 (`PLAN.md`).
