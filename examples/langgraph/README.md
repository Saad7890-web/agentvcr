# langgraph

A LangGraph agent — `create_agent`, one tool, ~45 lines. It runs through agentvcr
unmodified except for its `base_url`, and there is no `import agentvcr` in it.

## Record it

```bash
pip install langgraph langchain langchain-openai
export GROQ_API_KEY=...                          # any OpenAI-compatible provider works
export OPENAI_API_KEY=$GROQ_API_KEY
export MODEL=llama-3.1-8b-instant

agentvcr serve --preset groq                     # terminal 1
agentvcr run --name flights -- python agent.py   # terminal 2
```

`agentvcr run` creates the run, exports `OPENAI_BASE_URL`, and stores the argv, so the
agent needs **zero** changes. Without the wrapper, change one line instead:

```diff
-model = ChatOpenAI(model="llama-3.1-8b-instant")
+model = ChatOpenAI(model="llama-3.1-8b-instant", base_url="http://127.0.0.1:8484/openai/v1")
```

LangChain reads `OPENAI_API_BASE` where the OpenAI SDK reads `OPENAI_BASE_URL`; the
example honors both, which is why the wrapper works without setting anything extra.

## Replay it

```bash
agentvcr show <run-id>
agentvcr run --mode replay --run <run-id> -- python agent.py
agentvcr diff <run-id> <replay-id>     # identical — 2 step(s) aligned, no differences
```

```
STEP  MODEL                   HTTP  LATENCY   TOKENS  RESPONSE
0     fake-model              200   13ms      70      → search_flights({"origin": "SFO", "destination": "JFK"})
      ↳ search_flights        tool  -         -       {'flights': [{'flight': 'UA512', 'price': 312}, ...
1     fake-model              200   2ms       102     The cheapest is B6918 at $289.
```

The `search_flights` row is not something the agent reported. The tool ran inside the
graph, in this process, and the proxy never saw it — the timeline is reconstructed from
the two LLM calls around it (DESIGN.md §2).

## What replays, and what does not

A sequential tool loop like this one replays **exactly**: same answer, same responses
step for step, zero fingerprint drift, with the model process dead. Fan-out does not.
A graph with two branches leaving `START` issues both LLM calls in one superstep, so
their arrival order at the proxy is a race, and a replay can win it the other way round
— which hands each branch the other one's answer. It is always *flagged* when that
happens (the run is marked `diverged`), never quietly wrong.

[`../framework-check/`](../framework-check/) is the harness that measured this, against
a scripted upstream it then kills; DESIGN.md §5 is the design note, and fingerprint-first
matching for fan-out is on the post-MVP backlog (PLAN.md).

## No API key

The check harness plays the model, so the whole loop runs offline — from the
repository root:

```bash
python examples/framework-check/check.py --agent examples/langgraph/agent.py
```
