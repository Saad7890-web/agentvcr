# Framework reality check

Positional replay assumes an agent makes **the same LLM calls in the same order** every
time it runs. That assumption is cheap to state and expensive to be wrong about, and a
real framework is where it goes to die. This harness tries to break it — on purpose,
early, with no API key: `fake_upstream.py` plays the model.

```bash
pip install langgraph langchain-openai        # not agentvcr dependencies

python examples/framework-check/check.py                      # sequential ReAct agent
python examples/framework-check/check.py \
    --agent examples/framework-check/parallel_agent.py \
    --expect divergence                                       # parallel fan-out
```

Each run records the agent through the proxy, **kills the upstream process**, replays
the tape, and compares. Killing the upstream is the part that matters: it makes "replay
never touches the network" a fact about the run rather than a claim about the code.

## What it found (2026-08-19, langgraph 1.2.11, langchain-openai 1.5.2)

**Sequential agents replay perfectly.** A `create_react_agent` tool loop records two
steps, and replays them with the model process dead: same answer, same responses step
for step, and *zero* fingerprint divergence — LangChain rebuilds byte-stable request
bodies across runs, so even the advisory check is clean. No proxy-specific code in the
agent; only `base_url`.

**Parallel fan-out is a coin toss — and every bad flip is caught.** A graph with two
branches leaving `START` issues both LLM calls in one superstep, so their arrival order
at the proxy is a race. Over five runs the replay won the race the *other* way twice,
and when it did, each branch received the other branch's answer:

```
recorded:  SFO→AUS -> The cheapest is WN1104 at $178.     SFO→JFK -> (empty)
replayed:  SFO→AUS -> (empty)                             SFO→JFK -> The cheapest is WN1104 at $178.
```

The other three runs replayed correctly, by luck. So the honest claim is not "fan-out
always breaks" — it is that fan-out replays **nondeterministically**, which for a
debugging tool is the same thing as broken.

What holds in every run is the invariant the check actually asserts: a replay whose
requests drifted from the tape is *always* flagged. Both steps fail the fingerprint
check, the replay run is marked `diverged`, and `agentvcr runs` shows it. Positional
replay can be wrong for concurrent calls; it is never quietly wrong. Fingerprint-first
matching for fan-out is DESIGN.md §5's post-MVP milestone 7 — this is the evidence for
what happens until then.

Because it is a race, run the parallel probe a few times before concluding anything
from a single green result.

## Why these files are not `examples/`

They are instruments, not demos. `examples/` gets the polished ≤50-line agents in
Phase 6; these exist to answer a question, and they stay so the answer can be
re-checked when a framework releases a new major version.

`langgraph_agent.py` uses `langgraph.prebuilt.create_react_agent`, which LangGraph 1.0
deprecated in favor of `langchain.agents.create_agent` — the replacement lives in the
`langchain` package, which this check deliberately does not install. The deprecation
warning in the output is expected.
