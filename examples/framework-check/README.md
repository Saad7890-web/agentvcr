# Framework reality check

Positional replay assumes an agent makes **the same LLM calls in the same order** every
time it runs. That assumption is cheap to state and expensive to be wrong about, and a
real framework is where it goes to die. This harness tries to break it — on purpose,
early, with no API key: `fake_upstream.py` plays the model.

```bash
pip install langgraph langchain-openai openai-agents   # not agentvcr dependencies

python examples/framework-check/check.py                      # sequential ReAct agent
python examples/framework-check/check.py \
    --agent examples/framework-check/agents_sdk_agent.py      # OpenAI Agents SDK
python examples/framework-check/check.py \
    --agent examples/framework-check/parallel_agent.py \
    --expect divergence                                       # parallel fan-out
```

Each run records the agent through the proxy, **kills the upstream process**, replays
the tape, and compares. Killing the upstream is the part that matters: it makes "replay
never touches the network" a fact about the run rather than a claim about the code.

## What it found (2026-08-20, langgraph 1.2.11, langchain-openai 1.5.2, openai-agents 0.22.0)

**Sequential agents replay perfectly, in both frameworks.** A LangGraph
`create_react_agent` tool loop records two steps, and replays them with the model
process dead: same answer, same responses step for step, and *zero* fingerprint
divergence — LangChain rebuilds byte-stable request bodies across runs, so even the
advisory check is clean. The OpenAI Agents SDK does the same, with the same result. No
proxy-specific code in either agent; only `base_url`.

Two Agents SDK defaults have to be turned off, and neither is about replay:

- it calls **`/v1/responses`** unless told `set_default_openai_api("chat_completions")`.
  agentvcr does not speak the Responses API yet (DESIGN.md §12; post-MVP item 3), so
  this is the one framework default that currently needs changing;
- its **tracing uploads to api.openai.com**, which needs a real key and would make "the
  replay reached the network zero times" false for a reason that has nothing to do with
  the tape.

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

They are instruments, not demos. The polished agents live in
[`../langgraph/`](../langgraph/), [`../openai-agents-sdk/`](../openai-agents-sdk/) and
[`../crewai/`](../crewai/); these exist to answer a question, and they stay so the
answer can be re-checked when a framework releases a new major version. `check.py`
takes `--agent <path>`, so it doubles as the no-API-key way to run any of them.

`langgraph_agent.py` uses `langgraph.prebuilt.create_react_agent`, which LangGraph 1.0
deprecated in favor of `langchain.agents.create_agent` — the replacement lives in the
`langchain` package, which this check deliberately does not install. The deprecation
warning in the output is expected. The `langgraph/` example uses the replacement, and
records and replays identically through it.

The run each probe records is named after its agent file, so `agentvcr runs` in the
scratch database says which framework produced which tape.
