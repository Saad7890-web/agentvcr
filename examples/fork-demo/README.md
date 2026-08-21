# fork-demo

The story agentvcr exists for, in one command and with no API key:

```bash
python examples/fork-demo/demo.py
```

An agent asks its flight tool for SFO→JFK. The tool's data source has gone stale and
returns nothing, so the agent gives up. The tool is not the thing being debugged — the
question is *what would the agent have done if the search had worked?* Re-running it
means re-running the whole thing, paying for every LLM call again, and hoping the model
lands the same way.

Instead:

```bash
agentvcr fork <run> --at 0 --edit-tool-result search_flights=flights.json
agentvcr run --mode fork --run <fork> -- python agent.py
agentvcr diff <run> <fork>
```

Step 0 replays off the tape for free. From step 1 the agent runs for real — but the
proxy rewrites the tool result on its way upstream, so the model sees the flights the
search should have found, and books one:

```
  step 0  tool search_flights returned different results
    - {"flights": []}
    + {"flights": [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}]}
  step 1  response text differs
    - I could not find any flights from SFO to JFK.
    + The cheapest is B6918 at $289.

runs diverge at step 0 (tool search_flights returned different results)
```

## What is worth noticing

- **`agent.py` is byte-identical between the two runs.** Its `search_flights` still
  returns nothing, both times. The edit lives at the proxy, so the agent's own code,
  its tools and its prompt never had to be touched to ask the question.
- **The prefix is free.** The demo asserts it: a two-step branch costs one upstream
  call, not two.
- **The upstream reads the tool result.** `fake_upstream.py` decides what to answer
  from what the search came back with, so the second ending is *caused* by the edit
  rather than by a script advancing. A canned sequence would have proved nothing.

## Files

| file | what it is |
| --- | --- |
| `agent.py` | a ~50-line OpenAI-SDK tool loop; the only agentvcr-shaped line is `base_url` |
| `fake_upstream.py` | a scripted model that answers according to the tool result it is shown |
| `demo.py` | drives record → fork → re-run → diff and checks the outcome |

`demo.py` exits non-zero if any of it stops being true, so it doubles as an end-to-end
test of fork mode against the real CLI and a real HTTP server. `tests/test_fork.py` is
the fast, in-process version of the same contract.
