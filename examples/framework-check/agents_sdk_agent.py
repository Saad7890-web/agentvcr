"""An OpenAI Agents SDK agent — the second framework the reality check probes.

Like `langgraph_agent.py`, the point is that it is *ordinary*: one tool, one agent, no
awareness of agentvcr. Three lines are about the SDK rather than the proxy:

* `set_default_openai_client` — the SDK builds its own `AsyncOpenAI`, so pointing it at
  `OPENAI_BASE_URL` means handing it a client rather than setting an env var.
* `set_default_openai_api("chat_completions")` — the SDK defaults to the Responses API
  (`/v1/responses`), which agentvcr does not speak yet (DESIGN.md §12, post-MVP item 3).
* `set_tracing_disabled(True)` — tracing uploads to api.openai.com, which would both
  need a real key and make "the replay reached the network zero times" untrue for a
  reason that has nothing to do with replay.

It prints the final answer, which the check compares between the recording and the
replay.
"""

from __future__ import annotations

import asyncio
import os

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

FLIGHTS = {
    ("SFO", "JFK"): [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}],
}


@function_tool
def search_flights(origin: str, destination: str) -> str:
    """Find flights between two airports."""
    return str({"flights": FLIGHTS.get((origin.upper(), destination.upper()), [])})


def main() -> None:
    set_default_openai_client(
        AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY", "sk-not-a-real-key"),
            max_retries=0,
        ),
        use_for_tracing=False,
    )
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)

    agent = Agent(
        name="flights",
        instructions="You book flights. Use the tool, then answer in one sentence.",
        tools=[search_flights],
        model=os.environ.get("MODEL", "fake-model"),
    )
    result = asyncio.run(
        Runner.run(agent, "What is the cheapest flight from SFO to JFK?", max_turns=6)
    )
    print(result.final_output)


if __name__ == "__main__":
    main()
