"""An OpenAI Agents SDK agent — one tool, one agent, no awareness of agentvcr.

Three lines here are about the SDK rather than about the proxy, and the README says why:
the SDK builds its own client (so the base URL is handed to it rather than set in the
environment), it speaks the Responses API unless told otherwise, and its tracing uploads
to api.openai.com.

It prints the final answer, which is what a replay has to reproduce.
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
    set_default_openai_api("chat_completions")  # agentvcr does not speak /v1/responses yet
    set_tracing_disabled(True)  # tracing would upload to api.openai.com during a replay

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
