"""A LangGraph agent — one tool, one graph, and nothing in it that knows about agentvcr.

The only concession to the proxy is `base_url`, and `agentvcr run` even supplies that:
it exports `OPENAI_BASE_URL`, which is read here (LangChain's own variable is
`OPENAI_API_BASE`, so both are honored).

It prints the final answer, which is what a replay has to reproduce.
"""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

FLIGHTS = {
    ("SFO", "JFK"): [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}],
}


def search_flights(origin: str, destination: str) -> str:
    """Find flights between two airports."""
    return str({"flights": FLIGHTS.get((origin.upper(), destination.upper()), [])})


def main() -> None:
    model = ChatOpenAI(
        model=os.environ.get("MODEL", "fake-model"),
        base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY", "sk-not-a-real-key"),
        temperature=0,
        max_retries=0,
    )
    agent = create_agent(
        model,
        tools=[search_flights],
        system_prompt="You book flights. Use the tool, then answer in one sentence.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the cheapest flight from SFO to JFK?"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
