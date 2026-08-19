"""A LangGraph ReAct agent — used to falsify agentvcr's assumptions, not to ship.

The point of this file is that it is *ordinary*: `create_react_agent`, one tool, no
awareness of agentvcr. The only concession to the proxy is reading `OPENAI_BASE_URL`,
which is what `agentvcr run` exports (LangChain's own env var is `OPENAI_API_BASE`, so
both are honored here — see README.md).

It prints the final answer, which the check compares between the recording and the
replay.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

FLIGHTS = {
    ("SFO", "JFK"): [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}],
}


def search_flights(origin: str, destination: str) -> str:
    """Find flights between two airports."""
    return str({"flights": FLIGHTS.get((origin.upper(), destination.upper()), [])})


def main() -> None:
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    model = ChatOpenAI(
        model=os.environ.get("MODEL", "fake-model"),
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "sk-not-a-real-key"),
        temperature=0,
        max_retries=0,
    )
    agent = create_react_agent(model, [search_flights])
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the cheapest flight from SFO to JFK?"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
