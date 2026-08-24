"""A CrewAI crew — one agent, one tool, one task, and nothing in it that knows agentvcr.

The only concession to the proxy is the `base_url` handed to `LLM`, which is what
`agentvcr run` exports as `OPENAI_BASE_URL`. Two other lines are about CrewAI rather
than about the proxy — both of them uploads that would otherwise happen mid-replay, and
the README says what they are.

It prints the crew's final answer, which is what a replay has to reproduce.
"""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")  # read at import time

from crewai import LLM, Agent, Crew, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

FLIGHTS = {
    ("SFO", "JFK"): [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}],
}


@tool("search_flights")
def search_flights(origin: str, destination: str) -> str:
    """Find flights between two airports."""
    return str({"flights": FLIGHTS.get((origin.upper(), destination.upper()), [])})


def main() -> None:
    llm = LLM(
        model=f"openai/{os.environ.get('MODEL', 'fake-model')}",
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "sk-not-a-real-key"),
        temperature=0,
    )
    agent = Agent(
        role="flight finder",
        goal="Find the cheapest flight on a route",
        backstory="You look flights up with the tool, then answer in one sentence.",
        tools=[search_flights],
        llm=llm,
    )
    task = Task(
        description="What is the cheapest flight from SFO to JFK?",
        expected_output="One sentence naming the flight and its price.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], tracing=False)  # tracing uploads to crewai.com
    print(crew.kickoff().raw)


if __name__ == "__main__":
    main()
