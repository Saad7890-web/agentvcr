"""A LangGraph graph with two branches that call the model *concurrently*.

This is the probe for DESIGN.md §5's known limitation: positional replay assumes the
*N*th call of a replay is the *N*th call of the recording, and parallel fan-out makes
call order a race. Whether that actually breaks depends on how a framework schedules
its branches, which is a question to answer with evidence rather than assumption —
`check.py --agent parallel_agent.py` does exactly that.

Both branches ask a different question, so a swapped order is detectable: the answers
come back attached to the wrong branch.
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    answers: Annotated[list[str], operator.add]


def model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("MODEL", "fake-model"),
        base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY", "sk-not-a-real-key"),
        temperature=0,
        max_retries=0,
    )


def branch(question: str):
    def node(_state: State) -> State:
        reply = model().invoke([{"role": "user", "content": question}])
        return {"answers": [f"{question} -> {reply.content}"]}

    return node


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("left", branch("Cheapest flight from SFO to JFK?"))
    graph.add_node("right", branch("Cheapest flight from SFO to AUS?"))
    # Both edges leave START, so LangGraph runs the two nodes in one superstep.
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)

    result = graph.compile().invoke({"answers": []})
    for answer in sorted(result["answers"]):
        print(answer)


if __name__ == "__main__":
    main()
