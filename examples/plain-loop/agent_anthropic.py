"""The same agent as ``agent.py``, in the Anthropic wire format.

Two files, one point: the *only* thing that differs between them is the SDK and the
message shape it speaks. The proxy records, replays and diffs both through the same
recorder, the same positional replayer and the same tool extractor — the provider
layer absorbs the difference (DESIGN.md §9).

Nothing here knows about agentvcr. The only line that changes is ``base_url``; under
``agentvcr run`` even that is already set for you via ``ANTHROPIC_BASE_URL``.
"""

import json
import os

from anthropic import Anthropic

PROXY = "http://127.0.0.1:8484/anthropic"  # was https://api.anthropic.com
client = Anthropic(base_url=os.environ.get("ANTHROPIC_BASE_URL", PROXY))
MODEL = os.environ.get("MODEL", "claude-sonnet-4-5")

FLIGHTS = {
    ("SFO", "JFK"): [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}],
    ("SFO", "AUS"): [{"flight": "WN1104", "price": 178}],
}


def search_flights(origin: str, destination: str) -> dict:
    return {"flights": FLIGHTS.get((origin.upper(), destination.upper()), [])}


TOOLS = [
    {
        "name": "search_flights",
        "description": "Find flights between two airports.",
        "input_schema": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["origin", "destination"],
        },
    }
]

messages = [{"role": "user", "content": "What is the cheapest flight from SFO to JFK?"}]

for _ in range(6):
    reply = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You book flights. Use the tool, then answer in one sentence.",
        messages=messages,
        tools=TOOLS,
    )
    messages.append({"role": "assistant", "content": [b.model_dump() for b in reply.content]})
    calls = [block for block in reply.content if block.type == "tool_use"]
    if not calls:
        print("".join(block.text for block in reply.content if block.type == "text"))
        break
    # The result rides back inside the *next* request, which is how the proxy sees it.
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(search_flights(**call.input)),
                }
                for call in calls
            ],
        }
    )
