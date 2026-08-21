"""The agent that gets forked. A flight-booking loop with a stale data source.

Nothing here knows about agentvcr, and nothing here changes between the failing run and
the forked one — that is the point. ``search_flights`` returns an empty list both times;
the fork replaces its *result* at the proxy, and the model reacts to that instead.

Exits 1 when the agent gives up, 0 when it books, so the demo can assert on the outcome
rather than on the prose.
"""

import json
import os
import sys

from openai import OpenAI

PROXY = "http://127.0.0.1:8484/openai/v1"  # was https://api.openai.com/v1
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", PROXY))
MODEL = os.environ.get("MODEL", "llama-3.1-8b-instant")


def search_flights(origin: str, destination: str) -> dict:
    """The stale source: it has stopped returning anything for this route."""
    return {"flights": []}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Find flights between two airports.",
            "parameters": {
                "type": "object",
                "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
                "required": ["origin", "destination"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": "You book flights. Use the tool, then answer in one sentence."},
    {"role": "user", "content": "What is the cheapest flight from SFO to JFK?"},
]

for _ in range(6):
    reply = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    message = reply.choices[0].message
    messages.append(message.model_dump(exclude_none=True))
    if not message.tool_calls:
        print(message.content)
        sys.exit(1 if "could not" in (message.content or "") else 0)
    for call in message.tool_calls:
        result = search_flights(**json.loads(call.function.arguments))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
