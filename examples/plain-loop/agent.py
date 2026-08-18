"""A scripted tool loop on the OpenAI SDK — the Phase 1 record/replay contract.

Nothing here knows about agentvcr. The only line that changes is ``base_url``; under
``agentvcr run`` even that is already set for you via ``OPENAI_BASE_URL``.
"""

import json
import os

from openai import OpenAI

PROXY = "http://127.0.0.1:8484/openai/v1"  # was https://api.openai.com/v1
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", PROXY))
MODEL = os.environ.get("MODEL", "llama-3.1-8b-instant")

FLIGHTS = {
    ("SFO", "JFK"): [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}],
    ("SFO", "AUS"): [{"flight": "WN1104", "price": 178}],
}


def search_flights(origin: str, destination: str) -> dict:
    return {"flights": FLIGHTS.get((origin.upper(), destination.upper()), [])}


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
        break
    for call in message.tool_calls:
        result = search_flights(**json.loads(call.function.arguments))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
