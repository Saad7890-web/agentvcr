"""The Phase 1 acceptance check: a tool loop records end to end and `show` renders it.

The loop is written against raw HTTP rather than the OpenAI SDK so CI needs no extra
dependency — the bytes on the wire are the same ones `examples/plain-loop/agent.py`
sends.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from agentvcr.cli import app
from conftest import UPSTREAM

runner = CliRunner()
API_KEY = "sk-roundtrip-secret"

TOOL_STEP = {
    "id": "chatcmpl-1",
    "model": "llama-3.1-8b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"origin":"SFO","destination":"JFK"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"total_tokens": 412},
}
FINAL_STEP = {
    "id": "chatcmpl-2",
    "model": "llama-3.1-8b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The cheapest is B6918 at $289."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"total_tokens": 503},
}


@respx.mock
def test_a_tool_loop_records_and_show_renders_it(proxy) -> None:
    respx.post(f"{UPSTREAM}/chat/completions").mock(
        side_effect=[httpx.Response(200, json=TOOL_STEP), httpx.Response(200, json=FINAL_STEP)]
    )
    proxy.store.create_run(
        mode="record", run_id="LOOPRUN", name="flights", command=["python", "agent.py"]
    )

    messages = [
        {"role": "system", "content": "You book flights."},
        {"role": "user", "content": "Cheapest SFO to JFK?"},
    ]
    for _ in range(6):
        reply = proxy.client.post(
            "/r/LOOPRUN/openai/v1/chat/completions",
            json={"model": "llama-3.1-8b", "messages": messages},
            headers={"Authorization": f"Bearer {API_KEY}"},
        ).json()
        message = reply["choices"][0]["message"]
        messages.append(message)
        if not message.get("tool_calls"):
            break
        for call in message["tool_calls"]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"flights": [{"flight": "B6918", "price": 289}]}),
                }
            )

    tool_step, final_step = proxy.store.list_steps("LOOPRUN")
    assert tool_step.response == TOOL_STEP
    assert final_step.response == FINAL_STEP
    # the tool result the agent computed locally is visible in the next request (§2)
    tool_message = final_step.request["body"]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["flights"][0]["flight"] == "B6918"

    # …and it has been materialized into a tool step of its own, between the two
    # LLM steps, with no help at all from the agent.
    (tool_call,) = proxy.store.list_tool_calls("LOOPRUN")
    assert tool_call.after_step_idx == 0
    assert tool_call.tool_name == "search_flights"
    assert tool_call.args == {"origin": "SFO", "destination": "JFK"}
    assert tool_call.result == {"flights": [{"flight": "B6918", "price": 289}]}

    result = runner.invoke(app, ["show", "LOOPRUN", "--db", str(proxy.settings.db_path)])
    assert result.exit_code == 0
    assert "search_flights" in result.stdout
    assert "The cheapest is B6918" in result.stdout
    assert "python agent.py" in result.stdout
    assert API_KEY not in result.stdout


def test_the_example_agent_stays_valid_python() -> None:
    source = Path(__file__).resolve().parents[1] / "examples" / "plain-loop" / "agent.py"
    ast.parse(source.read_text())
