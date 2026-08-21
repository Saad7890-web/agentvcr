"""A scripted upstream that actually reads the tool result, so the demo needs no key.

Unlike the framework check's upstream, this one is not a fixed script: it answers
differently depending on what the tool came back with. That is what makes the fork
demo honest — the second run ends differently because the *model saw a different tool
result*, not because a canned sequence moved on by one.

    no tool result yet   -> ask for search_flights
    tool result, empty   -> give up
    tool result, flights -> book the cheapest one

Every request is appended to ``--log``, so the demo can show how many calls each run
actually paid for.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOL_CALL = {
    "id": "chatcmpl-tool",
    "object": "chat.completion",
    "created": 1,
    "model": "fake-model",
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
                            "arguments": '{"origin": "SFO", "destination": "JFK"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
}


def answer(text: str, *, tokens: int = 12) -> dict:
    return {
        "id": "chatcmpl-answer",
        "object": "chat.completion",
        "created": 2,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 90, "completion_tokens": tokens, "total_tokens": 90 + tokens},
    }


GAVE_UP = answer("I could not find any flights from SFO to JFK.")


def flights_seen(messages: list[dict]) -> list[dict] | None:
    """The flight list the newest tool result carried, or ``None`` if there is none."""
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        try:
            return json.loads(message.get("content") or "{}").get("flights") or []
        except ValueError:
            return []
    return None


def make_handler(log_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's API
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            request = json.loads(body or b"{}")
            messages = request.get("messages") or []
            with log_path.open("a") as fh:
                fh.write(json.dumps({"path": self.path, "messages": messages}) + "\n")

            flights = flights_seen(messages)
            if flights is None:
                self._send(TOOL_CALL)
            elif not flights:
                self._send(GAVE_UP)
            else:
                cheapest = min(flights, key=lambda f: f.get("price", 0))
                self._send(answer(f"The cheapest is {cheapest['flight']} at ${cheapest['price']}."))

        def _send(self, payload: dict) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args: object) -> None:
            pass  # the request log is the file, not stderr

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8597)
    parser.add_argument("--log", type=Path, default=Path("upstream-calls.jsonl"))
    args = parser.parse_args()

    args.log.write_text("")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.log))
    print(f"fake upstream on http://127.0.0.1:{args.port}/v1 (log: {args.log})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
