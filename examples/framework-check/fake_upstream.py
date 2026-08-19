"""A scripted OpenAI-compatible upstream, so the framework check needs no API key.

It answers `POST /v1/chat/completions` with a tool call the first time and a final
answer once it sees a tool result, which is the shape every tool-calling agent
framework drives. It also logs each request it receives to `--log`, so the check can
prove how many calls actually reached the network during a replay: zero.
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
                "content": "",
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
FINAL = {
    "id": "chatcmpl-final",
    "object": "chat.completion",
    "created": 2,
    "model": "fake-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The cheapest is B6918 at $289."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 90, "completion_tokens": 12, "total_tokens": 102},
}


def answer(text: str) -> dict:
    """A plain final answer — used by the parallel probe, where each branch differs."""
    return {
        **FINAL,
        "id": f"chatcmpl-{text.split()[0].lower()}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"The cheapest is {text}."},
                "finish_reason": "stop",
            }
        ],
    }


def make_handler(log_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's API
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            request = json.loads(body or b"{}")
            entry = {"path": self.path, "messages": request.get("messages")}
            with log_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")

            messages = request.get("messages") or []
            answered = any(m.get("role") == "tool" for m in messages)
            asked = " ".join(str(m.get("content") or "") for m in messages)
            if "AUS" in asked and not answered:
                self._send(answer("WN1104 at $178"))
            elif answered:
                self._send(FINAL)
            else:
                self._send(TOOL_CALL)

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
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--log", type=Path, default=Path("upstream-calls.jsonl"))
    args = parser.parse_args()

    args.log.write_text("")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.log))
    print(f"fake upstream on http://127.0.0.1:{args.port}/v1 (log: {args.log})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
