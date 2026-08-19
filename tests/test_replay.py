"""Replay mode: positional matching, mismatch policies, streams and lineage.

The headline test is :func:`test_a_recorded_tool_loop_replays_with_no_network` — the
Phase 2 contract. It records a tool loop against a mocked upstream, then replays it
with respx asserting that *no* route exists at all, so any attempt to reach the network
raises instead of silently succeeding.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
import respx

from agentvcr.core.models import STATUS_DIVERGED
from agentvcr.core.replayer import LIVE_FROM
from conftest import UPSTREAM

CHAT = f"{UPSTREAM}/chat/completions"

TOOL_CALL = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
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
ANSWER = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 2,
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

OPENING = [
    {"role": "system", "content": "You book flights."},
    {"role": "user", "content": "Cheapest SFO to JFK?"},
]
TOOL_RESULT = json.dumps({"flights": [{"flight": "B6918", "price": 289}]})


def drive(client, run_id: str, **extra) -> list[dict]:
    """Run the scripted tool loop against ``/r/<run_id>/…`` and return the transcript.

    The same function drives the recording and the replay, so the two runs make
    identical calls in identical order — which is exactly what replay assumes.
    """
    messages = list(OPENING)
    transcript: list[dict] = []
    for _ in range(6):
        response = client.post(
            f"/r/{run_id}/openai/v1/chat/completions",
            json={"model": "llama-3.1-8b", "messages": messages, **extra},
            headers={"Authorization": "Bearer sk-test"},
        )
        response.raise_for_status()
        transcript.append(response.json())
        message = transcript[-1]["choices"][0]["message"]
        messages.append(message)
        if not message.get("tool_calls"):
            break
        for call in message["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": TOOL_RESULT})
    return transcript


def record_the_loop(proxy) -> list[dict]:
    """Record the tool loop onto tape ``TAPE`` with a mocked upstream."""
    with respx.mock:
        respx.post(CHAT).mock(
            side_effect=[httpx.Response(200, json=TOOL_CALL), httpx.Response(200, json=ANSWER)]
        )
        proxy.store.create_run(mode="record", run_id="TAPE", name="flights")
        return drive(proxy.client, "TAPE")


def set_policy(proxy, policy: str) -> None:
    """Settings are frozen, so swap the whole object the running app reads."""
    proxy.client.app.state.settings = replace(proxy.settings, mismatch_policy=policy)


def replay_run(proxy, run_id: str = "REPLAY") -> str:
    proxy.store.create_run(mode="replay", run_id=run_id, replay_of="TAPE")
    return run_id


# --------------------------------------------------------------- the Phase 2 contract


def test_a_recorded_tool_loop_replays_with_no_network(proxy) -> None:
    recorded = record_the_loop(proxy)

    # No routes are registered, so any upstream call raises instead of being served.
    with respx.mock(assert_all_called=False):
        replayed = drive(proxy.client, replay_run(proxy))

    assert replayed == recorded
    assert replayed[-1]["choices"][0]["message"]["content"] == "The cheapest is B6918 at $289."

    tape = proxy.store.list_steps("TAPE")
    replay = proxy.store.list_steps("REPLAY")
    assert [s.response for s in replay] == [s.response for s in tape]
    assert [s.fingerprint for s in replay] == [s.fingerprint for s in tape]
    assert not any(s.diverged for s in replay)

    # The replay is a run of its own, linked to the tape it came from.
    session = proxy.store.get_run("REPLAY")
    assert session.replay_of == "TAPE"
    assert proxy.store.get_run("TAPE").replay_of is None


def test_replaying_never_writes_to_the_tape(proxy) -> None:
    record_the_loop(proxy)
    before = [(s.idx, s.response, s.diverged) for s in proxy.store.list_steps("TAPE")]

    with respx.mock(assert_all_called=False):
        drive(proxy.client, replay_run(proxy))

    assert [(s.idx, s.response, s.diverged) for s in proxy.store.list_steps("TAPE")] == before


# ------------------------------------------------------------------------- divergence


def ask(proxy, run_id: str, content: str, **extra):
    return proxy.client.post(
        f"/r/{run_id}/openai/v1/chat/completions",
        json={"model": "llama-3.1-8b", "messages": [{"role": "user", "content": content}], **extra},
    )


def test_warn_serves_the_positional_answer_and_flags_the_step(proxy) -> None:
    record_the_loop(proxy)

    with respx.mock(assert_all_called=False):
        response = ask(proxy, replay_run(proxy), "a completely different question")

    assert response.status_code == 200
    assert response.json() == TOOL_CALL  # positional: step 0 of the tape, regardless
    assert response.headers["x-agentvcr-diverged"] == "true"
    (step,) = proxy.store.list_steps("REPLAY")
    assert step.diverged
    assert proxy.store.get_run("REPLAY").status == STATUS_DIVERGED
    assert not proxy.store.list_steps("TAPE")[0].diverged  # not on the tape


def test_strict_refuses_a_drifted_request(proxy) -> None:
    record_the_loop(proxy)
    set_policy(proxy, "strict")

    with respx.mock(assert_all_called=False):
        response = ask(proxy, replay_run(proxy), "a completely different question")

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "fingerprint_mismatch"
    assert response.json()["error"]["step"] == 0
    assert proxy.store.count_steps("REPLAY") == 0


def test_live_on_miss_falls_through_to_the_real_api(proxy) -> None:
    record_the_loop(proxy)
    set_policy(proxy, "live-on-miss")

    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))
        first = ask(proxy, replay_run(proxy), "a completely different question")
        # ...and it stays live from here on, even for a request the tape could answer.
        second = ask(proxy, "REPLAY", "another one")

    assert first.json() == ANSWER
    assert first.headers["x-agentvcr-recorded"] == "true"
    assert route.call_count == 2
    assert second.status_code == 200

    session = proxy.store.get_run("REPLAY")
    assert session.meta[LIVE_FROM] == 0
    assert session.status == STATUS_DIVERGED
    assert proxy.store.count_steps("REPLAY") == 2


def test_the_tape_running_out_is_a_structured_error(proxy) -> None:
    record_the_loop(proxy)

    with respx.mock(assert_all_called=False):
        replay = replay_run(proxy)
        drive(proxy.client, replay)  # consumes both steps
        response = ask(proxy, replay, "one call too many")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "tape_exhausted"
    assert error["step"] == 2
    assert error["recorded_steps"] == 2


def test_a_replay_run_with_no_tape_says_so(proxy) -> None:
    proxy.store.create_run(mode="replay", run_id="ORPHAN")

    with respx.mock(assert_all_called=False):
        response = ask(proxy, "ORPHAN", "hello")

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "no_tape"


# ----------------------------------------------------------------------------- streams


STREAM = (
    b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi "}}]}\n\n'
    b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"there"}}]}\n\n'
    b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


def record_a_stream(proxy, run_id: str = "STREAMTAPE") -> bytes:
    with respx.mock:
        respx.post(CHAT).mock(
            return_value=httpx.Response(
                200, content=STREAM, headers={"content-type": "text/event-stream"}
            )
        )
        proxy.store.create_run(mode="record", run_id=run_id)
        response = proxy.client.post(
            f"/r/{run_id}/openai/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        return response.content


def test_a_recorded_stream_replays_byte_identically(proxy) -> None:
    """With a raw chunk tape, replay hands back the provider's own bytes."""
    recorded = record_a_stream(proxy)
    proxy.store.create_run(mode="replay", run_id="SREPLAY", replay_of="STREAMTAPE")

    with respx.mock(assert_all_called=False):
        response = ask(proxy, "SREPLAY", "hi", stream=True)

    assert response.content == recorded == STREAM
    assert response.headers["content-type"].startswith("text/event-stream")


def test_a_stream_is_synthesized_when_no_chunks_were_kept(tmp_path) -> None:
    """Without the raw tape the stream is rebuilt from the accumulated message."""
    from fastapi.testclient import TestClient

    from agentvcr.config import Settings
    from agentvcr.core.store import Store
    from agentvcr.server.app import create_app

    settings = Settings(
        db_path=tmp_path / "db.sqlite",
        upstreams={"openai": UPSTREAM, "anthropic": "https://anthropic.test"},
        record_chunks=False,
    )
    store = Store.open(settings.db_path)
    with TestClient(create_app(settings, store=store)) as client:
        harness = type("H", (), {"client": client, "store": store, "settings": settings})
        recorded = record_a_stream(harness)
        assert store.list_steps("STREAMTAPE")[0].response_chunks is None
        store.create_run(mode="replay", run_id="SREPLAY", replay_of="STREAMTAPE")

        with respx.mock(assert_all_called=False):
            response = client.post(
                "/r/SREPLAY/openai/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

    assert response.content != recorded  # not byte-identical...
    text = "".join(
        json.loads(line[len("data: ") :])["choices"][0]["delta"].get("content", "")
        for line in response.text.strip().split("\n\n")
        if line.startswith("data: ") and "[DONE]" not in line
    )
    assert text == "hi there"  # ...but the same message
    store.close()


def test_a_recorded_stream_replays_to_a_non_streaming_client(proxy) -> None:
    """One tape answers both shapes: the accumulated message is always stored."""
    record_a_stream(proxy)
    proxy.store.create_run(mode="replay", run_id="SREPLAY", replay_of="STREAMTAPE")

    with respx.mock(assert_all_called=False):
        response = ask(proxy, "SREPLAY", "hi")

    assert response.headers["content-type"] == "application/json"
    assert response.json()["choices"][0]["message"]["content"] == "hi there"


# ------------------------------------------------------------------------ error steps


def test_a_recorded_error_replays_as_the_error_it_was(proxy) -> None:
    """A 429 on the tape comes back as a 429, so the client's retry path replays too."""
    with respx.mock:
        respx.post(CHAT).mock(
            side_effect=[
                httpx.Response(429, json={"error": {"message": "slow down"}}),
                httpx.Response(200, json=ANSWER),
            ]
        )
        proxy.store.create_run(mode="record", run_id="TAPE")
        for _ in range(2):
            ask(proxy, "TAPE", "hi")

    with respx.mock(assert_all_called=False):
        replay = replay_run(proxy)
        first = ask(proxy, replay, "hi")
        second = ask(proxy, replay, "hi")

    assert first.status_code == 429
    assert first.json()["error"]["message"] == "slow down"
    assert second.status_code == 200
    assert second.json() == ANSWER


def test_replay_mode_never_reaches_upstream_for_unrecorded_paths(proxy) -> None:
    """Replay is offline by construction — even for calls no tape can answer."""
    with respx.mock(assert_all_called=False):
        response = proxy.client.get("/openai/v1/models", headers={"X-AgentVCR-Mode": "replay"})

    assert response.status_code == 501
    assert response.json()["error"]["type"] == "not_replayable"


@pytest.mark.parametrize("mode", ["replay"])
def test_a_recorded_run_can_be_replayed_by_pointing_at_it_directly(proxy, mode) -> None:
    """Point at the *tape* with the mode header and a replay run is created for you."""
    record_the_loop(proxy)

    with respx.mock(assert_all_called=False):
        response = proxy.client.post(
            "/r/TAPE/openai/v1/chat/completions",
            json={"model": "llama-3.1-8b", "messages": OPENING},
            headers={"X-AgentVCR-Mode": mode},
        )

    assert response.json() == TOOL_CALL
    session_id = response.headers["x-agentvcr-run"]
    assert session_id != "TAPE"
    assert proxy.store.get_run(session_id).replay_of == "TAPE"
    assert proxy.store.count_steps("TAPE") == 2  # tape untouched
