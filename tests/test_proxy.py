from __future__ import annotations

import json

import httpx
import pytest
import respx

from agentvcr.core.recorder import REDACTION_PLACEHOLDER
from conftest import UPSTREAM

API_KEY = "sk-super-secret-key-do-not-store"
CHAT = f"{UPSTREAM}/chat/completions"

ANSWER = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "llama-3.1-8b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
}


def ask(client, messages, **extra):
    return client.post(
        "/openai/v1/chat/completions",
        json={"model": "llama-3.1-8b", "messages": messages, **extra},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )


@respx.mock
def test_records_a_non_streaming_call(proxy) -> None:
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))

    response = ask(proxy.client, [{"role": "user", "content": "hello"}])

    assert response.status_code == 200
    assert response.json() == ANSWER
    assert response.headers["x-agentvcr-recorded"] == "true"
    assert response.headers["x-agentvcr-step"] == "0"
    assert route.called

    (run,) = proxy.store.list_runs()
    (step,) = proxy.store.list_steps(run.id)
    assert step.idx == 0
    assert step.model == "llama-3.1-8b"
    assert step.status_code == 200
    assert step.usage == ANSWER["usage"]
    assert step.latency_ms is not None
    assert step.fingerprint
    assert step.request["body"]["messages"][0]["content"] == "hello"
    assert step.response == ANSWER
    assert run.provider == "openai"
    assert run.upstream_url == UPSTREAM


@respx.mock
def test_credentials_go_upstream_but_never_to_disk(proxy) -> None:
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))

    ask(proxy.client, [{"role": "user", "content": "hello"}])

    # forwarded intact...
    assert route.calls.last.request.headers["authorization"] == f"Bearer {API_KEY}"

    # ...and absent from every column of every table, and from the file itself.
    (run,) = proxy.store.list_runs()
    (step,) = proxy.store.list_steps(run.id)
    assert step.request["headers"]["authorization"] == REDACTION_PLACEHOLDER

    tables = [
        r[0] for r in proxy.store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]
    dumped = ""
    for table in tables:
        for row in proxy.store.conn.execute(f"SELECT * FROM {table}"):
            dumped += "".join(str(v) for v in tuple(row))
    assert API_KEY not in dumped

    proxy.store.conn.execute("PRAGMA wal_checkpoint(FULL)")
    assert API_KEY.encode() not in proxy.settings.db_path.read_bytes()


@respx.mock
def test_records_a_streaming_call_and_accumulates_it(proxy) -> None:
    events = [
        {
            "id": "chatcmpl-1",
            "model": "llama-3.1-8b",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        },
        {"choices": [{"index": 0, "delta": {"content": "hi "}}]},
        {"choices": [{"index": 0, "delta": {"content": "there"}}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        },
    ]
    tape = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"
    respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, content=tape, headers={"content-type": "text/event-stream"}
        )
    )

    response = ask(proxy.client, [{"role": "user", "content": "hello"}], stream=True)

    assert response.status_code == 200
    assert response.content == tape  # the client sees the upstream bytes, unaltered
    assert response.headers["x-agentvcr-recorded"] == "true"

    (run,) = proxy.store.list_runs()
    (step,) = proxy.store.list_steps(run.id)
    assert step.response["choices"][0]["message"]["content"] == "hi there"
    assert step.response["choices"][0]["finish_reason"] == "stop"
    assert step.usage == {"total_tokens": 10}
    assert step.response_chunks == tape  # raw tape kept for byte-exact replay


@respx.mock
def test_streaming_chunks_are_optional(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from agentvcr.config import Settings
    from agentvcr.server.app import create_app

    respx.post(CHAT).mock(
        return_value=httpx.Response(200, content=b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
    )
    settings = Settings(
        db_path=tmp_path / "a.db", upstreams={"openai": UPSTREAM}, record_chunks=False
    )
    from agentvcr.core.store import Store

    store = Store.open(settings.db_path)
    with TestClient(create_app(settings, store=store)) as client:
        ask(client, [{"role": "user", "content": "hi"}], stream=True)
    (run,) = store.list_runs()
    assert store.list_steps(run.id)[0].response_chunks is None
    store.close()


@respx.mock
def test_upstream_errors_are_recorded_so_retries_replay(proxy) -> None:
    """An SDK that retries a 429 produces two steps; the tape reproduces the sequence."""
    respx.post(CHAT).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limit", "type": "rate_limit"}}),
            httpx.Response(200, json=ANSWER),
        ]
    )
    messages = [{"role": "user", "content": "hello"}]

    first = ask(proxy.client, messages)
    second = ask(proxy.client, messages)  # the retry, byte-identical

    assert first.status_code == 429
    assert second.status_code == 200

    (run,) = proxy.store.list_runs()
    failed, ok = proxy.store.list_steps(run.id)
    assert (failed.status_code, failed.ok) == (429, False)
    assert (ok.status_code, ok.ok) == (200, True)
    assert failed.fingerprint == ok.fingerprint  # same call, recorded twice


@respx.mock
def test_unrecorded_paths_are_proxied_verbatim(proxy) -> None:
    route = respx.get(f"{UPSTREAM}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama-3.1-8b"}]})
    )

    response = proxy.client.get("/openai/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "llama-3.1-8b"
    assert response.headers["x-agentvcr-recorded"] == "false"
    assert route.calls.last.request.headers["authorization"] == f"Bearer {API_KEY}"
    assert proxy.store.list_runs() == []


@respx.mock
def test_an_unrecorded_path_under_a_recording_mount_is_proxied(proxy) -> None:
    """Only a provider's ``recorded_paths`` become steps; the rest of its mount is a
    plain proxy, which is what makes token counting and model listing keep working."""
    route = respx.post("https://anthropic.test/v1/messages/count_tokens").mock(
        return_value=httpx.Response(200, json={"input_tokens": 12})
    )

    response = proxy.client.post(
        "/anthropic/v1/messages/count_tokens", json={"model": "claude", "messages": []}
    )

    assert response.status_code == 200
    assert route.called
    assert response.headers["x-agentvcr-recorded"] == "false"
    assert proxy.store.list_runs() == []


@respx.mock
def test_passthrough_mode_records_nothing(proxy) -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))

    response = proxy.client.post(
        "/openai/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-AgentVCR-Mode": "passthrough"},
    )

    assert response.status_code == 200
    assert response.headers["x-agentvcr-recorded"] == "false"
    assert proxy.store.list_runs() == []


@pytest.mark.parametrize("mode,phase", [("fork", "4")])
def test_unimplemented_modes_answer_with_a_structured_error(proxy, mode, phase) -> None:
    response = proxy.client.post(
        "/openai/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"X-AgentVCR-Mode": mode},
    )
    assert response.status_code == 501
    body = response.json()["error"]
    assert body["type"] == "mode_not_implemented"
    assert f"Phase {phase}" in body["message"]


@respx.mock
def test_an_unknown_mode_is_an_error_not_a_silent_passthrough(proxy) -> None:
    """A typo in the mode header must not look like recording while recording nothing."""
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))

    response = proxy.client.post(
        "/openai/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"X-AgentVCR-Mode": "recrod"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unknown_mode"
    assert not route.called


@respx.mock
def test_unreachable_upstream_is_reported_not_recorded(proxy) -> None:
    respx.post(CHAT).mock(side_effect=httpx.ConnectError("nope"))

    response = ask(proxy.client, [{"role": "user", "content": "hi"}])

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unreachable"
    (run,) = proxy.store.list_runs()
    assert proxy.store.list_steps(run.id) == []


@respx.mock
def test_the_run_travels_in_the_base_url(proxy) -> None:
    """`agentvcr run` points the SDK at /r/<id>/openai/v1, so no header support needed."""
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))
    proxy.store.create_run(mode="record", run_id="RUNFROMCLI", command=["python", "agent.py"])

    response = proxy.client.post(
        "/r/RUNFROMCLI/openai/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.headers["x-agentvcr-run"] == "RUNFROMCLI"
    (run,) = proxy.store.list_runs()
    assert run.id == "RUNFROMCLI"
    assert run.command == ["python", "agent.py"]
    assert proxy.store.count_steps("RUNFROMCLI") == 1


@respx.mock
def test_the_run_header_also_assigns(proxy) -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))

    for content in ("one", "unrelated two"):
        proxy.client.post(
            "/openai/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": content}]},
            headers={"X-AgentVCR-Run": "EXPLICIT"},
        )

    (run,) = proxy.store.list_runs()
    assert run.id == "EXPLICIT"
    assert proxy.store.count_steps("EXPLICIT") == 2  # despite unrelated message lists


@respx.mock
def test_a_growing_conversation_is_grouped_into_one_run(proxy) -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=ANSWER))
    conversation = [{"role": "user", "content": "hello"}]

    ask(proxy.client, conversation)
    conversation += [
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "and now?"},
    ]
    ask(proxy.client, conversation)
    ask(proxy.client, [{"role": "user", "content": "a totally different agent"}])

    grouped, separate = sorted(proxy.store.list_runs(), key=lambda r: proxy.store.count_steps(r.id))
    assert proxy.store.count_steps(separate.id) == 2
    assert proxy.store.count_steps(grouped.id) == 1


@respx.mock
def test_a_runs_own_mode_beats_the_server_default(proxy) -> None:
    """`agentvcr run --mode replay` stores the mode on the run; the proxy honors it.

    The server default here is `record`, and nothing is mocked: reaching the upstream
    at all would fail the test.
    """
    proxy.store.create_run(mode="record", run_id="EMPTYTAPE")
    proxy.store.create_run(mode="replay", run_id="REPLAYRUN", replay_of="EMPTYTAPE")

    response = proxy.client.post(
        "/r/REPLAYRUN/openai/v1/chat/completions",
        json={"model": "m", "messages": []},
    )

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "tape_exhausted"
