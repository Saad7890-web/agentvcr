"""The control API the web UI runs on (PLAN.md phase 5).

Two things are being checked here. The first is that a click does what the equivalent
command does — fork, re-run, diff — because the UI is a second front end onto one
engine, not a second implementation of it.

The second is the guard, and it is the reason this file leads with it: the API is
reachable from *any* page the user's browser happens to open, it hands out whole
recorded prompts, and ``POST rerun`` starts a process. The tests below pin both halves
of the answer — a cross-origin page is refused, and so is a host we were never bound
to, which is the shape DNS rebinding arrives in.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from agentvcr.core.store import Store
from test_differ import tool_loop

FLIGHTS = {"flights": [{"flight": "B6918", "price": 289}]}
NOTHING = {"flights": []}

#: An "agent" that reports the environment `agentvcr run` would have given it. Enough
#: to prove a re-run was pointed at the right run without needing a real model.
ECHO_AGENT = [
    sys.executable,
    "-c",
    "import os; print(os.environ['AGENTVCR_MODE'], os.environ['OPENAI_BASE_URL'])",
]


def recorded(store: Store, run_id: str = "TAPE", *, result: dict = NOTHING, **fields: Any) -> str:
    """A two-step run — ask, tool call, result, answer — plus any run fields wanted."""
    tool_loop(store, run_id, result=result, final="No flights found.")
    if fields:
        store.update_run(run_id, **fields)
    return run_id


def wait_for_job(client: Any, job_id: str, *, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()["job"]
        if not job["running"]:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


# -------------------------------------------------------------------------------- guard


def test_a_cross_origin_page_is_refused(proxy) -> None:
    """Any site the user visits can call localhost; none of them may read a tape."""
    allowed = proxy.client.get("/api/runs", headers={"Origin": "http://127.0.0.1:8484"})
    refused = proxy.client.get("/api/runs", headers={"Origin": "https://evil.example"})
    assert allowed.status_code == 200
    assert refused.status_code == 403
    assert "cross-origin" in refused.json()["detail"]


def test_a_host_we_were_never_bound_to_is_refused(proxy) -> None:
    """The Origin check alone loses to DNS rebinding, where the attacker *is* us."""
    rebound = proxy.client.get("/api/runs", headers={"Host": "rebound.evil.example"})
    assert rebound.status_code == 403
    assert "rebinding" in rebound.json()["detail"]


def test_the_guard_covers_reads_as_well_as_writes(proxy) -> None:
    """A read is not harmless here: the tapes hold whole prompts."""
    for path in ("/api/status", "/api/runs", "/api/jobs", "/api/diff?a=x&b=y"):
        assert proxy.client.get(path, headers={"Origin": "https://evil.example"}).status_code == 403


# --------------------------------------------------------------------------------- runs


def test_the_run_list_summarizes_without_reading_the_bodies(proxy) -> None:
    recorded(proxy.store, "TAPE", name="booking")
    rows = proxy.client.get("/api/runs").json()["runs"]
    assert [row["id"] for row in rows] == ["TAPE"]
    assert rows[0]["stats"] == {
        "steps": 2,
        "tool_calls": 1,
        "tokens": None,
        "model": "m",
        "diverged": False,
        "errors": 0,
    }
    assert rows[0]["label"] == "booking"
    # Nothing in a listing row carries a request body — that is a whole conversation
    # per step, and the list view has no use for it (DESIGN.md §8).
    assert "request" not in json.dumps(rows[0])


def test_a_run_carries_its_timeline_and_lineage(proxy) -> None:
    recorded(proxy.store, "TAPE")
    fork = proxy.client.post(
        "/api/runs/TAPE/fork",
        json={
            "at": 0,
            "edits": [{"kind": "tool_result", "target": "search_flights", "value": FLIGHTS}],
        },
    ).json()["fork"]

    tape = proxy.client.get("/api/runs/TAPE").json()
    assert [step["idx"] for step in tape["steps"]] == [0, 1]
    assert tape["steps"][0]["tool_calls"][0]["tool_name"] == "search_flights"
    assert tape["steps"][1]["text"] == "No flights found."
    assert [call["tool_name"] for call in tape["tool_calls"]] == ["search_flights"]
    # The fork hangs off the run it branched from, which is how the list nests it.
    assert [child["id"] for child in tape["children"]] == [fork["id"]]

    branch = proxy.client.get(f"/api/runs/{fork['id']}").json()
    assert branch["parent"]["id"] == "TAPE"
    assert branch["edits"][0]["description"] == "tool result of search_flights at step 0"


def test_an_unknown_run_is_a_404_that_says_so(proxy) -> None:
    response = proxy.client.get("/api/runs/NOPE")
    assert response.status_code == 404
    assert response.json()["detail"] == "no such run: NOPE"


# -------------------------------------------------------------------------------- steps


def test_a_step_opens_with_its_conversation_and_its_response(proxy) -> None:
    recorded(proxy.store, "TAPE")
    step = proxy.client.get("/api/runs/TAPE/steps/1").json()["step"]

    assert [message["role"] for message in step["conversation"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert step["request_settings"] == {"model": "m"}
    assert step["full_text"] == "No flights found."
    assert step["request"]["body"]["model"] == "m"
    assert proxy.client.get("/api/runs/TAPE/steps/9").status_code == 404


def test_a_step_of_the_tool_call_lists_what_ran_after_it(proxy) -> None:
    recorded(proxy.store, "TAPE")
    payload = proxy.client.get("/api/runs/TAPE/steps/0").json()
    assert [call["tool_name"] for call in payload["tool_calls"]] == ["search_flights"]
    assert payload["tool_calls"][0]["result"] == NOTHING


# --------------------------------------------------------------------------------- fork


def test_an_edit_from_the_ui_creates_the_same_fork_the_cli_would(proxy) -> None:
    recorded(proxy.store, "TAPE", command=["python", "agent.py"])
    created = proxy.client.post(
        "/api/runs/TAPE/fork",
        json={
            "at": 0,
            "name": "with-flights",
            "edits": [{"kind": "tool_result", "target": "search_flights", "value": FLIGHTS}],
        },
    )
    assert created.status_code == 201
    body = created.json()
    fork_id = body["fork"]["id"]

    assert body["fork"]["parent_run_id"] == "TAPE"
    assert body["fork"]["fork_step"] == 0
    assert body["edits"][0]["patch"]["result"] == FLIGHTS
    assert body["command"] == f"agentvcr run --mode fork --run {fork_id} -- python agent.py"
    # A fork is created empty: the prefix is replayed onto it, never copied into it.
    assert proxy.store.count_steps(fork_id) == 0


def test_a_fork_the_tape_cannot_support_is_refused_in_prose(proxy) -> None:
    recorded(proxy.store, "TAPE")
    refused = proxy.client.post(
        "/api/runs/TAPE/fork",
        json={"at": 0, "edits": [{"kind": "tool_result", "target": "book_flight", "value": {}}]},
    )
    assert refused.status_code == 400
    assert "calls no tool named 'book_flight'" in refused.json()["detail"]
    assert "it calls: search_flights" in refused.json()["detail"]


# -------------------------------------------------------------------------------- rerun


def test_rerunning_a_fork_starts_the_stored_command_pointed_at_the_fork(proxy) -> None:
    """The Re-run button, end to end — and the command comes off the tape, never the
    request. What the child prints is the environment it was actually handed."""
    recorded(proxy.store, "TAPE", command=ECHO_AGENT)
    fork_id = proxy.client.post(
        "/api/runs/TAPE/fork",
        json={
            "at": 0,
            "edits": [{"kind": "tool_result", "target": "search_flights", "value": FLIGHTS}],
        },
    ).json()["fork"]["id"]

    started = proxy.client.post(f"/api/runs/{fork_id}/rerun")
    assert started.status_code == 202
    job = started.json()["job"]
    assert job["command"] == ECHO_AGENT
    assert job["mode"] == "fork"

    finished = wait_for_job(proxy.client, job["id"])
    assert finished["exit_code"] == 0
    assert finished["output"] == [f"fork http://127.0.0.1:8484/r/{fork_id}/openai/v1"]
    # The fork learns its argv the first time anything re-runs it.
    assert proxy.store.get_run(fork_id).command == ECHO_AGENT


def test_rerunning_a_recording_replays_it_onto_a_new_run(proxy) -> None:
    recorded(proxy.store, "TAPE", command=ECHO_AGENT)
    started = proxy.client.post("/api/runs/TAPE/rerun").json()

    assert started["job"]["mode"] == "replay"
    replay_id = started["run"]["id"]
    assert replay_id != "TAPE"
    assert proxy.store.get_run(replay_id).replay_of == "TAPE"
    assert wait_for_job(proxy.client, started["job"]["id"])["output"] == [
        f"replay http://127.0.0.1:8484/r/{replay_id}/openai/v1"
    ]


def test_a_run_with_no_stored_command_says_why_it_cannot_be_rerun(proxy) -> None:
    recorded(proxy.store, "TAPE")
    summary = proxy.client.get("/api/runs/TAPE").json()["run"]
    assert summary["rerun"]["available"] is False
    assert "not launched with `agentvcr run`" in summary["rerun"]["reason"]

    refused = proxy.client.post("/api/runs/TAPE/rerun")
    assert refused.status_code == 400
    assert "no command was stored" in refused.json()["detail"]


def test_a_fork_that_already_ran_is_not_offered_a_second_run(proxy) -> None:
    """Position on the tape is how many steps the fork has: a second run would resume
    in the middle of its own branch (PLAN.md phase 4)."""
    recorded(proxy.store, "TAPE", command=ECHO_AGENT)
    fork_id = proxy.client.post("/api/runs/TAPE/fork", json={"at": 0, "edits": []}).json()["fork"][
        "id"
    ]
    wait_for_job(proxy.client, proxy.client.post(f"/api/runs/{fork_id}/rerun").json()["job"]["id"])
    proxy.store.add_step(_a_step(fork_id))  # the branch it would have recorded

    summary = proxy.client.get(f"/api/runs/{fork_id}").json()["run"]
    assert summary["rerun"]["available"] is False
    assert "already ran" in summary["rerun"]["reason"]
    assert proxy.client.post(f"/api/runs/{fork_id}/rerun").status_code == 400


def test_a_running_rerun_can_be_watched_and_stopped(proxy) -> None:
    """An agent the UI started is one the UI can stop; its run keeps what it recorded."""
    slow = [sys.executable, "-c", "import time; print('working', flush=True); time.sleep(30)"]
    recorded(proxy.store, "TAPE", command=slow)

    job = proxy.client.post("/api/runs/TAPE/rerun").json()["job"]
    assert job["running"] is True

    assert proxy.client.post(f"/api/jobs/{job['id']}/stop").status_code == 200
    finished = wait_for_job(proxy.client, job["id"])
    assert finished["running"] is False
    assert finished["exit_code"] != 0


def test_an_unknown_job_says_jobs_do_not_outlive_the_server(proxy) -> None:
    missing = proxy.client.get("/api/jobs/NOPE")
    assert missing.status_code == 404
    assert "as long as the server" in missing.json()["detail"]


def _a_step(run_id: str):
    from agentvcr.core.models import Step

    return Step(run_id=run_id, idx=0, request={"body": {}}, response={}, status_code=200)


# --------------------------------------------------------------------------------- diff


def test_diff_reports_the_diverging_step(proxy) -> None:
    recorded(proxy.store, "A", result=NOTHING)
    tool_loop(proxy.store, "B", result=FLIGHTS, final="Booked B6918.")
    payload = proxy.client.get("/api/diff", params={"a": "A", "b": "B"}).json()

    assert payload["diff"]["identical"] is False
    assert "diverge at step 0" in payload["diff"]["summary"]
    assert payload["left"]["id"] == "A" and payload["right"]["id"] == "B"


def test_diff_of_two_wire_formats_is_refused_not_called_identical(proxy) -> None:
    recorded(proxy.store, "A")
    tool_loop(proxy.store, "B", result=NOTHING, final="x")
    proxy.store.update_run("B", provider="anthropic")
    proxy.store.update_run("A", provider="openai")
    refused = proxy.client.get("/api/diff", params={"a": "A", "b": "B"})
    assert refused.status_code == 400
    assert "cannot diff two wire formats" in refused.json()["detail"]


# ----------------------------------------------------------------------------------- ui


def test_the_ui_is_served_or_explains_how_to_build_it(proxy) -> None:
    """A wheel ships the bundle; a checkout may not have built it yet. Either is fine —
    silence is not."""
    page = proxy.client.get("/ui/")
    assert page.status_code in (200, 501)
    if page.status_code == 501:
        assert "npm --prefix ui run build" in page.text
    else:
        # Asset URLs carry a content hash and can be cached forever; index.html's URL
        # never changes, so a cached copy of it survives an upgrade and then asks for
        # asset files that are no longer there.
        assert page.headers["cache-control"] == "no-cache"
    assert proxy.client.get("/", follow_redirects=False).headers["location"] == "/ui/"
