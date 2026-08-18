from __future__ import annotations

from agentvcr.core.models import STATUS_COMPLETED
from agentvcr.core.recorder import REDACTION_PLACEHOLDER, RunRouter, redact_headers
from agentvcr.core.store import Store
from agentvcr.providers import OPENAI


def body(*contents: str) -> dict:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": c} for c in contents],
    }


def test_redact_headers_keeps_shape_but_not_secrets() -> None:
    out = redact_headers(
        {
            "Authorization": "Bearer sk-secret",
            "X-Api-Key": "sk-also-secret",
            "Content-Type": "application/json",
            "Host": "localhost:8484",
        }
    )
    assert out == {
        "authorization": REDACTION_PLACEHOLDER,
        "x-api-key": REDACTION_PLACEHOLDER,
        "content-type": "application/json",
    }


def test_explicit_run_id_wins_and_is_created_on_demand(store: Store) -> None:
    router = RunRouter(store)
    run = router.resolve(provider=OPENAI, body=body("a"), mode="record", run_id="RUN123")
    assert run.id == "RUN123"
    # a second call with an unrelated message list still lands on the named run
    again = router.resolve(provider=OPENAI, body=body("z"), mode="record", run_id="RUN123")
    assert again.id == "RUN123"
    assert len(store.list_runs()) == 1


def test_growing_conversations_chain_onto_one_run(store: Store) -> None:
    router = RunRouter(store)
    first = router.resolve(provider=OPENAI, body=body("a"), mode="record")
    second = router.resolve(provider=OPENAI, body=body("a", "b"), mode="record")
    third = router.resolve(provider=OPENAI, body=body("a", "b", "c"), mode="record")
    assert first.id == second.id == third.id
    assert len(store.list_runs()) == 1


def test_an_unrelated_conversation_starts_a_new_run(store: Store) -> None:
    router = RunRouter(store)
    first = router.resolve(provider=OPENAI, body=body("a", "b"), mode="record")
    other = router.resolve(provider=OPENAI, body=body("different"), mode="record")
    assert other.id != first.id
    assert len(store.list_runs()) == 2


def test_a_retry_of_the_same_request_stays_on_its_run(store: Store) -> None:
    router = RunRouter(store)
    first = router.resolve(provider=OPENAI, body=body("a"), mode="record")
    retry = router.resolve(provider=OPENAI, body=body("a"), mode="record")
    assert retry.id == first.id


def test_the_longest_matching_prefix_wins(store: Store) -> None:
    """Two active runs both match; the more specific one gets the call."""
    router = RunRouter(store)
    router.resolve(provider=OPENAI, body=body("a"), mode="record", run_id="SHORT")
    router.resolve(provider=OPENAI, body=body("a", "b"), mode="record", run_id="LONG")

    chained = router.resolve(provider=OPENAI, body=body("a", "b", "c"), mode="record")
    assert chained.id == "LONG"


def test_idle_runs_are_closed(store: Store) -> None:
    now = [1000.0]
    router = RunRouter(store, idle_timeout_s=60.0, clock=lambda: now[0])
    first = router.resolve(provider=OPENAI, body=body("a"), mode="record")

    now[0] += 61.0
    later = router.resolve(provider=OPENAI, body=body("a", "b"), mode="record")

    assert later.id != first.id  # the old run was expired before matching
    assert store.get_run(first.id).status == STATUS_COMPLETED


def test_close_all_completes_tracked_runs(store: Store) -> None:
    router = RunRouter(store)
    run = router.resolve(provider=OPENAI, body=body("a"), mode="record")
    router.close_all()
    assert store.get_run(run.id).status == STATUS_COMPLETED
