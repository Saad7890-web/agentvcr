"""How a child agent process is pointed at a run.

Small, but load-bearing twice over: ``agentvcr run`` and the UI's Re-run button both
hand an unmodified SDK its base URL this way, and the URL shape is a contract with the
proxy's routes (DESIGN.md §4).
"""

from __future__ import annotations

from agentvcr.config import Settings
from agentvcr.core.launch import agent_env, local_base


def test_the_run_id_rides_in_the_base_url_of_every_format() -> None:
    env = agent_env("http://127.0.0.1:8484", run_id="RUN1", mode="fork")
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8484/r/RUN1/openai/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8484/r/RUN1/anthropic"
    assert env["AGENTVCR_RUN"] == "RUN1"
    assert env["AGENTVCR_MODE"] == "fork"


def test_a_wildcard_bind_is_dialed_over_the_loopback() -> None:
    """0.0.0.0 is an address to listen on, not one to connect to."""
    assert local_base(Settings(host="0.0.0.0", port=9000)) == "http://127.0.0.1:9000"
    assert local_base(Settings(host="::", port=9000)) == "http://127.0.0.1:9000"


def test_an_ipv6_host_keeps_its_brackets() -> None:
    assert local_base(Settings(host="::1", port=8484)) == "http://[::1]:8484"
