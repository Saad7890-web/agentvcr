"""How a child agent process is pointed at a run.

Two things launch an agent — ``agentvcr run`` and the UI's *Re-run* button — and both
have to hand it the same environment: the run id rides in the base URL
(``/r/<id>/openai/v1``, DESIGN.md §4), which is what lets an unmodified SDK carry it
without any header support. That URL shape is a contract between the proxy's routes
and every launcher, so it is written once, here, rather than in each of them.
"""

from __future__ import annotations

from ..config import Settings
from ..providers import REGISTRY

#: Environment variable each SDK reads its base URL from, by provider name.
BASE_URL_ENV = {"openai": "OPENAI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL"}


def local_base(settings: Settings) -> str:
    """The proxy's address as a child process on this machine should dial it.

    A server bound to a wildcard is still reached over the loopback — an agent handed
    ``http://0.0.0.0:8484`` would be one that connects by accident, not by design.
    """
    host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    if ":" in host:  # a literal IPv6 address needs brackets to sit in a URL
        host = f"[{host}]"
    return f"http://{host}:{settings.port}"


def agent_env(base: str, *, run_id: str, mode: str) -> dict[str, str]:
    """The environment that points an unmodified agent at ``run_id``.

    Layer it over the environment the agent should otherwise inherit — its API key
    included, since the proxy forwards credentials it never stores (DESIGN.md §11).
    """
    env = {"AGENTVCR_RUN": run_id, "AGENTVCR_MODE": mode}
    for name, variable in BASE_URL_ENV.items():
        provider = REGISTRY.get(name)
        if provider is not None:
            env[variable] = f"{base.rstrip('/')}/r/{run_id}{provider.mount_path}"
    return env
