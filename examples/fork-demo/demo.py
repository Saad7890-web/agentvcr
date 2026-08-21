"""The 30-second story, headless: a failing run, one edit, and a run that works.

This is the Phase 4 acceptance check as a script you can watch (PLAN.md), and the
fallback the launch GIF ships from if the web UI slips:

    record   the agent asks its flight tool, gets nothing back, and gives up
    fork     replace that one tool result — nothing about the agent changes
    re-run   step 0 replays for free; from step 1 the model runs for real and books
    diff     name the step where the branch left the tape, and why

Needs no API key: `fake_upstream.py` plays the model, and it *reads the tool result*,
so the different ending is caused by the edit rather than by a canned script.

    python examples/fork-demo/demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
UPSTREAM_PORT = 8597
PROXY_PORT = 8596

#: What the flight search *should* have returned — the edit the fork applies.
FLIGHTS = {"flights": [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}]}


def wait_for_port(port: int, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise SystemExit(f"nothing came up on port {port}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(HERE.parent / ".demo"))
    args = parser.parse_args()

    workdir = Path(args.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    db = workdir / "agentvcr.db"
    upstream_log = workdir / "upstream-calls.jsonl"
    edit_file = workdir / "flights.json"
    edit_file.write_text(json.dumps(FLIGHTS, indent=2))

    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-not-a-real-key",
        "AGENTVCR_PORT": str(PROXY_PORT),
        "AGENTVCR_DB": str(db),
    }

    def cli(*command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "agentvcr.cli", *command],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )

    upstream = subprocess.Popen(
        [
            sys.executable,
            str(HERE.parent / "fake_upstream.py"),
            "--port",
            str(UPSTREAM_PORT),
            "--log",
            str(upstream_log),
        ],
        cwd=ROOT,
    )
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentvcr.cli",
            "serve",
            "--port",
            str(PROXY_PORT),
            "--db",
            str(db),
            "--openai-upstream",
            f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        wait_for_port(UPSTREAM_PORT)
        wait_for_port(PROXY_PORT)
        agent = [sys.executable, str(HERE.parent / "agent.py")]

        print("== 1. record the run that fails ==")
        recorded = cli("run", "--name", "booking", "--", *agent)
        print(recorded.stdout, recorded.stderr, sep="")
        run_id = run_id_of(recorded.stdout)
        gave_up = recorded.returncode == 1
        print(cli("show", run_id).stdout)

        print("== 2. fork it, with the tool result the search should have returned ==")
        forked = cli(
            "fork", run_id, "--at", "0", "--edit-tool-result", f"search_flights={edit_file}"
        )
        print(forked.stdout, forked.stderr, sep="")
        fork_id = fork_id_of(forked.stdout)
        before = calls_so_far(upstream_log)

        print("== 3. re-run the same agent against the fork ==")
        branched = cli("run", "--mode", "fork", "--run", fork_id, "--", *agent)
        print(branched.stdout, branched.stderr, sep="")
        booked = branched.returncode == 0
        print(cli("show", fork_id).stdout)

        print("== 4. diff the two runs ==")
        difference = cli("diff", run_id, fork_id)
        print(difference.stdout, difference.stderr, sep="")

        print("== verdict ==")
        paid_for = calls_so_far(upstream_log) - before
        steps = json.loads(cli("show", fork_id, "--json").stdout)["steps"]
        ok = True
        ok &= check("the recorded run gave up", gave_up)
        ok &= check("the forked run booked a flight", booked)
        ok &= check(
            "the branch replayed its prefix and paid only for the rest",
            paid_for == len(steps) - 1,
            f"{len(steps)} step(s), {paid_for} upstream call(s)",
        )
        ok &= check(
            "the model saw the edited tool result",
            any("B6918" in json.dumps(entry) for entry in tail(upstream_log, paid_for)),
        )
        ok &= check("the diff reports a difference", difference.returncode == 1)
        return 0 if ok else 1
    finally:
        for process in (upstream, proxy):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    process.kill()


def run_id_of(output: str) -> str:
    """``agentvcr run`` opens with ``run <id> — mode=…``."""
    return output.split()[1]


def fork_id_of(output: str) -> str:
    """``agentvcr fork`` opens with ``fork <id> — from …``."""
    return output.split()[1]


def calls_so_far(log_path: Path) -> int:
    return len([line for line in log_path.read_text().splitlines() if line.strip()])


def tail(log_path: Path, count: int) -> list[str]:
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    return lines[-count:] if count else []


def check(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {label}{f'  ({detail})' if detail else ''}")
    return passed


if __name__ == "__main__":
    raise SystemExit(main())
