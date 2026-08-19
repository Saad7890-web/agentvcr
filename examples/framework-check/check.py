"""Record a real framework agent, then replay it with the upstream switched off.

This is the reality check PLAN.md schedules at the top of Phase 3: positional replay
assumes an agent makes the same LLM calls in the same order every time, and a framework
is where that assumption goes to die (parallel fan-out, retries, extra bookkeeping
calls). Running it needs no API key — `fake_upstream.py` plays the model.

    python examples/framework-check/check.py                     # sequential ReAct agent
    python examples/framework-check/check.py \
        --agent examples/framework-check/parallel_agent.py \
        --expect divergence                                     # parallel fan-out

What it proves, or fails to:

1. the agent runs unmodified through the proxy and its calls land on a tape;
2. replaying that tape reproduces the same final answer;
3. every replayed request matches the tape it was served from (the fingerprint check —
   this is what catches a *reordered* replay, which positional matching alone cannot);
4. with the upstream process dead, the replay still works — so it truly served from
   the tape rather than quietly reaching the network.

`--expect divergence` relaxes (3) for the fan-out probe, where the call order is a race
and can legitimately land either way. What is checked in *both* modes is the invariant
that matters: a replay whose requests drifted from the tape is always flagged. A wrong
replay is allowed; a quietly wrong one is not.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
UPSTREAM_PORT = 8599
PROXY_PORT = 8598


def wait_for_port(port: int, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise SystemExit(f"nothing came up on port {port}")


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, cwd=ROOT, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=str(HERE.parent / "langgraph_agent.py"))
    parser.add_argument("--workdir", default=str(HERE.parent / ".check"))
    parser.add_argument(
        "--expect",
        choices=("clean", "divergence"),
        default="clean",
        help="whether the replay is expected to match the tape (default) or drift from it",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    db = workdir / "agentvcr.db"
    upstream_log = workdir / "upstream-calls.jsonl"

    env = {**os.environ, "OPENAI_API_KEY": "sk-not-a-real-key"}
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
        env={**env, "AGENTVCR_PORT": str(PROXY_PORT), "AGENTVCR_DB": str(db)},
    )
    try:
        wait_for_port(UPSTREAM_PORT)
        wait_for_port(PROXY_PORT)

        print("== recording ==")
        recorded = run(
            [
                sys.executable,
                "-m",
                "agentvcr.cli",
                "run",
                "--name",
                "langgraph",
                "--",
                sys.executable,
                args.agent,
            ],
            env={**env, "AGENTVCR_PORT": str(PROXY_PORT), "AGENTVCR_DB": str(db)},
        )
        print(recorded.stdout, recorded.stderr, sep="")
        if recorded.returncode != 0:
            return fail("the agent did not run through the proxy")

        listing = run([sys.executable, "-m", "agentvcr.cli", "runs", "--db", str(db)])
        print(listing.stdout)
        run_id = next(line.split()[0] for line in listing.stdout.splitlines()[1:] if line.strip())
        show = run([sys.executable, "-m", "agentvcr.cli", "show", run_id, "--db", str(db)])
        print(show.stdout)
        tape = json.loads(
            run(
                [sys.executable, "-m", "agentvcr.cli", "show", run_id, "--json", "--db", str(db)]
            ).stdout
        )
        upstream_calls = len(upstream_log.read_text().strip().splitlines())
        print(f"steps recorded: {len(tape['steps'])}   upstream calls: {upstream_calls}")

        # The whole point: replay must not depend on the model being reachable.
        print("\n== killing the upstream ==")
        upstream.send_signal(signal.SIGTERM)
        upstream.wait(timeout=10)

        print("== replaying ==")
        replayed = run(
            [
                sys.executable,
                "-m",
                "agentvcr.cli",
                "run",
                "--mode",
                "replay",
                "--run",
                run_id,
                "--",
                sys.executable,
                args.agent,
            ],
            env={**env, "AGENTVCR_PORT": str(PROXY_PORT), "AGENTVCR_DB": str(db)},
        )
        print(replayed.stdout, replayed.stderr, sep="")
        if replayed.returncode != 0:
            return fail("the replay did not complete")

        listing = run([sys.executable, "-m", "agentvcr.cli", "runs", "--db", str(db)])
        print(listing.stdout)
        replay_id = next(
            line.split()[0] for line in listing.stdout.splitlines()[1:] if line.strip()
        )
        replay = json.loads(
            run(
                [sys.executable, "-m", "agentvcr.cli", "show", replay_id, "--json", "--db", str(db)]
            ).stdout
        )

        print("== verdict ==")
        ok = True
        ok &= check("the replay is linked to the tape", replay["run"]["replay_of"] == run_id)
        ok &= check(
            "same number of LLM calls",
            len(replay["steps"]) == len(tape["steps"]),
            f"{len(replay['steps'])} vs {len(tape['steps'])}",
        )
        ok &= check(
            "same responses, step for step",
            [s["response"] for s in replay["steps"]] == [s["response"] for s in tape["steps"]],
        )
        # Did the agent actually send what the tape expected, call for call? For a
        # fan-out agent this is a coin toss: the branches race, and the replay can win
        # the race the other way round.
        drifted = [
            s["idx"]
            for s, recorded_step in zip(replay["steps"], tape["steps"], strict=False)
            if s["fingerprint"] != recorded_step["fingerprint"]
        ]
        diverged = [s["idx"] for s in replay["steps"] if s["diverged"]]

        # The invariant, and the one that matters: a wrong replay is never a quiet one.
        ok &= check(
            "drift and divergence agree — nothing served silently wrong",
            drifted == diverged,
            f"drifted {drifted}, flagged {diverged}",
        )
        if args.expect == "clean":
            ok &= check("the agent replayed its calls in the recorded order", not drifted)
        else:
            print(
                f"  NOTE  call order {'flipped' if drifted else 'happened to hold'} this run"
                f" — it is a race, so both outcomes are expected"
            )
        after = len(upstream_log.read_text().strip().splitlines())
        ok &= check(
            "the replay reached the network zero times",
            after == upstream_calls,
            f"{after - upstream_calls} extra call(s)",
        )
        print(f"\n  recorded answer: {recorded.stdout.splitlines()[1:-1]}")
        print(f"  replayed answer: {replayed.stdout.splitlines()[1:-1]}")
        return 0 if ok else 1
    finally:
        for process in (upstream, proxy):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    process.kill()


def check(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {label}{f'  ({detail})' if detail else ''}")
    return passed


def fail(message: str) -> int:
    print(f"  FAIL  {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
