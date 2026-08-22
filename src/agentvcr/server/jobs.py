"""Re-running an agent from the UI: spawn its stored command, watch it, show its output.

The *Re-run* button is what makes the fork loop a loop (DESIGN.md §10): edit a step,
fork, re-run, watch the new branch land — without leaving the page for a terminal.

**The command is never taken from the request.** It comes from the run being re-run —
the argv ``agentvcr run`` stored when it first recorded — and it is spawned as an argv
list, never through a shell. A request can only ask *which run* to re-run; what that
means was decided when the run was created. Everything the browser can reach here is
also behind the same-origin guard in :mod:`agentvcr.server.api`, because this endpoint
starts a process on the user's machine.
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..core.store import new_id, utcnow

#: How much of a re-run's output is kept in memory, in lines. The full output is the
#: child's own business; this is a tail for the UI to show while it runs.
MAX_OUTPUT_LINES = 500

STATUS_RUNNING = "running"
STATUS_EXITED = "exited"
STATUS_FAILED = "failed"


@dataclass
class Job:
    """One spawned agent process, and what the UI knows about it."""

    id: str
    run_id: str
    mode: str
    command: list[str]
    cwd: str | None
    started_at: str
    status: str = STATUS_RUNNING
    exit_code: int | None = None
    finished_at: str | None = None
    error: str | None = None
    output: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_OUTPUT_LINES))
    process: subprocess.Popen[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "mode": self.mode,
            "command": self.command,
            "cwd": self.cwd,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "exit_code": self.exit_code,
            "error": self.error,
            "output": list(self.output),
            "running": self.status == STATUS_RUNNING,
        }


class JobRunner:
    """The re-runs this server process started, alive only as long as it is.

    Jobs are deliberately in-memory: a re-run is something you watch, and what it
    produces — a run, its steps, its divergence — is on the tape already.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        run_id: str,
        mode: str,
        command: list[str],
        env: dict[str, str],
        cwd: str | None = None,
    ) -> Job:
        job = Job(
            id=new_id(),
            run_id=run_id,
            mode=mode,
            command=list(command),
            cwd=cwd,
            started_at=utcnow(),
        )
        with self._lock:
            self._jobs[job.id] = job
        try:
            job.process = subprocess.Popen(  # noqa: S603 - argv from the tape, no shell
                command,
                env=env,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            # A command that cannot start at all (renamed script, wrong directory) is a
            # finished job with an explanation, not an exception the browser has to parse.
            job.status, job.error, job.finished_at = STATUS_FAILED, str(exc), utcnow()
            job.output.append(f"cannot run {command[0]!r}: {exc}")
            return job
        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

    def _watch(self, job: Job) -> None:
        """Drain the child's output into the job's tail, then record how it ended."""
        process = job.process
        assert process is not None  # only started jobs are watched
        if process.stdout is not None:
            for line in process.stdout:
                with self._lock:
                    job.output.append(line.rstrip("\n"))
        exit_code = process.wait()
        with self._lock:
            job.exit_code = exit_code
            job.status = STATUS_EXITED
            job.finished_at = utcnow()

    def running_for(self, run_id: str) -> Job | None:
        """The live job re-running ``run_id``, if there is one."""
        with self._lock:
            live = (j for j in self._jobs.values() if j.status == STATUS_RUNNING)
            return next((j for j in live if j.run_id == run_id), None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def stop(self, job: Job) -> Job:
        """Ask a running child to stop. Its run keeps whatever steps it recorded."""
        if job.process is not None and job.process.poll() is None:
            job.process.terminate()
        return job

    def stop_all(self) -> None:
        """Server shutdown: an agent the UI started must not outlive the UI."""
        for job in self.list():
            if job.status == STATUS_RUNNING:
                self.stop(job)
