/** One run: its timeline, the step under inspection, and any re-run in flight. */

import { useEffect, useState } from "react";
import { api } from "./api";
import type { Job, Run, Step, ToolCall } from "./api";
import StepInspector from "./StepInspector";
import { Badge, CopyLine, Empty, ErrorNote, Link, count, go, line, ms, statusKind, useLoad, when } from "./ui";

export default function RunView({ runId, stepIdx }: { runId: string; stepIdx: number | null }) {
  const { data, error, loading, reload } = useLoad(`run:${runId}`, () => api.run(runId));
  const [job, setJob] = useState<Job | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // A re-run belongs to the page you started it from, so it is kept until you leave:
  // the run detail stops reporting a job the moment it exits, and its output and exit
  // code are the part worth seeing after it does.
  useEffect(() => setJob(null), [runId]);
  useEffect(() => {
    const running = data?.job;
    if (running) setJob((current) => (current?.id === running.id ? current : running));
  }, [data?.job?.id]);

  // While an agent is running, its output is the interesting thing on the page; when it
  // stops, the run it wrote is, so the timeline is re-read once.
  useEffect(() => {
    if (!job?.running) return;
    const timer = setInterval(async () => {
      try {
        const next = await api.job(job.id);
        setJob(next.job);
        if (!next.job.running) reload();
      } catch (exc) {
        setActionError((exc as Error).message);
      }
    }, 700);
    return () => clearInterval(timer);
  }, [job?.id, job?.running, reload]);

  if (error) return <ErrorNote message={error} />;
  if (!data) return <Empty>{loading ? "loading run…" : "no run"}</Empty>;

  const { run, steps, tool_calls, edits, parent, children } = data;

  const rerun = async () => {
    setActionError(null);
    try {
      const started = await api.rerun(run.id);
      setJob(started.job);
      // A replay is a run of its own, so the page follows the run being written.
      if (started.run.id !== run.id) go(`/runs/${started.run.id}`);
      else reload();
    } catch (exc) {
      setActionError((exc as Error).message);
    }
  };

  return (
    <>
      <RunHeader run={run} parent={parent} children={children} edits={edits} onRerun={rerun} />
      {actionError && <ErrorNote message={actionError} />}
      {job && <JobPanel job={job} onStop={() => api.stopJob(job.id).then((r) => setJob(r.job))} />}
      <div className="split">
        <Timeline run={run} steps={steps} toolCalls={tool_calls} selected={stepIdx} />
        <div className="inspector">
          {stepIdx === null ? (
            <Empty>
              {steps.length ? "Pick a step to inspect it — or to fork from it." : "This run has no steps yet."}
            </Empty>
          ) : (
            <StepInspector run={run} idx={stepIdx} onChanged={reload} />
          )}
        </div>
      </div>
    </>
  );
}

// ------------------------------------------------------------------------- header

function RunHeader({
  run,
  parent,
  children,
  edits,
  onRerun,
}: {
  run: Run;
  parent: Run | null;
  children: Run[];
  edits: { id: number; description: string }[];
  onRerun: () => void;
}) {
  return (
    <div className="card">
      <div className="toolbar">
        <h2>
          <span className="mono">{run.id}</span>
          {run.name && <span className="run-name">{run.name}</span>}
        </h2>
        <div className="toolbar-actions">
          <Badge kind={run.mode}>{run.mode}</Badge>
          <Badge kind={statusKind(run)}>{run.diverged ? "diverged" : run.status}</Badge>
          {parent && (
            <Link className="button ghost" to={`/diff/${parent.id}/${run.id}`}>
              Diff against {parent.id.slice(0, 6)}…
            </Link>
          )}
          <button className="button" onClick={onRerun} disabled={!run.rerun.available} title={run.rerun.reason}>
            {run.rerun.available && run.rerun.mode === "fork" ? "Run this fork" : "Re-run"}
          </button>
        </div>
      </div>

      <dl className="facts">
        <div>
          <dt>created</dt>
          <dd>{when(run.created_at)}</dd>
        </div>
        <div>
          <dt>steps</dt>
          <dd>{run.stats.steps}</dd>
        </div>
        <div>
          <dt>tool runs</dt>
          <dd>{run.stats.tool_calls}</dd>
        </div>
        <div>
          <dt>tokens</dt>
          <dd>{count(run.stats.tokens)}</dd>
        </div>
        <div>
          <dt>model</dt>
          <dd>{run.stats.model ?? "-"}</dd>
        </div>
        <div>
          <dt>format</dt>
          <dd>{run.provider ?? "-"}</dd>
        </div>
      </dl>

      {run.parent_run_id && (
        <p className="lineage">
          forked from <Link to={`/runs/${run.parent_run_id}`}>{run.parent_run_id}</Link> at step{" "}
          {run.fork_step}
          {edits.map((edit) => (
            <span key={edit.id} className="edit-chip">
              edited {edit.description}
            </span>
          ))}
        </p>
      )}
      {run.replay_of && (
        <p className="lineage">
          replay of <Link to={`/runs/${run.replay_of}`}>{run.replay_of}</Link>
          {typeof run.meta.live_from === "number" && <> — went live at step {String(run.meta.live_from)}</>}
        </p>
      )}
      {children.length > 0 && (
        <p className="lineage">
          branches:{" "}
          {children.map((child) => (
            <Link key={child.id} to={`/runs/${child.id}`} className="edit-chip">
              {child.id} at step {child.fork_step}
            </Link>
          ))}
        </p>
      )}
      {run.command && <CopyLine text={run.command.join(" ")} />}
      {!run.rerun.available && run.rerun.reason && <p className="muted small">{run.rerun.reason}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------- re-run job

function JobPanel({ job, onStop }: { job: Job; onStop: () => void }) {
  return (
    <div className={`card job ${job.running ? "job-running" : ""}`}>
      <div className="toolbar">
        <h3>
          {job.running ? "running" : "finished"} — {job.command.join(" ")}
        </h3>
        <div className="toolbar-actions">
          {job.running ? (
            <button className="button ghost" onClick={onStop}>
              Stop
            </button>
          ) : (
            <Badge kind={job.exit_code === 0 ? "good" : "bad"}>exit {job.exit_code ?? "?"}</Badge>
          )}
        </div>
      </div>
      <pre className="output">{job.output.join("\n") || "…"}</pre>
      {job.error && <ErrorNote message={job.error} />}
    </div>
  );
}

// ------------------------------------------------------------------------ timeline

function Timeline({
  run,
  steps,
  toolCalls,
  selected,
}: {
  run: Run;
  steps: Step[];
  toolCalls: ToolCall[];
  selected: number | null;
}) {
  const after = new Map<number, ToolCall[]>();
  for (const call of toolCalls) {
    after.set(call.after_step_idx, [...(after.get(call.after_step_idx) ?? []), call]);
  }
  return (
    <div className="timeline">
      {steps.map((step) => (
        <div key={step.idx}>
          <button
            className={`step ${selected === step.idx ? "selected" : ""} ${step.ok ? "" : "failed"}`}
            onClick={() => go(`/runs/${run.id}/steps/${step.idx}`)}
          >
            <span className="step-idx">{step.idx}</span>
            <span className="step-body">
              <span className="step-line">
                {step.tool_calls.length > 0 ? (
                  <span className="calls">
                    → {step.tool_calls.map((call) => `${call.tool_name}(${line(call.args, 40)})`).join(", ")}
                  </span>
                ) : (
                  <span>{step.error ? `! ${step.error.message ?? "upstream error"}` : line(step.text, 110) || "—"}</span>
                )}
              </span>
              <span className="step-meta muted small">
                {step.model ?? "-"} · {step.status_code ?? "-"} · {ms(step.latency_ms)} · {count(step.tokens)} tok
                {step.diverged && (
                  <>
                    {" "}
                    <Badge kind="warn">diverged</Badge>
                  </>
                )}
              </span>
            </span>
          </button>
          {(after.get(step.idx) ?? []).map((call) => (
            // The proxy never saw this run; it was reconstructed from the next request
            // (DESIGN.md §2), which is why it sits between two steps rather than in one.
            <div key={call.id} className="tool-run">
              <span className="tool-name">↳ {call.tool_name}</span>
              <span className="muted">{line(call.result, 110) || "(no result)"}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
