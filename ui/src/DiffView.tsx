/** Two runs, side by side. What is compared is behavior, not bytes (DESIGN.md §7):
 *  ids, timestamps and token counts differ between any two live calls and are left out
 *  so that what is left is the answer to "did the agent decide something else?". */

import { api } from "./api";
import type { DiffPair, Run } from "./api";
import { Badge, Empty, ErrorNote, Link, statusKind, useLoad } from "./ui";

export default function DiffView({ a, b }: { a: string; b: string }) {
  const { data, error, loading } = useLoad(`diff:${a}:${b}`, () => api.diff(a, b));
  if (error) return <ErrorNote message={error} />;
  if (!data) return <Empty>{loading ? "diffing…" : "no diff"}</Empty>;

  const { left, right, diff } = data;
  // Identical steps are the boring majority; what earns a row is a change, or a step
  // one run made and the other did not.
  const changed = diff.steps.filter(
    (pair) => pair.changes.length > 0 || pair.left_idx === null || pair.right_idx === null,
  );

  return (
    <>
      <div className="card">
        <div className="toolbar">
          <h2>Diff</h2>
          <Badge kind={diff.identical ? "good" : "warn"}>{diff.identical ? "identical" : "diverged"}</Badge>
        </div>
        <div className="diff-sides">
          <Side run={left} label="a" />
          <Side run={right} label="b" />
        </div>
        <p className="summary">{diff.summary}</p>
      </div>

      {diff.identical ? (
        <Empty>
          Every aligned step matches: same requests, same decisions, same tool results. A recording
          diffed against its own replay landing here is what proves the replay reproduced the run.
        </Empty>
      ) : (
        <table className="grid diff">
          <thead>
            <tr>
              <th>Step</th>
              <th>What changed</th>
              <th>{left.id.slice(0, 8)}…</th>
              <th>{right.id.slice(0, 8)}…</th>
            </tr>
          </thead>
          <tbody>
            {changed.map((pair, index) => (
              <Rows key={index} pair={pair} left={left} right={right} />
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

function Side({ run, label }: { run: Run; label: string }) {
  return (
    <div className="diff-side">
      <span className="side-label">{label}</span>
      <Link to={`/runs/${run.id}`} className="mono">
        {run.id}
      </Link>
      <Badge kind={run.mode}>{run.mode}</Badge>
      <Badge kind={statusKind(run)}>{run.diverged ? "diverged" : run.status}</Badge>
      <span className="muted">
        {run.stats.steps} step(s) · {run.label}
      </span>
    </div>
  );
}

function Rows({ pair, left, right }: { pair: DiffPair; left: Run; right: Run }) {
  const label =
    pair.left_idx === pair.right_idx
      ? String(pair.left_idx)
      : `${pair.left_idx ?? "-"}/${pair.right_idx ?? "-"}`;

  if (!pair.changes.length) {
    const side = pair.left_idx !== null ? left : right;
    return (
      <tr className="unmatched">
        <td className="mono">{label}</td>
        <td colSpan={3}>
          this step exists only in{" "}
          <Link to={`/runs/${side.id}/steps/${pair.left_idx ?? pair.right_idx}`}>{side.id}</Link>
        </td>
      </tr>
    );
  }

  return (
    <>
      {pair.changes.map((change, index) => (
        <tr key={index}>
          <td className="mono">{index === 0 ? label : ""}</td>
          <td>
            <span className={`where where-${change.where}`}>{change.where}</span> {change.detail}
          </td>
          <td className="side-left">{change.left ?? <span className="muted">—</span>}</td>
          <td className="side-right">{change.right ?? <span className="muted">—</span>}</td>
        </tr>
      ))}
    </>
  );
}
