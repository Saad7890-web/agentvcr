/** The run list: every recording, replay and fork, nested by where it came from. */

import { useState } from "react";
import { api } from "./api";
import type { Run } from "./api";
import { Badge, CopyLine, Empty, ErrorNote, count, go, statusKind, useLoad, when } from "./ui";

type Node = { run: Run; children: Node[]; relation: string | null };

/** Nest each run under the run it came from — a fork under its parent, a replay under
 *  the tape it replays. Anything whose origin is not in the list stands on its own. */
function lineage(runs: Run[]): Node[] {
  const nodes = new Map(runs.map((run) => [run.id, { run, children: [], relation: null } as Node]));
  const roots: Node[] = [];
  for (const node of nodes.values()) {
    const { parent_run_id, replay_of, fork_step } = node.run;
    const parent = parent_run_id ? nodes.get(parent_run_id) : replay_of ? nodes.get(replay_of) : undefined;
    if (!parent) {
      roots.push(node);
      continue;
    }
    node.relation = parent_run_id ? `forked at step ${fork_step}` : "replay";
    parent.children.push(node);
  }
  return roots;
}

export default function RunList() {
  const { data, error, loading } = useLoad("runs", () => api.runs());
  const [compare, setCompare] = useState<string[]>([]);

  const toggle = (id: string) =>
    setCompare((current) =>
      current.includes(id) ? current.filter((other) => other !== id) : [...current, id].slice(-2),
    );

  if (error) return <ErrorNote message={error} />;
  if (loading && !data) return <Empty>loading runs…</Empty>;

  const runs = data?.runs ?? [];
  if (!runs.length) {
    return (
      <Empty>
        <p>No runs on this tape yet. Point an agent at the proxy and it will appear here:</p>
        <CopyLine text="agentvcr run --name my-agent -- python agent.py" />
        <p className="muted">
          Or change one line in the agent itself — <code>base_url="http://localhost:8484/openai/v1"</code>{" "}
          — and run it however you normally do.
        </p>
      </Empty>
    );
  }

  const rows: { node: Node; depth: number }[] = [];
  const walk = (nodes: Node[], depth: number) =>
    nodes.forEach((node) => {
      rows.push({ node, depth });
      walk(node.children, depth + 1);
    });
  walk(lineage(runs), 0);

  return (
    <>
      <div className="toolbar">
        <h2>Runs</h2>
        <div className="toolbar-actions">
          {compare.length === 2 ? (
            <button className="button" onClick={() => go(`/diff/${compare[0]}/${compare[1]}`)}>
              Diff {compare[0].slice(0, 6)}… → {compare[1].slice(0, 6)}…
            </button>
          ) : (
            <span className="muted">tick two runs to diff them</span>
          )}
        </div>
      </div>
      <table className="grid">
        <thead>
          <tr>
            <th />
            <th>Run</th>
            <th>Created</th>
            <th>Mode</th>
            <th>Status</th>
            <th className="num">Steps</th>
            <th className="num">Tools</th>
            <th className="num">Tokens</th>
            <th>Model</th>
            <th>Name</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ node, depth }) => {
            const run = node.run;
            return (
              <tr key={run.id} className="row" onClick={() => go(`/runs/${run.id}`)}>
                <td onClick={(event) => event.stopPropagation()}>
                  <input
                    type="checkbox"
                    aria-label={`compare ${run.id}`}
                    checked={compare.includes(run.id)}
                    onChange={() => toggle(run.id)}
                  />
                </td>
                <td style={{ paddingLeft: `${depth * 1.25 + 0.6}rem` }}>
                  <span className="mono">{run.id}</span>
                  {node.relation && <span className="muted small"> {node.relation}</span>}
                </td>
                <td className="muted">{when(run.created_at)}</td>
                <td>
                  <Badge kind={run.mode}>{run.mode}</Badge>
                </td>
                <td>
                  <Badge kind={statusKind(run)}>{run.diverged ? "diverged" : run.status}</Badge>
                </td>
                <td className="num">{run.stats.steps}</td>
                <td className="num">{run.stats.tool_calls}</td>
                <td className="num">{count(run.stats.tokens)}</td>
                <td className="muted">{run.stats.model ?? "-"}</td>
                <td>{run.name ?? <span className="muted">{run.label}</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}
