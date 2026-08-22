/** The shell: a header that says which tape you are looking at, and the route below it. */

import { api } from "./api";
import DiffView from "./DiffView";
import RunList from "./RunList";
import RunView from "./RunView";
import { Empty, Link, useLoad, useRoute } from "./ui";

export default function App() {
  const route = useRoute();
  const { data: status } = useLoad("status", () => api.status());

  return (
    <div className="app">
      <header className="app-header">
        <Link to="/" className="brand">
          agentvcr
        </Link>
        <nav>
          <Link to="/">runs</Link>
        </nav>
        {status && (
          <span className="muted small header-facts">
            v{status.version} · mode {status.mode} · policy {status.mismatch_policy} ·{" "}
            <span title={status.db}>{status.db}</span>
          </span>
        )}
      </header>
      <main>{view(route)}</main>
    </div>
  );
}

function view(route: string[]) {
  if (route.length === 0) return <RunList />;
  if (route[0] === "runs" && route[1]) {
    const step = route[2] === "steps" && route[3] !== undefined ? Number(route[3]) : null;
    return <RunView runId={route[1]} stepIdx={Number.isInteger(step) ? step : null} />;
  }
  if (route[0] === "diff" && route[1] && route[2]) return <DiffView a={route[1]} b={route[2]} />;
  return (
    <Empty>
      Nothing here. <Link to="/">Back to the run list.</Link>
    </Empty>
  );
}
