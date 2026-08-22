/**
 * The control API, typed. Mirrors the serializers in `agentvcr/server/api.py`.
 *
 * Every request is same-origin: the UI is served by the very process it talks to, and
 * that server refuses `/api` calls from anywhere else — so there is no base URL to
 * configure and no key to carry.
 */

export type RunStats = {
  steps: number;
  tool_calls: number;
  tokens: number | null;
  model: string | null;
  diverged: boolean;
  errors: number;
};

export type Rerun = {
  available: boolean;
  mode?: string;
  command?: string[];
  reason?: string;
};

export type Run = {
  id: string;
  name: string | null;
  created_at: string;
  mode: string;
  status: string;
  parent_run_id: string | null;
  fork_step: number | null;
  replay_of: string | null;
  command: string[] | null;
  provider: string | null;
  upstream_url: string | null;
  meta: Record<string, unknown>;
  label: string;
  stats: RunStats;
  diverged: boolean;
  rerun: Rerun;
};

export type RequestedCall = {
  tool_name: string | null;
  tool_call_id: string | null;
  args: unknown;
};

export type Step = {
  idx: number;
  model: string | null;
  status_code: number | null;
  ok: boolean;
  latency_ms: number | null;
  tokens: number | null;
  usage: Record<string, number> | null;
  diverged: boolean;
  started_at: string | null;
  fingerprint: string | null;
  has_chunks: boolean;
  messages: number;
  text: string | null;
  tool_calls: RequestedCall[];
  error: { message?: string } | null;
};

export type Message = { role?: string; content?: unknown; [key: string]: unknown };

export type StepDetail = Step & {
  request: { headers: Record<string, string>; body: Record<string, unknown> };
  response: Record<string, unknown> | null;
  conversation: Message[];
  request_settings: Record<string, unknown>;
  full_text: string | null;
};

export type ToolCall = {
  id: number;
  run_id: string;
  after_step_idx: number;
  tool_name: string;
  args: unknown;
  result: unknown;
  tool_call_id: string | null;
  result_preview: string | null;
};

export type Edit = {
  id: number;
  run_id: string;
  step_idx: number;
  kind: string;
  patch: Record<string, unknown> | null;
  description: string;
};

export type Job = {
  id: string;
  run_id: string;
  mode: string;
  command: string[];
  cwd: string | null;
  started_at: string;
  finished_at: string | null;
  status: string;
  exit_code: number | null;
  error: string | null;
  output: string[];
  running: boolean;
};

export type RunDetail = {
  run: Run;
  steps: Step[];
  tool_calls: ToolCall[];
  edits: Edit[];
  parent: Run | null;
  children: Run[];
  job: Job | null;
};

export type StepPayload = { step: StepDetail; tool_calls: ToolCall[] };

export type DiffChange = { where: string; detail: string; left: string | null; right: string | null };
export type DiffPair = { left_idx: number | null; right_idx: number | null; changes: DiffChange[] };
export type DiffPayload = {
  left: Run;
  right: Run;
  diff: { left: string; right: string; identical: boolean; summary: string; steps: DiffPair[] };
};

export type Status = {
  version: string;
  mode: string;
  db: string;
  schema_version: number;
  upstreams: Record<string, string>;
  base_url: string;
  mismatch_policy: string;
};

export type EditSpec = { kind: string; value: unknown; target?: string | null };

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) {
    // FastAPI puts the prose in `detail`, and these messages are written to be read —
    // they say what to do instead, so they are shown to the user verbatim.
    const detail = body && typeof body.detail === "string" ? body.detail : response.statusText;
    throw new ApiError(detail, response.status);
  }
  return body as T;
}

export const api = {
  status: () => request<Status>("/status"),
  runs: () => request<{ runs: Run[] }>("/runs"),
  run: (id: string) => request<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  step: (id: string, idx: number) =>
    request<StepPayload>(`/runs/${encodeURIComponent(id)}/steps/${idx}`),
  diff: (a: string, b: string) =>
    request<DiffPayload>(`/diff?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`),
  fork: (id: string, at: number, edits: EditSpec[], name?: string) =>
    request<{ fork: Run; edits: Edit[]; command: string }>(
      `/runs/${encodeURIComponent(id)}/fork`,
      { method: "POST", body: JSON.stringify({ at, edits, name: name || null }) },
    ),
  rerun: (id: string) =>
    request<{ job: Job; run: Run }>(`/runs/${encodeURIComponent(id)}/rerun`, { method: "POST" }),
  job: (id: string) => request<{ job: Job; run: Run | null }>(`/jobs/${encodeURIComponent(id)}`),
  stopJob: (id: string) =>
    request<{ job: Job }>(`/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" }),
};
