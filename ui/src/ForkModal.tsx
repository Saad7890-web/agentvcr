/** Edit a step and branch from it — the fork loop, without leaving the page.
 *
 * The three edit kinds are not three ways of saying the same thing: each one puts the
 * branch in a different place, because of where the thing being edited physically
 * lives in the conversation (DESIGN.md §6). The modal says which, in the line under
 * the picker, because that is the part people get wrong.
 */

import { useEffect, useState } from "react";
import { api } from "./api";
import type { Run, StepDetail, ToolCall } from "./api";
import { CopyLine, ErrorNote, contentText, go, line } from "./ui";

type Kind = "response" | "tool_result" | "request_patch";

/** Steps `0…at-1`, said the way it reads when there are none of them. */
function prefix(at: number): string {
  return at === 0 ? "Nothing replays" : `Steps 0…${at - 1} replay from the tape`;
}

const EXPLAINS: Record<Kind, (at: number) => string> = {
  response: (at) =>
    `${prefix(at)}, step ${at} is answered with this response, and from step ${at + 1} the ` +
    `agent runs for real against what you wrote.`,
  tool_result: (at) =>
    `Steps 0…${at} replay from the tape. The result lives inside request ${at + 1}, so it is ` +
    `rewritten on the way upstream — on every later call too, since each one carries the ` +
    `whole conversation.`,
  request_patch: (at) =>
    `${prefix(at)}, and request ${at} goes live carrying your edited message.`,
};

export default function ForkModal({
  run,
  step,
  recorded,
  onClose,
  onForked,
}: {
  run: Run;
  step: StepDetail;
  recorded: ToolCall[];
  onClose: () => void;
  onForked: () => void;
}) {
  const callable = step.tool_calls.map((call) => call.tool_name ?? "").filter(Boolean);
  const [kind, setKind] = useState<Kind>(callable.length ? "tool_result" : "response");
  const [target, setTarget] = useState<string>(callable[0] ?? "0");
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [made, setMade] = useState<{ fork: Run; command: string } | null>(null);

  useEffect(() => {
    const escape = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [onClose]);

  // Start from what was actually recorded: an edit is a change to something, and
  // retyping a tool result from scratch is how typos get into a branch.
  useEffect(() => {
    setText(prefill(kind, target, step, recorded));
    setError(null);
  }, [kind, target, step.idx]);

  useEffect(() => {
    setTarget(kind === "tool_result" ? (callable[0] ?? "") : "0");
  }, [kind]);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await api.fork(
        run.id,
        step.idx,
        [{ kind, value: asJsonOrText(text), target: kind === "response" ? null : target }],
        name.trim() || undefined,
      );
      setMade({ fork: created.fork, command: created.command });
      onForked();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const runIt = async (forkId: string) => {
    setBusy(true);
    try {
      await api.rerun(forkId);
      go(`/runs/${forkId}`);
      onClose();
    } catch (exc) {
      setError((exc as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="Fork this run" onClick={(e) => e.stopPropagation()}>
        {made ? (
          <>
            <h3>Forked at step {step.idx}</h3>
            <p>
              Fork <span className="mono">{made.fork.id}</span> is ready. It holds no steps yet — it
              collects them when the agent runs against it.
            </p>
            <CopyLine text={made.command} />
            <div className="modal-actions">
              <button className="button ghost" onClick={() => (go(`/runs/${made.fork.id}`), onClose())}>
                Open the fork
              </button>
              <button className="button" disabled={busy || !made.fork.rerun.available} onClick={() => runIt(made.fork.id)}>
                {made.fork.rerun.available ? "Run it now" : "Run it from a terminal"}
              </button>
            </div>
            {!made.fork.rerun.available && <p className="muted small">{made.fork.rerun.reason}</p>}
          </>
        ) : (
          <>
            <h3>Fork {run.id} at step {step.idx}</h3>
            <div className="field">
              <label>What are you changing?</label>
              <div className="choices">
                <Choice value="response" kind={kind} onPick={setKind}>
                  the model's response
                </Choice>
                <Choice value="tool_result" kind={kind} onPick={setKind} disabled={!callable.length}>
                  a tool result
                </Choice>
                <Choice value="request_patch" kind={kind} onPick={setKind}>
                  a prompt message
                </Choice>
              </div>
              <p className="muted small">{EXPLAINS[kind](step.idx)}</p>
              {!callable.length && (
                <p className="muted small">
                  This step asked for no tool calls, so it has no tool result to edit.
                </p>
              )}
            </div>

            {kind === "tool_result" && (
              <div className="field">
                <label htmlFor="tool">Which tool</label>
                <select id="tool" value={target} onChange={(event) => setTarget(event.target.value)}>
                  {callable.map((tool) => (
                    <option key={tool} value={tool}>
                      {tool}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {kind === "request_patch" && (
              <div className="field">
                <label htmlFor="message">Which message</label>
                <select id="message" value={target} onChange={(event) => setTarget(event.target.value)}>
                  {step.conversation.map((message, index) => (
                    <option key={index} value={String(index)}>
                      {index} · {String(message.role ?? "?")} · {line(contentText(message.content), 60)}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="field">
              <label htmlFor="value">New value</label>
              <textarea
                id="value"
                spellCheck={false}
                value={text}
                onChange={(event) => setText(event.target.value)}
                rows={14}
              />
              <p className="muted small">JSON is stored as JSON; anything else is stored as text.</p>
            </div>

            <div className="field">
              <label htmlFor="name">Name this fork (optional)</label>
              <input id="name" value={name} onChange={(event) => setName(event.target.value)} />
            </div>

            {error && <ErrorNote message={error} />}
            <div className="modal-actions">
              <button className="button ghost" onClick={onClose}>
                Cancel
              </button>
              <button className="button" onClick={submit} disabled={busy}>
                {busy ? "forking…" : "Create the fork"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Choice({
  value,
  kind,
  onPick,
  disabled,
  children,
}: {
  value: Kind;
  kind: Kind;
  onPick: (kind: Kind) => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      className={`choice ${kind === value ? "picked" : ""}`}
      disabled={disabled}
      onClick={() => onPick(value)}
    >
      {children}
    </button>
  );
}

function prefill(kind: Kind, target: string, step: StepDetail, recorded: ToolCall[]): string {
  if (kind === "response") return JSON.stringify(step.response, null, 2);
  if (kind === "tool_result") {
    const call = recorded.find((entry) => entry.tool_name === target);
    return JSON.stringify(call ? call.result : {}, null, 2);
  }
  const message = step.conversation[Number(target)] ?? {};
  const content = message.content;
  return typeof content === "string" ? content : JSON.stringify(content ?? "", null, 2);
}

/** Same rule the CLI uses when it reads an edit off disk: JSON if it parses, text if not. */
function asJsonOrText(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
