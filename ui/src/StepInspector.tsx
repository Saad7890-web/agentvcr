/** One step in full: what was sent, what came back, and the button that forks it. */

import { useState } from "react";
import { api } from "./api";
import type { Message, Run } from "./api";
import ForkModal from "./ForkModal";
import { Badge, Empty, ErrorNote, Json, contentText, count, line, ms, useLoad } from "./ui";

export default function StepInspector({
  run,
  idx,
  onChanged,
}: {
  run: Run;
  idx: number;
  onChanged: () => void;
}) {
  const { data, error, loading } = useLoad(`step:${run.id}:${idx}`, () => api.step(run.id, idx));
  const [forking, setForking] = useState(false);

  if (error) return <ErrorNote message={error} />;
  if (!data) return <Empty>{loading ? "loading step…" : "no step"}</Empty>;

  const { step, tool_calls } = data;

  return (
    <div className="card">
      <div className="toolbar">
        <h3>Step {step.idx}</h3>
        <div className="toolbar-actions">
          {step.diverged && <Badge kind="warn">diverged from the tape</Badge>}
          {!step.ok && <Badge kind="bad">HTTP {step.status_code}</Badge>}
          <button className="button" onClick={() => setForking(true)}>
            Fork from here
          </button>
        </div>
      </div>

      <dl className="facts">
        <div>
          <dt>model</dt>
          <dd>{step.model ?? "-"}</dd>
        </div>
        <div>
          <dt>latency</dt>
          <dd>{ms(step.latency_ms)}</dd>
        </div>
        <div>
          <dt>tokens</dt>
          <dd>{count(step.tokens)}</dd>
        </div>
        <div>
          <dt>messages sent</dt>
          <dd>{step.messages}</dd>
        </div>
        <div>
          <dt>stream</dt>
          <dd>{step.has_chunks ? "recorded chunks" : "final message only"}</dd>
        </div>
      </dl>

      <section>
        <h4>The model answered</h4>
        {step.error ? (
          <ErrorNote message={step.error.message ?? "upstream error"} />
        ) : step.tool_calls.length > 0 ? (
          <ul className="calls-list">
            {step.tool_calls.map((call, position) => (
              <li key={call.tool_call_id ?? position}>
                <span className="tool-name">{call.tool_name}</span>
                <code>{line(call.args, 200)}</code>
              </li>
            ))}
          </ul>
        ) : step.full_text ? (
          <p className="text-block">{step.full_text}</p>
        ) : (
          <p className="muted">no text and no tool calls</p>
        )}
      </section>

      {tool_calls.length > 0 && (
        <section>
          <h4>Then these tools ran</h4>
          {tool_calls.map((call) => (
            <div key={call.id} className="tool-detail">
              <div className="tool-name">
                {call.tool_name}
                <code>{line(call.args, 120)}</code>
              </div>
              <Json value={call.result} label={`result of ${call.tool_name}`} open />
            </div>
          ))}
        </section>
      )}

      <section>
        <h4>Conversation sent ({step.conversation.length})</h4>
        <ol className="messages">
          {step.conversation.map((message, position) => (
            <MessageRow key={position} index={position} message={message} />
          ))}
        </ol>
      </section>

      <section>
        <h4>Request</h4>
        <Json value={step.request_settings} label="settings (model, tools, temperature…)" />
        <Json value={step.request.body} label="raw request body" />
        <Json value={step.response} label="raw response" />
      </section>

      {forking && (
        <ForkModal
          run={run}
          step={step}
          recorded={tool_calls}
          onClose={() => setForking(false)}
          onForked={onChanged}
        />
      )}
    </div>
  );
}

function MessageRow({ index, message }: { index: number; message: Message }) {
  const role = typeof message.role === "string" ? message.role : "?";
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const text = contentText(message.content);
  return (
    <li className="message">
      <span className={`role role-${role}`}>
        {index} {role}
      </span>
      <div className="message-body">
        {text && <pre className="text-block">{text}</pre>}
        {calls.length > 0 && <Json value={calls} label="tool calls in this message" />}
        {!text && calls.length === 0 && <span className="muted">(empty)</span>}
      </div>
    </li>
  );
}
