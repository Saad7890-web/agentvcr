/** Small shared pieces: routing, data loading, and the handful of repeated widgets. */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

// ------------------------------------------------------------------ hash routing

/** Path segments of the current hash route, e.g. `["runs", "01K9…", "steps", "6"]`.
 *  Hash routing on purpose: the bundle is served as static files, so a deep link has
 *  to survive a reload without the server knowing any of the UI's routes. */
export function useRoute(): string[] {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
}

export function go(path: string): void {
  window.location.hash = path;
}

export function Link({ to, children, className }: { to: string; children: ReactNode; className?: string }) {
  return (
    <a className={className} href={`#${to}`}>
      {children}
    </a>
  );
}

// -------------------------------------------------------------------- data loading

export type Loaded<T> = { data: T | null; error: string | null; loading: boolean; reload: () => void };

/** Fetch on mount and whenever `key` changes; `reload()` re-fetches on demand. */
export function useLoad<T>(key: string, load: () => Promise<T>): Loaded<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const latest = useRef(load);
  latest.current = load;

  useEffect(() => {
    let live = true;
    setLoading(true);
    latest
      .current()
      .then((value) => live && (setData(value), setError(null)))
      .catch((exc: Error) => live && setError(exc.message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [key, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload };
}

// ------------------------------------------------------------------------ widgets

export function Badge({ kind, children }: { kind?: string; children: ReactNode }) {
  return <span className={`badge badge-${kind ?? "neutral"}`}>{children}</span>;
}

export function ErrorNote({ message }: { message: string }) {
  return <div className="error-note">{message}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

/** A pretty-printed JSON block, collapsed by default when it is long. */
export function Json({ value, open = false, label }: { value: unknown; open?: boolean; label?: string }) {
  const [shown, setShown] = useState(open);
  const text = JSON.stringify(value, null, 2) ?? "null";
  if (!shown) {
    return (
      <div className="json-block">
        <button className="link-button" onClick={() => setShown(true)}>
          show {label ?? "JSON"} ({text.length.toLocaleString()} chars)
        </button>
      </div>
    );
  }
  return (
    <div className="json-block">
      {label && (
        <button className="link-button" onClick={() => setShown(false)}>
          hide {label}
        </button>
      )}
      <pre>{text}</pre>
    </div>
  );
}

export function CopyLine({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="copy-line">
      <code>{text}</code>
      <button
        className="link-button"
        onClick={() => {
          navigator.clipboard?.writeText(text).then(
            () => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            },
            () => undefined,
          );
        }}
      >
        {copied ? "copied" : "copy"}
      </button>
    </div>
  );
}

// -------------------------------------------------------------------- formatting

export function when(iso: string | null): string {
  return iso ? iso.slice(0, 19).replace("T", " ") : "-";
}

export function ms(value: number | null): string {
  return value === null ? "-" : `${value.toLocaleString()}ms`;
}

export function count(value: number | null): string {
  return value === null ? "-" : value.toLocaleString();
}

/** Any value as one line, for a table cell. */
export function line(value: unknown, limit = 120): string {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > limit ? `${flat.slice(0, limit - 1)}…` : flat;
}

/** Message content is a string in one wire format and a list of blocks in the other;
 *  both have to read as text in the inspector. */
export function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        const part = block as Record<string, unknown>;
        if (typeof part.text === "string") return part.text;
        return JSON.stringify(part, null, 2);
      })
      .join("\n");
  }
  if (content === null || content === undefined) return "";
  return JSON.stringify(content, null, 2);
}

export function statusKind(run: { status: string; diverged: boolean }): string {
  if (run.diverged) return "warn";
  if (run.status === "failed") return "bad";
  return run.status === "active" ? "live" : "good";
}
