# agentvcr web UI

React + Vite. Built into `../src/agentvcr/ui_dist/`, which the wheel ships as an
artifact — so `pip install agentvcr` gets the UI with no Node involved.

```bash
npm install
npm run build        # -> ../src/agentvcr/ui_dist, served at /ui by `agentvcr serve`
npm run dev          # UI on :5173, /api proxied to a running `agentvcr serve` on :8484
```

The bundle is gitignored on purpose: a built asset in the tree is a merge conflict
waiting to happen, and CI builds it for the wheel.

## How it is put together

- `api.ts` — the typed client for `/api`, mirroring `agentvcr/server/api.py`. Every call
  is same-origin; the server refuses `/api` from anywhere else, so there is nothing to
  configure and no key to hold.
- `ui.tsx` — hash routing (a static bundle has to survive a reload on a deep link), a
  small `useLoad` hook, and the repeated widgets.
- `RunList` → `RunView` (timeline + `StepInspector` + `ForkModal` + the re-run panel) →
  `DiffView`. That order is also the order they matter in.

State is `useState` and `fetch`. There is no store, no router package and no component
library, because there are five views and they read a REST API.
