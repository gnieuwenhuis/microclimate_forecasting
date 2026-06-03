# Dashboard (thin client)

Static files (`index.html`, `styles.css`, `app.js`) served from the `gh-pages` branch root and
published there by `.github/workflows/inference.yml` (hourly), alongside `forecasts/<id>.json`.

Reads `forecasts/${deployment}.json` from the same origin (`?deployment=` selects it, default
`lethbridge`) and renders: an A-style header (deployment + status pill + a "now" hero showing the
soonest upcoming hour), a dual-axis temperature/PoP chart (hand-rolled inline SVG, with a marker
at each local-midnight day transition), a run-metadata + champion `model_versions` panel, and an
hourly table. Local time only; auto light/dark via `prefers-color-scheme`.

The view is a **sliding `WINDOW_HOURS` (12 h) window** computed from the browser clock at render
time: the chart/table/hero show only `now → now + 12 h`. The published JSON holds the full 48 h
run; filtering client-side keeps the window correct ("next 12 h from now") without re-publishing
every hour, and drops already-elapsed leads (the run's `issue_time` can be several hours old — see
ADR-0019 on HRDPS publish lag).

Accepts any forecast whose `schema_version` major is **1** (`SUPPORTED_SCHEMA_MAJOR`). No build
step, no secrets, no raw observations — only the derived forecast document (ADR-0009).
