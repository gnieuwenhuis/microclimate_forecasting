# Dashboard (thin client)

Static files (`index.html`, `styles.css`, `app.js`) served from the `gh-pages` branch root and
published there by `.github/workflows/inference.yml` (hourly), alongside `forecasts/<id>.json`.

Reads `forecasts/${deployment}.json` from the same origin (`?deployment=` selects it, default
`lethbridge`) and renders: an A-style header (deployment + status pill + the next-1-hour "now"
hero), a dual-axis temperature/PoP chart (hand-rolled inline SVG), a run-metadata + champion
`model_versions` panel, and a scrollable hourly table. Local time only; auto light/dark via
`prefers-color-scheme`.

Accepts any forecast whose `schema_version` major is **1** (`SUPPORTED_SCHEMA_MAJOR`). No build
step, no secrets, no raw observations — only the derived forecast document (ADR-0009).
