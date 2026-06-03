# Forecast dashboard redesign + actually deploy it — Design Spec

- **Date:** 2026-06-03
- **Relates to:** ADR-0009 (the four homes — forecast JSON + dashboard + registry on gh-pages;
  published artifacts are derived-only, never raw observations), ADR-0016 (live-always), the
  inference-serves-champion slice (PR #29) and far-lead truncation (PR #30) — which produce the
  `status` ∈ `ok`/`stale`/`degraded` and per-task `model_versions` the dashboard surfaces.
- **Status:** Awaiting review → implementation plan
- **Design question answered by:** `dashboard/_prototype.html` (throwaway prototype, 3 variants).
  Verdict captured in `dashboard/_PROTOTYPE_NOTES.md`: **A-header + C-body hybrid**.

## Goal

Make the GitHub Pages site show a useful, good-looking forecast dashboard instead of a bland
rendered README, and make it **actually deploy**. Two defects today:

1. **The dashboard never reaches gh-pages.** `dashboard/{index.html,app.js}` live on `main`, but
   the hourly `inference.yml` only writes `forecasts/<id>.json` into the gh-pages worktree and
   pushes — it never copies the dashboard. With no `index.html` at the gh-pages root, Pages
   renders `README.md`. (Root cause of the screenshot.)
2. **The client rejects the real forecast.** `app.js` compares `doc.schema_version` to the literal
   `"1"`, but the contract emits `FORECAST_SCHEMA_VERSION = "1.0.0"`, so even once deployed the
   thin client would show "Unsupported schema_version 1.0.0."

Success = visiting the Pages URL renders the redesigned dashboard against the live
`forecasts/lethbridge.json` (champion-served, currently `stale`, 42 leads), in light or dark, on
desktop or phone — and a fresh deploy stays correct because the deploy step and the schema match
are guarded by tests.

## Scope

**In:** redesign `dashboard/index.html` + `dashboard/app.js`, add `dashboard/styles.css`; fix the
schema-version match (major-version, not literal); add the dashboard-publish step (+ `.nojekyll`)
to `inference.yml`; two cheap Python regression guards; update `dashboard/README.md` and the
root README "Project status"; delete the prototype artifacts.

**Out:** any change to the forecast contract, the inference/training pipelines, the JSON shape, or
the `status`/`model_versions` semantics (consumed as-is). Multi-deployment switching UI (the page
is lethbridge-only but `?deployment=`-parameterised — no picker). A JS build step, framework, or
JS test runner (the project has none and ADR-0009's "no build step" forbids adding one). Charting
libraries (hand-rolled inline SVG, locked).

## Locked design decisions (from the prototype)

- **Hand-rolled inline SVG** charts, zero dependencies.
- **Auto light/dark** via `prefers-color-scheme` (CSS custom properties; no toggle).
- **Local time only** (no UTC toggle) — `Date`/`toLocale*` render in the viewer's zone.
- **lethbridge-only but `?deployment=`-aware**: `const id = (?deployment ?? "lethbridge")`; fetch
  `forecasts/${id}.json`. No deployment picker UI.
- **Schema match on major version**: supported = `"1"`; accept any `schema_version` whose major
  (`split(".")[0]`) is `"1"`, reject otherwise. (Fixes defect 2 and tolerates future `1.x`.)

## Layout (A-header + C-body hybrid)

Single column, max-width container, top → bottom:

1. **Header (from variant A).** Deployment name (capitalised) + a **status pill**
   (`ok`/`stale`/`degraded`, colour-coded, `title=` hover explaining each), and a "now" hero: the
   **next-1-hour interval** (the `lead_hour == 1` step) shown large — temperature prominent, PoP
   beside it — plus "issued <local time>" and "updated <relative>".
2. **Combined chart (from variant C).** One dual-axis inline-SVG chart: temperature **line** (left
   axis) overlaid on PoP **bars** (right axis, 0–100%), shared local-time x-axis (~6 h labels),
   dashed 0 °C reference line when the temp range spans zero, a small legend.
3. **Run-metadata + model panel (from variant C).** status, issued, updated (relative), horizon
   (`<len(series)> / 48 h`), `schema_version`, and the **per-task `model_versions`** as badges
   (champion version vs `baseline`, visually distinct) — this is what makes it a *verifiable*
   service.
4. **Hourly table (from variant C).** A scrollable table: lead (`+N`), valid time (local), temp
   (1-decimal °), and **PoP as a percentage only — no inline bar** (user's explicit cut).
5. **Attribution footer.** Renders `doc.attribution[]` verbatim (ADR-0009).

## Files & responsibilities

- **`dashboard/index.html`** — static skeleton: `<head>` (charset, viewport, title, `styles.css`),
  a single `#app` mount point, and `<script src="app.js">`. No inline data.
- **`dashboard/styles.css`** (new) — all styling; `:root` light vars + a
  `@media (prefers-color-scheme: dark)` override block; layout, pill/badge, chart text, table.
- **`dashboard/app.js`** — the only logic. `fetch` the forecast (`cache: no-store`), validate
  schema major, then render each section into `#app`. Pure functions building HTML/SVG strings
  (no framework). Helpers: local-time formatters, relative "ago", temp/PoP formatters, SVG line +
  dual-axis builders. The prototype's variant-A header + variant-C chart/metadata/table functions
  are the reference implementation to port (minus the micro-bar, minus the switcher).

## Data contract it reads (unchanged, `contracts/forecast.py`)

`ForecastDocument`: `schema_version` ("1.0.0"), `deployment_id`, `issue_time`, `last_updated`,
`status` ∈ {`ok`,`stale`,`degraded`}, `model_versions: {temp, pop}` (version string or
`"baseline"`), `attribution: string[]` (≥1), `series: ForecastStep[]` where each step has
`lead_hour` (1..48), `valid_time` (aware ISO), `temp_c`, `pop` (0..1). The dashboard treats the
series as the available contiguous prefix (may be < 48 when `stale`).

## Deploy fix — `inference.yml`

In the "Publish forecast JSON to gh-pages" job, **before** `git add -A`, copy the dashboard into
the gh-pages worktree root and assert no-Jekyll:

```bash
cp dashboard/index.html dashboard/app.js dashboard/styles.css gp/
touch gp/.nojekyll
```

- `.nojekyll` disables Jekyll on Pages (hygiene; also future-proofs `_`-prefixed paths).
- The existing `git add -A` + last-wins push already picks these up; the copy is idempotent, so the
  dashboard is re-asserted every hourly run and updates whenever the files change on `main`.
- **Decision:** publish the dashboard from `inference.yml` only (hourly, always runs), not also
  from `training.yml` — the first hourly run after merge populates it and every run keeps it fresh;
  duplicating the copy into the monthly training workflow adds no coverage.
- GitHub Pages must serve from the `gh-pages` branch root (already the case — that's why the README
  renders today); adding `index.html` makes Pages serve it instead of the README. No Pages-settings
  change is in scope, but call it out for the live-validation step.

## Error handling / states

- **Loading** — `#app` shows a neutral "Loading forecast…" until the fetch resolves.
- **Fetch fails / non-OK** (e.g. unknown `?deployment=` → 404) — a clear "Forecast unavailable"
  message; no stack traces, no secrets.
- **Schema major ≠ 1** — "This dashboard doesn't support forecast schema `<v>`." (forward-safe).
- **`status` variants** — pill colour + hover text per status; `stale` also reflected by the
  horizon line (`42 / 48 h`); `degraded` explained as "a baseline served because an expected
  champion was unusable".
- **Empty/short series** — guard `series.length === 0` (show "no forecast steps"); charts/table
  handle a short series (already the live case at 42).
- **No raw observations ever rendered** — the client only reads the derived forecast doc
  (ADR-0009); it has no obs access and adds none.

## Testing strategy

No JS test runner exists (and none is in scope). Guards are cheap Python tests (fit the existing
`pytest` gate, mirror `tests/` layout) plus an out-of-band visual check:

- **Schema-major guard** (`tests/dashboard/test_dashboard_schema.py`, new): read `dashboard/app.js`,
  extract its supported-major constant, assert it equals
  `FORECAST_SCHEMA_VERSION.split(".")[0]`. This is the regression test for defect 2 — if the
  contract bumps major and the client isn't updated, the gate fails.
- **Deploy guard** (same test module): read `.github/workflows/inference.yml` and assert the publish
  job copies `index.html`/`app.js`/`styles.css` into `gp/` and touches `.nojekyll`. Regression test
  for defect 1 — the dashboard can't silently stop deploying.
- **Visual verification (out-of-band, like the pipeline's `workflow_dispatch` validations):** serve
  `dashboard/` locally and render via Playwright against three crafted fixtures — `ok` (full 48),
  `stale` (truncated, the live case), `degraded` (a baseline-fallback `model_versions`) — in light,
  dark, and a phone width. Confirms the pill/metadata/short-series paths. (Recorded in the plan, not
  a CI gate — there's no headless-browser runner in CI.)
- **Full gate green:** `ruff format --check`, `ruff check`, `lint-imports`, `pyright`, `pytest`.
  (No `src/` change expected; `lint-imports`/`pyright` are unaffected but must stay green.)
- **Cleanup verified:** `dashboard/_prototype.html`, `dashboard/_prototype_forecast.json`,
  `dashboard/_PROTOTYPE_NOTES.md`, and root `proto-*.png` are deleted in this change.

## Docs to update in the same PR

- `dashboard/README.md` — describe the redesigned client (sections, schema-major match, local time,
  `?deployment=`), keep the ADR-0009 derived-only/no-build note.
- Root `README.md` "Project status" — note the dashboard is built and auto-deployed via
  `inference.yml`.
- No ADR change: this implements ADR-0009's gh-pages dashboard home; it doesn't alter a decision.
  (If review decides the deploy-coupling to `inference.yml` is decision-worthy, capture it via
  `/grill-with-docs`.)

## Open risks

- **gh-pages contention** is unchanged — inference and training both push; the existing
  base-on-`origin/gh-pages` + last-wins + non-ff-tolerant push covers the dashboard copy too.
- **Pages-settings drift** — if Pages isn't serving from gh-pages root, `index.html` won't show;
  verify during live validation (out of code scope).
- **No CI visual regression** — the Python guards catch the two structural defects, but a CSS/SVG
  visual regression would only be caught by the manual Playwright pass. Acceptable for a static,
  dependency-free client; revisit only if the dashboard grows.
