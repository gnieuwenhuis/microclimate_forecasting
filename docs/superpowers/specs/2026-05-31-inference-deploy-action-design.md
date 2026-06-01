# Inference deployment Action — hourly data collection to a public store (subsystem 2 follow-on)

- **Date:** 2026-05-31
- **Status:** Approved (brainstorming) — pending implementation plan
- **Relates to:** ADR-0003 (server-side inference Action), ADR-0007 (logger → training-data
  branch), ADR-0009 (public-derived vs private-raw store — **amended here**), ADR-0010 (ACIS
  dropped → ECCC-only), ADR-0015 (training store), ADR-0016 (baseline forecaster), the
  `pipelines.inference` CLI.

## Context & goal

The logger-first pivot needs **data accumulating forward now**. `pipelines.inference.run_inference`
(in-process baseline forecast + `append_snapshot`) is built and merged; the scheduled GitHub
Action that runs it hourly and **persists** the snapshots is the missing piece. The existing
`.github/workflows/inference.yml` scaffold has the right skeleton (hourly cron, matrix,
`DATA_REPO_TOKEN`) but **collects nothing**: it writes the store to a throwaway runner dir and
never commits it, doesn't install eccodes (so the Datamart GRIB2 decode would fail), and
doesn't publish the forecast.

**Key decision (amends ADR-0009):** ADR-0009 made the raw store *private* solely because of
ACIS's unsettled redistribution rights. **ADR-0010 dropped ACIS** for ECCC (`envcanada` obs +
HRDPS), both **redistributable with attribution**. So the store now holds only redistributable
data and can be **public** — committed to a `training-data` branch in this repo via the
built-in `GITHUB_TOKEN`, with **zero external setup** (no private repo, no PAT/secret). This is
the fastest path to data flowing and is exactly the escape hatch ADR-0009 named.

## Scope

**In scope**

- Flesh out `.github/workflows/inference.yml` so the hourly run **persists** snapshots:
  install eccodes; check out the `training-data` branch into a store dir; run inference per
  deployment writing the store there; commit + push new snapshots to `training-data` via
  `GITHUB_TOKEN`. Single job that loops deployments (avoids the shared-branch push race).
- **ADR-0017** recording the store-is-public decision; an amendment note on **ADR-0009**.
- Docs: README "Project status", CONTEXT "Training store" → public branch, a short runbook.

**Out of scope (follow-on specs)**

- The **gh-pages forecast-JSON publish** (the public live-service surface) — secondary to data
  collection; the forecast is written to the runner FS and discarded for now.
- **Registry / champion-loading** (a trained model superseding the baseline).
- **Per-deployment parallel push** (one deployment today; the single-job loop is correct and
  race-free until there are many).
- Any new **Python** — `inference.main()` already reads `TRAINING_STORE_ROOT`; the Action just
  points it at the checked-out store dir. The unused `DATA_REPO_TOKEN` env is removed.

## The workflow (`.github/workflows/inference.yml`)

Triggers unchanged: `schedule: cron "0 * * * *"` + `workflow_dispatch`. Add a top-level
`concurrency: { group: inference, cancel-in-progress: false }` so overlapping hourly runs
serialize (no two runs push `training-data` at once). Permissions: `contents: write` (so the
job can push via `GITHUB_TOKEN`).

Replace the discover+matrix structure with a **single `run` job** (matrix would race on the
shared branch):

1. `actions/checkout@v4` (code, default branch).
2. **Install eccodes:** `sudo apt-get update && sudo apt-get install -y libeccodes0` (+ a quick
   `uv run python -c "import cfgrib"` sanity check after `uv sync`). cfgrib's `findlibs` locates
   the apt-installed `libeccodes`.
3. `uv sync`.
4. **Check out the store branch** into `./store`: a second `actions/checkout@v4` with
   `ref: training-data`, `path: store`. **First-run bootstrap:** if `training-data` doesn't
   exist yet, create it as an empty orphan and use an empty `./store` instead (a guarded shell
   step: try to fetch the branch; if absent, `git init`-style bootstrap so the subsequent push
   creates it).
5. **Run inference per deployment:**
   ```bash
   for id in $(ls config/deployments/*.yml | xargs -n1 basename | sed 's/\.yml$//'); do
     TRAINING_STORE_ROOT=store uv run python -m microclimate.pipelines.inference --deployment "$id"
   done
   ```
   (`forecast_json` writes to the runner FS and is discarded — gh-pages publish deferred.)
6. **Commit + push the store** to `training-data`:
   ```bash
   cd store
   git add -A
   git diff --cached --quiet || git commit -m "data: snapshots $(date -u +%FT%TZ)"
   git push origin HEAD:training-data
   ```
   Auth is the checkout's `GITHUB_TOKEN`. `git diff --cached --quiet ||` skips the commit/push
   when there are no new files (e.g. a fully-degraded run that wrote nothing — defensive).

The exact bootstrap mechanics for the orphan branch (e.g. `git checkout --orphan` + empty
commit on first run vs `actions/checkout` with a create-if-missing guard) are a plan detail;
the requirement is: first run creates `training-data`, subsequent runs append to it.

## ADR / store-location change

- **New ADR-0017** — "Training store is public (`training-data` branch), committed by the
  inference Action; the private store / `DATA_REPO_TOKEN` are retired." Rationale: ADR-0010
  dropped ACIS, so the store holds only ECCC-redistributable data (attribution still required);
  this is ADR-0009's own escape hatch. Consequences: zero-setup data collection via
  `GITHUB_TOKEN`; the store branch grows with hourly commits (compaction/prune is future work,
  consistent with ADR-0015); training (subsystem 3) reads the `training-data` branch.
- **Amend ADR-0009** with a header note pointing to ADR-0017 (mirroring how ADR-0009 amended
  ADR-0003/0007): the raw store is now public, not a private repo.

## Data flow

GitHub Action (hourly) → checkout code + `training-data`→`./store` → per deployment:
`run_inference` builds a snapshot (live Datamart + EnvCanada) → baseline forecast (forecast JSON
to runner FS, discarded) → `append_snapshot` into `./store` → commit + push `./store` to
`training-data`. Over time the branch accumulates the per-deployment partitioned-Parquet store
that subsystem 3 (training) will read.

## Error handling

- A failed run (e.g. Datamart unreachable → `build_snapshot` raises) fails that hour's job; no
  partial commit (nothing was appended, or `git diff --cached --quiet` skips). The next hourly
  run retries. Acceptable per ADR-0003 (freshness bounded by last successful run).
- eccodes-missing would fail loudly at the cfgrib import sanity check (step 2) — a clear CI
  failure, not a silent bad forecast.
- The store push uses the built-in token; no secret to misconfigure.

## Validation (NOT unit-testable — this is infra)

This subsystem is workflow YAML + git/shell, validated by **running** it, not by pytest:

- **Operator step (gate):** after merge, trigger the workflow via `workflow_dispatch` and
  confirm (a) the job is green, (b) eccodes/cfgrib import passes, (c) a commit appears on the
  `training-data` branch containing `snapshots/deployment_id=lethbridge/ym=YYYYMM/*.parquet`.
  This requires live network to MSC Datamart + ECCC from the GitHub runner (both public).
- No new Python ⇒ no new unit tests; existing coverage (`run_inference` integration test +
  the `network`-marked Datamart test) stands. Optionally lint the workflow with `actionlint`
  locally (not wired into CI — YAGNI).
- The repo must have **Actions read/write permission** (default for `GITHUB_TOKEN` with
  `permissions: contents: write` in the workflow) — noted in the runbook.

## Documentation updates (same PR, per CLAUDE.md)

- ADR-0017 + the ADR-0009 amendment note.
- README "Project status": the inference Action collects snapshots hourly to the public
  `training-data` branch (gh-pages forecast publish + registry still to come).
- CONTEXT "Training store": public `training-data` branch (not a private repo), per ADR-0017.
- A short **runbook** (in the workflow header comment and/or README): how to dispatch, where
  data lands, the first-run branch bootstrap, and the `contents: write` permission requirement.

## Decomposition (for the plan)

1. ADR-0017 + ADR-0009 amendment note (the decision record, independent of the YAML).
2. Rewrite `.github/workflows/inference.yml`: single job, eccodes install + cfgrib check,
   store-branch checkout + first-run bootstrap, per-deployment inference loop, commit + push,
   `concurrency` + `permissions`, drop `DATA_REPO_TOKEN`.
3. README + CONTEXT + runbook.

## Open items deferred to the plan

- Orphan-branch bootstrap mechanics (checkout-with-guard vs an explicit `git checkout --orphan`
  + empty initial commit step).
- Whether to keep the `discover` job (deployment list) or inline the `ls config/deployments`
  loop in the single job (lean: inline the loop — one job, no matrix, simplest).
- The exact eccodes apt package(s) (`libeccodes0` vs `libeccodes-dev`/`libeccodes-tools`) —
  finalized by the cfgrib-import sanity check in the workflow.
