# Training Pipeline: backfill → train → gate → promote → publish — Design Spec

- **Date:** 2026-06-03
- **Relates to:** ADR-0006 (champion/challenger), ADR-0016 (baseline is the initial champion),
  ADR-0017/0018 (public training store, coalesce + force-push), ADR-0009 (the four homes),
  ADR-0019 (Open-Meteo source, seed backfill). Builds on PR #26/#27.
- **Status:** Awaiting review → implementation plan

## Goal

Turn the stubbed `pipelines/training.py` into the real monthly **retrain pipeline**: pull the
deep seed into the persistent store, train temp + PoP, run the **champion/challenger publish
gate** on a temporal holdout, and on promotion publish the champion model binary (GitHub
Release asset) and the updated `registry.json` (gh-pages) — the path the inference pipeline will
later read to swap the baseline for a trained model.

Success = `run_training("lethbridge")` runs end-to-end (locally and in CI), promotes a model only
when it strictly beats both raw HRDPS and the current champion on the most-recent-12-months
holdout, and leaves `registry.json` + champion assets published. The earlier validated run (temp
MAE skill +0.23, PoP BSS +0.38) would promote both tasks off the baseline.

## Scope

In: `publish_gate`, `registry_store`, a champion-publisher module, a months-based temporal
split, `run_training` orchestration, store-sync + publish wiring in `training.yml`, the
champion-load-for-re-evaluation path. This is the full vertical (one comprehensive spec, incl.
real external publishing) per the brainstorming decision.

Out: changing the **inference** pipeline to *read* the registry/champion (separate slice — it
still publishes the baseline until then); the §1b forward-capture; multi-deployment tuning.

## Architecture

```
run_training(deployment_id):
  load_deployment + validate_config_sources
  store-sync: TRAINING_STORE_ROOT = cloned training-data branch (CI) / local dir (dev)
  backfill_store(seed.start → now)            # incremental, idempotent, throttled (ADR-0019)
  rows = assemble_from_store(config, store)
  train, calib, test = temporal_split(rows, holdout_months)   # NEW months-based split
  for task in (temp, pop):
      challenger = TemperatureRegressor()/PrecipOccurrenceClassifier().fit(...)   # calib for PoP
      champion   = load_champion(registry, task)  # model from Release asset, or None→baseline
      result     = evaluate_challenger(task, challenger, champion, baseline_on(test), test)
      if result.promote:
          entry = RegistryEntry(version, release_asset_url, promoted_at, holdout_metrics)
          manifest = registry_store.promote(manifest, task, deployment_id, entry)
          write champion binary + manifest to local OUTPUT dir
  → workflow: gh release upload promoted binaries; push registry.json to gh-pages; force-push store
```

Layering: `publish_gate` stays in `evaluation` (L4, imports no model classes — keeps the
models/evaluation sibling independence); `registry_store` + champion-publisher in `publication`
(L5); `run_training` in `pipelines` (L6) ties them together.

## Components

### 1. Temporal split — `pipelines/training_data.py`

Add `temporal_split(rows, *, holdout_months, calib_months) -> (train, calib, test)`:
- `test` = rows whose `issue_time` is within the most recent `holdout_months` (config:
  `training.holdout_months`, default 12).
- `calib` = the `calib_months` immediately before `test` (disjoint slice for PoP isotonic
  calibration — CONTEXT "calibration slice"). Default `calib_months = 3`.
- `train` = everything before `calib`.
- Split on whole `issue_time` boundaries (never across one), like `chronological_split`.
- Raise if any slice is empty (too little history). Keep the existing fraction-based
  `chronological_split` for the notebook; the pipeline uses `temporal_split`.

### 2. Publish gate — `evaluation/publish_gate.py`

Implement the existing signature
`evaluate_challenger(task, challenger, champion, baseline, holdout) -> GateResult`:
- `challenger`: a fitted model exposing `.predict(holdout) -> pd.Series`.
- `champion`: a fitted model, or `None` when the current champion is the baseline.
- `baseline`: `holdout` with the raw-HRDPS baseline column already attached
  (`temp` → `nwp_temp_c_h{lead}` passthrough; `pop` → `nwp_pop_baseline`).
- Metric per task (overall, across the holdout): **temp → MAE**, **pop → Brier**.
- Compute `m_challenger`, `m_baseline`, and `m_champion` (champion preds on the *same* holdout;
  when `champion is None`, `m_champion = m_baseline`).
- **Promote iff `m_challenger < m_baseline` AND `m_challenger < m_champion`** (strictly beats
  both; lower MAE/Brier is better). Ties/regressions do not promote (fail-safe).
- Return `GateResult(promote, reason, metrics={...})` where `metrics` carries
  `mae`/`brier`, the skill vs HRDPS (`mae_skill`/`bss`), and the champion metric — these become
  the `RegistryEntry.holdout_metrics`.
- **Reuse `evaluation.metrics`** for the baseline column and skill computation (the same
  `temp_skill_by_lead` / `nwp_pop_baseline` / `pop_skill_by_lead` the notebook uses) so the
  gate's verdict and the reported skill plots can't diverge. "Overall" metric = the aggregate
  over all holdout rows (not a per-lead vote); per-lead skill is reported for visibility only.
- Pure: no I/O, no model-class imports (predict is duck-typed via the passed objects); may
  import `evaluation.metrics` (same L4 module).

### 3. Registry store — `publication/registry_store.py`

- `read_registry(path) -> RegistryManifest`: parse `registry.json` if present, else empty
  `RegistryManifest()`.
- `promote(manifest, task, deployment_id, entry) -> RegistryManifest`: return a new manifest
  with `entries[manifest_key(deployment_id, task)] = entry` (immutable update).
- `write_registry(manifest, path)`: serialize to `registry.json` (pretty JSON).

### 4. Champion publisher — `publication/champion_publisher.py` (new)

- `champion_version(deployment_id, task, run_time) -> str`: deterministic, e.g.
  `f"{deployment_id}-{task}-{run_time:%Y%m%dT%H%M}Z"`. (No `Math.random`/`Date.now` surprises —
  `run_time` is passed in.)
- `release_tag(version) -> str` and `asset_filename(version) -> str` (e.g. `{version}.joblib`).
- `release_asset_url(repo, version) -> str`:
  `https://github.com/{repo}/releases/download/{release_tag}/{asset_filename}` — **computed
  before upload** so the `RegistryEntry` can be written without waiting on the upload.
- `save_champion(model, out_dir, version) -> Path`: write the model `.joblib` to a local
  staging dir the workflow uploads from.
- The actual `gh release upload` + gh-pages push live in the workflow (Python stays I/O-light
  and offline-testable; matches `forecast_writer`).

### 5. Champion load (re-evaluation) — in `run_training`

`load_champion(config, registry, task, work_dir) -> model | None`:
- Look up `manifest_key(deployment_id, task)`; if absent or `version == "baseline"` → return
  `None` (gate treats `None` as the raw-HRDPS baseline).
- Else download `entry.release_asset_url` to `work_dir` and `TemperatureRegressor.load` /
  `PrecipOccurrenceClassifier.load`. Download via `connectors.http.http_get_bytes` (reuse) →
  write file → `load`. Network failure → raise (don't silently skip the champion comparison).

### 6. Orchestration — `pipelines/training.py`

`run_training(deployment_id, *, nwp=None, observations=None, store=None, output_dir=None,
registry_path=None, now=None, do_backfill=True) -> TrainingSummary`:
- Defaults resolve the real `get_source(...)`, `TrainingStore(TRAINING_STORE_ROOT)`,
  `output_dir` (staging for assets + registry.json), `registry_path` (the read location),
  `now = datetime.now(UTC)`. Injected params make it hermetically testable.
- `validate_config_sources(config)`.
- If `do_backfill`: `backfill_store(config, …, hrdps_issue_times(seed.start, now))`.
- `rows = assemble_from_store(config, store)`; `train, calib, test = temporal_split(...)`.
- Fit challengers; load champions; gate each task; on promote, `save_champion` +
  `registry_store.promote` + `write_registry`.
- Return `TrainingSummary{rows, per-task GateResult, promoted: list[Task], registry_path}` and
  `print` a concise report (rows, split sizes, per-task metric vs baseline/champion, promote
  decision + reason).
- `main()`: argparse `--deployment` (+ `--no-backfill`); call `run_training`; exit non-zero only
  on real failure (a no-promote is a normal, successful outcome).

### 7. CI — `.github/workflows/training.yml`

Per matrix deployment:
1. **Store-sync (in):** clone the `training-data` branch into `store/` via
   `https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git`
   (resurrect the removed inference pattern); bootstrap an orphan branch if absent.
   `TRAINING_STORE_ROOT=store`.
2. `uv run python -m microclimate.pipelines.training --deployment <id>`
   (writes promoted assets + `registry.json` under `OUTPUT_DIR`).
3. **Publish (on promotion):** `gh release upload {tag} {asset}` for each promoted task
   (`--clobber`); push the updated `registry.json` to `gh-pages`.
4. **Store-sync (out):** coalesced force-push of `store/` back to `training-data` (ADR-0018).
   `permissions: contents: write`; serialize with a `concurrency` group.
   Keep the monthly cron; **drop the `config/deployments/**` push trigger** (a config edit
   shouldn't kick a full retrain) — leave `workflow_dispatch` for manual runs.

## Data flow & invariants

- **Store persists across runs** (training-data branch) → CI backfill is incremental; first run
  is the full ~3,500-run seed (~30 min), later monthly runs add ~120.
- **Gate is fail-safe:** strictly-beats-both, so a non-improving challenger never demotes the
  champion; a missing/!improving result leaves the registry untouched.
- **Deterministic asset URL** lets `registry.json` reference the asset before upload; the
  workflow uploads to exactly that tag/filename.
- Temp and PoP promote **independently** (separate entries, separate Release assets).

## Error handling / edge cases

- **No champion yet** (`baseline`): `load_champion` → `None`; gate compares challenger vs
  baseline only (champion metric = baseline metric) — first promotion swaps off the baseline.
- **Too little history** for `temporal_split` (e.g. < holdout+calib+1 month): raise a clear
  error (don't train on an empty/degenerate split).
- **Champion download fails:** raise (the comparison must be apples-to-apples; don't silently
  promote against a missing champion).
- **Backfill gaps:** `backfill_store` already logs+skips unavailable runs (ADR-0019).
- **gh release upload / gh-pages push failure:** workflow step fails loudly; the store push is a
  separate step so a publish failure doesn't corrupt the store. Re-run is safe (idempotent
  backfill; `--clobber` upload; registry write is last-wins).

## Testing strategy

- **Unit (pure, no network):**
  - `evaluate_challenger`: promote when strictly better; **no** promote on tie, on regression, or
    when it beats champion but not baseline (and vice-versa); `champion=None` path.
  - `registry_store`: `read_registry` (present/absent), `promote` immutability, round-trip
    `write_registry` → `read_registry`.
  - `champion_publisher`: deterministic version/tag/url/filename; `save_champion` writes a
    loadable file.
  - `temporal_split`: test = recent `holdout_months`; calib disjoint; empty-slice raises.
- **Orchestration (`run_training`, hermetic):** fake NWP + fake obs (varying precip so PoP has
  both classes) + tmp store + tmp `output_dir`/`registry_path`, `do_backfill=True`, small window.
  Assert: store populated, both tasks evaluated, promote decision drives a correct local
  `registry.json` (entries present with computed URLs) + saved champion binaries, summary
  returned. A second run with an injected better/worse challenger exercises promote/no-promote.
- **Workflow:** `yamllint`-style load check; steps are shell/`gh` (not unit-tested) — covered by
  a first real `workflow_dispatch` run.
- Full gate green: `ruff format --check`, `ruff check`, `lint-imports` (verify L4/L5/L6 layering
  holds — `publish_gate` imports no model classes), `pyright`, `pytest`.

## Open risks

- **First CI run is heavy** (~30-min full backfill) before the store exists on the branch — run
  it once via `workflow_dispatch` to seed, then monthly runs are light.
- **Champion re-evaluation cost:** downloads + predicts the champion each run — negligible vs the
  backfill.
- **Inference doesn't read the registry yet** — promoted champions sit published but unused until
  the inference-reads-registry slice lands (explicitly out of scope here).
- ADR note: implementing the gate + registry publish resolves the "deferred follow-on" called
  out in ADR-0016; update ADR-0016's consequences (and ADR-0009's four-homes status) in the
  implementation PR.
