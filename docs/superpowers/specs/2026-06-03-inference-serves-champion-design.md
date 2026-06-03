# Inference Serves the Champion (falls back to baseline) — Design Spec

- **Date:** 2026-06-03
- **Relates to:** ADR-0016 (baseline is the initial champion; trained model swaps in via champion/challenger), ADR-0006 (champion/challenger), ADR-0009 (the four homes — forecast JSON + registry.json on gh-pages, models as Release assets), ADR-0011/0012 (snapshot/feature path), the training-pipeline slice (PR #28).
- **Status:** Awaiting review → implementation plan

## Goal

Teach the hourly inference pipeline to **load and serve the promoted champion model** from
`registry.json` (per task, temp/pop independently), falling back to the raw-HRDPS baseline when
no champion is published or one can't be used — and **publish the forecast JSON to gh-pages** so
the served forecast actually reaches thin clients. This closes the loop opened by the training
pipeline: today's promotions (champion Release assets + `registry.json` on gh-pages) become the
forecast clients read.

Success = `run_inference` reads `registry.json`, serves the champion when present and usable
(stamping `model_versions` with its version), falls back to baseline otherwise (marking
`status="degraded"` only when a champion was *expected* but unusable), and the inference workflow
publishes the forecast JSON to gh-pages. The service never goes dark (ADR-0016).

## Scope

In: a shared L5 `load_champion`; `run_inference` champion-or-baseline serving with fallback +
status/model_versions; per-task model_versions; gh-pages forecast publish in `inference.yml`;
refactor `training.py` to use the shared `load_champion`.

Out: changing the champion/challenger gate or the registry format; a persistent champion cache;
multi-deployment tuning; the `stale` status (run-freshness) — only `ok`/`degraded` are produced
here.

## Architecture

```
inference.yml (hourly): clone gh-pages -> run_inference(registry_path=gp/registry.json,
                        forecast into gp/...) -> push gh-pages

run_inference(config, *, nwp, observations, forecast_path, issue_time,
              registry_path, work_dir, fetch_bytes=http_get_bytes):
  snapshot = build_snapshot(...); matrix = build_features(...)
  base = baseline_predictions(matrix, threshold)        # pred_temp_c, pred_pop, lead_hour, valid_time
  manifest = read_registry(registry_path)
  for task in ("temp", "pop"):
      version, preds, degraded = _serve_task(task, manifest, matrix, base, config, work_dir, fetch_bytes)
      # version = champion version or "baseline"; preds = pd.Series; degraded = bool
  status = "degraded" if any task degraded else "ok"
  frame = base with pred_temp_c <- temp preds, pred_pop <- pop preds
  doc = _assemble_forecast(config, frame, issue_time, last_updated, model_versions, status)
  write_forecast(doc, forecast_path)
```

Layering: `load_champion` moves to `publication` (L5), imported by both `pipelines.training` and
`pipelines.inference` (L6 → L5, downward — clean). `baseline_predictions` (L4 models) and
`build_features` stay as-is.

## Components

### 1. `publication/champion_loader.py` (new, L5) — `load_champion`

Move the function currently in `pipelines/training.py` here, unchanged in behavior:
```text
load_champion(deployment_id, registry_path, task, work_dir, *, fetch_bytes=lambda u: http_get_bytes(u)) -> object | None
  manifest = read_registry(registry_path)
  entry = manifest.entries.get(manifest_key(deployment_id, task))
  if entry is None or entry.version == "baseline": return None
  download entry.release_asset_url via fetch_bytes -> work_dir/asset_filename(version)
  return TemperatureRegressor.load(...) if task=="temp" else PrecipOccurrenceClassifier.load(...)
```
- `pipelines/training.py` imports `load_champion` from here and deletes its local copy (behavior identical; its `_do_promote`/orchestration unchanged).
- `champion_loader` may import `models`, `publication.registry_store`, `publication.champion_publisher`, `connectors.http`, `contracts.registry` — all ≤ L5. (Importing `models` from `publication` is a downward import; confirm `lint-imports` is happy — publication is L5, models is L4.)

### 2. `pipelines/inference.py` — serve champion or baseline

- Add a helper `_serve_task(task, manifest, matrix, base, config, work_dir, fetch_bytes) -> tuple[str, pd.Series, bool]`:
  - `entry = manifest.entries.get(manifest_key(config.deployment_id, task))`.
  - **No real champion** (`entry is None` or `version=="baseline"`): return
    `("baseline", base["pred_temp_c" if temp else "pred_pop"], False)` — normal, not degraded.
  - **Real champion expected**: `champion = load_champion(...)`; try `preds = champion.predict(matrix)`;
    on success return `(entry.version, preds, False)`. On `ConnectorError`/`SourceUnavailable`
    (download) or `ValueError` (the model's `feature_schema_version` mismatch — stale champion
    refused) or any load error: log the reason and return
    `("baseline", base[...], True)` (degraded — a champion was expected but unusable).
- `run_inference` gains params `registry_path: Path`, `work_dir: Path`, and
  `fetch_bytes: Callable[[str], bytes] = …` (injectable for tests). It builds `base`, reads the
  manifest, calls `_serve_task` for both tasks, composes the prediction frame (override
  `pred_temp_c`/`pred_pop` from each task's served series), and assembles the doc with
  `model_versions={"temp": tver, "pop": pver}` and `status`.
- `predict` returns a Series indexed like `matrix`; assign back by index so the
  `(issue_time, lead_hour)` rows stay aligned.

### 3. `pipelines/inference.py::_assemble_forecast`

Add `model_versions: dict[Literal["temp","pop"], str]` and `status: Literal["ok","degraded"]`
params (drop the hardcoded `{"temp": BASELINE_VERSION, ...}` and `status="ok"`). The reshape into
`ForecastStep` series is otherwise unchanged.

### 4. `pipelines/inference.py::main` + `.github/workflows/inference.yml`

- `main()` resolves `registry_path` and `work_dir` from env (e.g. `REGISTRY_PATH`,
  `CHAMPION_CACHE_DIR`) with sensible defaults, and a forecast output path that the workflow
  points into the gh-pages worktree.
- `inference.yml` (the `run` job): clone the `gh-pages` branch into `gp/` via
  `https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git`
  (bootstrap orphan if absent), set `REGISTRY_PATH=gp/registry.json`, run inference per deployment
  writing the forecast JSON to its gh-pages path (`config.output.forecast_json`, rooted in `gp/`),
  then commit + push `gp/` to `gh-pages`. `permissions: contents: write`; a workflow-wide
  `concurrency` group (gh-pages is shared) with `cancel-in-progress: false`. Keep the hourly cron
  + `workflow_dispatch`. (Mirror the gh-pages publish pattern proven by `training.yml`: base the
  worktree on `origin/gh-pages`, push `HEAD:gh-pages`.)

## Data flow & invariants

- **Per-task independence:** temp and pop each serve champion-or-baseline independently;
  `model_versions` records the actual producer per task; a mixed run (temp champion + pop
  baseline-because-no-entry) is `status="ok"`.
- **`degraded` ⇔ expected-but-unusable:** set only when `entry` names a real champion that fails
  to download/load/predict. Absent/`"baseline"` entry → `ok`.
- **Stale champion refused:** the model's existing `predict` guard raises on
  `feature_schema_version` mismatch; inference catches it → baseline + degraded (CONTEXT: "a
  stale-feature champion is refused").
- **Never dark:** every code path yields a writable forecast (baseline is always computable from
  the matrix), so the hourly product always publishes.

## Error handling / edge cases

- Registry file missing/unreadable → `read_registry` returns an empty manifest → both tasks
  baseline, `status="ok"` (no champion was expected). (If `registry.json` is present but corrupt,
  `read_registry` raising is acceptable — the run fails and the next hour retries; or catch and
  treat as empty. **Decision: catch a corrupt-registry parse error, log, treat as empty manifest →
  baseline/ok**, so a bad publish can't dark the product.)
- Champion download/load failure for an expected champion → that task baseline + degraded.
- NWP/obs unavailable for the chosen run → `build_snapshot` raises as today; the hourly retry
  covers it (unchanged).
- `gh-pages` clone/push failure in CI → the publish step fails loudly; re-run is safe (forecast
  write is idempotent last-wins).

## Testing strategy

- **`load_champion` (unit, hermetic):** returns `None` for no-entry and for `version=="baseline"`;
  for a real entry, an injected `fetch_bytes` returns a saved tiny model's bytes and it loads the
  right class per task.
- **`run_inference` (unit, hermetic, no network):** inject fake NWP/obs + a tmp `registry_path` +
  injected `fetch_bytes`:
  1. empty/absent registry → both `model_versions=="baseline"`, `status=="ok"`, forecast written.
  2. real temp + pop champions (saved tiny models) → `model_versions` carry the versions,
     `status=="ok"`, served preds differ from baseline.
  3. expected champion, `fetch_bytes` raises → that task `"baseline"`, `status=="degraded"`.
  4. expected champion whose `feature_schema_version` ≠ current → caught, `"baseline"`,
     `status=="degraded"`.
  5. temp champion present, pop no-entry → `model_versions={"temp": ver, "pop":"baseline"}`,
     `status=="ok"`.
  6. corrupt `registry.json` → treated as empty → baseline/ok (no raise).
- **`training.py`** unchanged behavior after the `load_champion` import move — its existing test
  still passes.
- **Workflow:** YAML-load check; the gh-pages clone/push + champion HTTP download are CI-only,
  validated by a `workflow_dispatch` run (the champion published by the training seed run is the
  fixture).
- Full gate green: `ruff format --check`, `ruff check`, `lint-imports` (confirm L5 `champion_loader`
  importing `models` is legal; L6 inference importing it is downward), `pyright`, `pytest`.

## Open risks

- **`publication` (L5) importing `models` (L4):** intended downward, but verify `lint-imports`
  contracts allow it (the layered contract should; the "models/evaluation independent siblings"
  contract is about those two, not publication). If a contract forbids it, put `champion_loader`
  in a layer that may import models, or inject the loader classes — resolve during implementation.
- **gh-pages contention:** inference (hourly) and training (monthly) both push gh-pages. The
  workflow-wide concurrency group serializes within each workflow, but inference and training are
  *different* workflows — a rare interleaving could cause a non-fast-forward. Mitigate by basing
  the worktree on `origin/gh-pages` and pushing `HEAD:gh-pages` (a failed push just retries next
  hour); a cross-workflow lock is out of scope.
- First inference run after this lands will serve the champions promoted by the training seed run
  — validate `model_versions` flips from `"baseline"` to the real versions in the published JSON.
