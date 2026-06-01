# Inference logger — first slice: in-process inference + log vertical (subsystem 2)

- **Date:** 2026-05-31
- **Status:** Approved (brainstorming) — pending implementation plan
- **Relates to:** ADR-0003 (server-side inference, thin clients), ADR-0007 (logger →
  accumulating store), ADR-0009 (public forecast JSON + attribution; private store),
  ADR-0004 (two models / row-based predict), ADR-0012 (pipeline owns build_features + reshape),
  ADR-0015 (training store), ADR-0008 (logger-first / cold-start pivot).

## Context

Subsystem 2 of the logger-first pivot. With CaSPAr (historical HRDPS forecasts) unavailable,
the service ships **now** publishing a raw-HRDPS **baseline** forecast and **logging each
hourly snapshot forward** (ADR-0007/0008), becoming trainable as labels accumulate. The full
inference Action (ADR-0003) is `build snapshot → predict → publish JSON → log snapshot`, which
spans several concerns; this spec is the **first vertical slice**: the in-process pipeline that
produces and writes a baseline forecast and appends the snapshot to the training store, to
local paths, fully offline-testable. The registry/trained-model loading and the private-repo +
gh-pages git sync are explicit follow-on specs.

## Scope

**In scope**

- `models.baseline.baseline_predictions(rows, threshold)` — the raw-HRDPS baseline forecaster.
- `publication.write_forecast(doc, path)` — fill the stub (atomic, validated JSON write).
- `pipelines.inference.run_inference(...)` — fill the stub: orchestrate build → predict →
  assemble `ForecastDocument` → write → `append_snapshot`. `main()` wires live sources for prod.
- `FORECAST_SCHEMA_VERSION` constant in `contracts/forecast.py`.
- A new ADR recording "baseline is the initial published champion; service live before a
  trained model." README + CONTEXT touch-ups.

**Out of scope (follow-on specs)**

- The model **registry** + trained-model/champion **loading** and **promotion**
  (`publication.registry_store` stays a stub; `read_registry`/`promote` deferred to the
  publish-gate / training work).
- The **GitHub Action deployment**: private-repo store sync + gh-pages publish via
  `DATA_REPO_TOKEN` (ADR-0009). `run_inference` writes to plain local paths; the Action that
  syncs those paths is a separate spec.
- `status` beyond `"ok"`: the `degraded`/`stale` nuance (obs-dependent once trained models run;
  freshness is the Action's concern) is deferred — see §ForecastDocument.

## Components

### 1. `src/microclimate/models/baseline.py` (L4 models)

```python
def baseline_predictions(rows: pd.DataFrame, threshold_mm: float) -> pd.DataFrame
```
Per feature-matrix row (one per `(issue_time, lead_hour)` from `build_features`):
- `pred_temp_c = nwp_temp_c` (raw HRDPS 2 m temperature passthrough).
- `pred_pop = (nwp_precip_mm >= threshold_mm).astype(float)` — 0.0/1.0, identical in meaning to
  `evaluation.nwp_pop_baseline` (the publish-gate floor). Self-contained — does **not** import
  `evaluation` (models/evaluation sibling-independence holds; the one-liner is duplicated, not
  shared).

Module constant `BASELINE_VERSION = "baseline"` for `model_versions`.

Sibling of `temp_model`/`pop_model`; pure (no I/O). Returns the input frame with `pred_temp_c`
and `pred_pop` columns added (mirrors the trained wrappers' per-row prediction shape so the
pipeline reshape is uniform when champion-loading lands).

### 2. `src/microclimate/publication/forecast_writer.py` (L5) — fill `write_forecast`

```python
def write_forecast(doc: ForecastDocument, path: Path) -> None
```
Serialize via `doc.model_dump_json(indent=2)`; write **atomically** (temp file in the same
dir + `os.replace`); create parent dirs. The `ForecastDocument` is already schema-valid by
construction (Pydantic), so writing the model is the validation boundary.

### 3. `src/microclimate/pipelines/inference.py` (L6) — fill `run_inference`

```python
def run_inference(
    deployment_id: str,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    store: TrainingStore,
    forecast_path: Path,
    issue_time: datetime,
) -> ForecastDocument
```
Flow:
1. `config = load_deployment(deployment_id)`.
2. `snapshot = build_snapshot(config, issue_time, nwp, observations)`.
3. `matrix = build_features(snapshot, config)`.
4. `preds = baseline_predictions(matrix, config.label.precip_occurrence_threshold_mm)`.
5. `doc = _assemble_forecast(config, preds, issue_time, last_updated=issue_time)` — the
   row→series reshape (ADR-0012: the pipeline owns it).
6. `write_forecast(doc, forecast_path)`.
7. `store.append_snapshot(snapshot)`.
8. return `doc`.

`_assemble_forecast` builds, per row, `ForecastStep(lead_hour, valid_time, temp_c=pred_temp_c,
pop=clip(pred_pop, 0, 1))`, and the `ForecastDocument` with `FORECAST_SCHEMA_VERSION`,
`deployment_id`, `issue_time`, `last_updated`, `status="ok"`,
`model_versions={"temp": BASELINE_VERSION, "pop": BASELINE_VERSION}`,
`attribution=_ATTRIBUTION`, `series` sorted by `lead_hour`.

`main()` (prod CLI) resolves the live sources from the connector registry
(`get_source(config.nwp.live_connector)` for NWP; `get_source(k)` per station connector_key
for observations), constructs a `TrainingStore` at the configured root, derives `issue_time`
from the current hour (UTC), and calls `run_inference`. (`main()` is exercised by an
arg-parsing test only — the live source wiring is the Action's concern, not unit-tested.)

### 4. `ForecastDocument` assembly

- Add `FORECAST_SCHEMA_VERSION: str = "1.0.0"` to `contracts/forecast.py`.
- `model_versions = {"temp": "baseline", "pop": "baseline"}`.
- `attribution = ["Data Source: Environment and Climate Change Canada (ECCC)"]` (module
  constant `_ATTRIBUTION` in `inference.py`; mandatory per ADR-0009).
- `status = "ok"`. **Deferred:** `degraded` (an obs source failed — only meaningful once
  obs-dependent trained models run; the baseline is NWP-only) and `stale` (a freshness signal
  the Action sets when a run is old/failed). Documented as deferred, not implemented.

## Data flow

`run_inference` → `build_snapshot` (live/injected Datamart NWP ✓ + EnvCanada obs) →
`build_features` (matrix, label-free) → `baseline_predictions` (adds `pred_temp_c`/`pred_pop`)
→ `_assemble_forecast` (→ `ForecastDocument`) → `write_forecast` (JSON at `forecast_path`) →
`store.append_snapshot` (raw snapshot to the `TrainingStore`).

## Error handling

- Missing NWP backbone → `build_snapshot` raises `ForecastUnavailable`/`SourceUnavailable`;
  `run_inference` propagates (a failed Action run; the client sees the previous JSON). A
  transient obs failure → `build_snapshot` emits an NWP-only snapshot (masks); the baseline is
  unaffected.
- `write_forecast` is atomic — a crash never leaves a half-written JSON.
- Ordering: write the forecast first, then `append_snapshot`; any step failing fails the run.

## Testing (offline)

- **`baseline_predictions`** unit test: a synthetic feature matrix → `pred_temp_c` equals
  `nwp_temp_c`; `pred_pop` is 1.0 where `nwp_precip_mm >= threshold` else 0.0 (incl. the
  inclusive boundary).
- **`write_forecast`** unit test: round-trip (write → re-read → `ForecastDocument` equal);
  atomic (no `.tmp` left; partial dir state never a valid `.json`).
- **`run_inference`** integration test (fake NWP/Obs from `tests/fakes` + `TrainingStore(tmp_path)`
  + tmp forecast path): the written JSON validates as a `ForecastDocument` with
  `series` length = `horizon_hours`, `temp_c` == HRDPS passthrough at each lead, `pop` ∈ {0,1},
  `model_versions == {"temp":"baseline","pop":"baseline"}`, non-empty `attribution`,
  `status=="ok"`; and `store.read_snapshots(deployment_id)` returns the logged snapshot.
- **`main()`** arg test (mirrors the existing CLI stub test): missing `--deployment` → `SystemExit`.
- Full gate: ruff format/check, `lint-imports`, pyright strict, pytest.

## Documentation updates (same PR, per CLAUDE.md)

- **New ADR-0016** — "Baseline raw-HRDPS forecaster is the initial published champion; the
  service is live before a trained model" (records the logger-first pivot's pre-model
  publishing decision; relates to ADR-0003/0004/0007/0008). Note `status`/registry/sync
  follow-ons.
- **README** "Project status" — inference logger (baseline forecast + snapshot logging)
  implemented in-process; registry/champion-loading and the private-repo/gh-pages Action
  deferred.
- **CONTEXT.md** — add/confirm a **baseline forecaster** term (raw-HRDPS passthrough temp +
  threshold PoP; the initial champion / the floor the trained model must beat).

## Decomposition (for the plan)

1. `baseline_predictions` + unit test.
2. `FORECAST_SCHEMA_VERSION` + `write_forecast` + unit test.
3. `run_inference` + `_assemble_forecast` + the integration test; `main()` wiring + arg test.
4. ADR-0016 + README + CONTEXT.

## Open items deferred to the plan

- Whether `_assemble_forecast` is a private helper in `inference.py` or a small public function
  (lean: private helper — it's pipeline-internal reshape).
- Exact `issue_time` derivation in `main()` (floor to the hour, UTC) and which NWP `live`
  vs `historical` connector to use (lean: `config.nwp.live_connector` = Datamart).
- Whether `baseline_predictions` returns a copy with added columns or a new 2-column frame
  (lean: copy + add columns, like the labeler, to keep `issue_time`/`lead_hour`/`valid_time`
  available for the reshape).
