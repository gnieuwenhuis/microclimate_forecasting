# Local notebook model training & exploration — design

- **Date:** 2026-05-31
- **Status:** Approved — implementation plan written (`docs/superpowers/plans/2026-05-31-notebook-model-training.md`) and implemented
- **Relates to:** ADR-0004 (two LightGBM models), ADR-0011 (snapshot normalization
  boundary), ADR-0012 (feature builder read-time transform), ADR-0006 (multi-deployment /
  champion-challenger), ADR-0009 (public-derived / private-raw store).

## Purpose

Enable a developer to **train the temp and PoP models locally in a notebook and explore the
results**, so model quality is verifiable during development. This is a *model-development*
surface, **not** part of the production deployment — but it must drive the **same shared
code** the production training pipeline will use, so the path never forks and never bitrots.

Reaching a trained model also surfaces two feature-engineering prerequisites that ADR-0012
explicitly deferred: **label attachment** (`FEATURE_ROW` → `TRAINING_ROW`) and
**training-data assembly** over a date range. Both are built here as shared, tested code.

## Scope

**In scope**

1. `features.attach_labels` — pure label-attachment (`FEATURE_ROW` + observed target values →
   `TRAINING_ROW`).
2. `pipelines.training_data` — `assemble_training_rows(...)` + a local-Parquet-cached
   `assemble_or_load(...)`; the shared assembly seam.
3. `models.TemperatureRegressor` / `models.PrecipOccurrenceClassifier` — LightGBM wrappers
   with **row-based `predict`**, plus PoP calibration.
4. `evaluation.metrics` — skill scores vs raw-HRDPS baseline (per lead hour) + PoP
   Brier/BSS + reliability bins.
5. `notebooks/model_dev.ipynb` — a **thin** orchestration-and-plots notebook.
6. A fast CI **smoke** test of the assemble → fit → predict → metrics path.
7. Doc updates: a new ADR; amendments to ADR-0004 / ADR-0012; CONTEXT.md terms.

**Explicitly out of scope (deferred)**

- The **private training-store read/write path** (ADR-0009/0007). Local dev backfills
  snapshots from CaSPAr historical instead of reading an accumulated store.
- The production **training pipeline orchestration** (`pipelines.training.run_training`),
  the **publish gate**, the **inference pipeline**, and registry/forecast publication.
- **Walk-forward / rolling-origin CV** — the temporal holdout below is the v1 evaluation;
  CV is a possible later add-on.

## Design intent (the one idea that drives everything)

The leakage-critical and fitted code stays **isolated and unit-testable**, and the notebook
holds **zero logic**:

- The **labeler is pure** — it takes an already-read target-observation frame, so it has no
  connectors. The only *future-facing* read (target observations at `valid_time`, which are
  after `issue_time`) happens in the **assembly orchestrator** in `pipelines`, where a
  training-only future read is legal. It is categorically absent from the shared
  `build_snapshot` / `build_features` path, preserving ADR-0011's no-leakage guarantee.
- The notebook calls `assemble_or_load`, the model wrappers, and `evaluation.metrics`, then
  plots. Anything reusable lives in `src/microclimate` and is tested.

## Components

Layering (existing contract): `contracts` → `config` → `connectors` → `features` →
`evaluation` / `models` → `publication` → `pipelines`. import-linter forbids only
`features.feature_builder` from importing `connectors` — a *pure* labeler elsewhere in
`features` is permitted.

### 1. `features/labeler.py` — `attach_labels` (L3, pure)

```python
def attach_labels(
    matrix: pd.DataFrame,        # FEATURE_ROW (label-free)
    target_obs: pd.DataFrame,    # target-station observations indexed/keyed by timestamp
    threshold_mm: float,         # config.precip_occurrence_threshold_mm
) -> pd.DataFrame:               # TRAINING_ROW
```

- For each row, look up the target station's observed temperature at `valid_time` →
  `label_temp_c`, and observed precip → `label_precip_occurrence = int(precip >= threshold_mm)`.
- Both labels are **nullable**: where the target obs for that `valid_time` is missing, the
  label is `NaN` / null (matching `TRAINING_ROW`; rows with missing labels are dropped by the
  caller before fitting, per task).
- **Pure** — no I/O, no connectors. Deterministic from inputs. Unit-tested directly.

### 2. `pipelines/training_data.py` — assembly seam (L6)

```python
def assemble_training_rows(
    config: DeploymentConfig,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    issue_times: Iterable[datetime],
) -> pd.DataFrame:               # concatenated TRAINING_ROW

def assemble_or_load(
    ..., cache_path: Path,
) -> pd.DataFrame:               # partitioned-Parquet cache wrapper
```

- For each `issue_time`: `build_snapshot(config, issue_time, nwp, observations)` →
  `build_features(snapshot, config)`; collect the per-issue matrices.
- **One batched future read** of the target station:
  `target_source.fetch_historical(start=min(valid_time), end=max(valid_time))` across the
  whole assembled span (not per issue_time) — efficient and the single point where future
  obs enter, training-only.
- `attach_labels` each matrix (or the concatenated matrix) with
  `config.precip_occurrence_threshold_mm`; concat to one `TRAINING_ROW` frame.
- `assemble_or_load` writes/reads a per-deployment **partitioned Parquet** cache so notebook
  re-runs don't re-pull CaSPAr. Cache key includes `deployment_id`, the issue-time range, and
  `SNAPSHOT_SCHEMA_VERSION` (raw values are feature-version-independent per ADR-0012, so the
  derived `FEATURE_SCHEMA_VERSION` is *not* part of the cache key — features are recomputed on
  read). *(Open detail for the plan: whether the cache stores raw snapshots or assembled
  rows; storing snapshots is more ADR-0012-faithful but the notebook only needs rows — decide
  in the plan.)*
- Reused **as-is** by the future `pipelines.training` orchestration.

### 3. Model wrappers (L4)

`TemperatureRegressor`:

```python
def fit(self, rows: pd.DataFrame) -> None        # LightGBM regressor
def predict(self, rows: pd.DataFrame) -> pd.Series  # one temp per row
def save(self, path: Path) -> None
@classmethod
def load(cls, path: Path) -> "TemperatureRegressor"
```

`PrecipOccurrenceClassifier`:

```python
def fit(self, rows: pd.DataFrame) -> None        # LightGBM classifier
def calibrate(self, rows: pd.DataFrame) -> None  # isotonic/Platt (ADR-0004)
def predict(self, rows: pd.DataFrame) -> pd.Series  # calibrated prob in [0, 1] per row
def save(self, path: Path) -> None               # booster + calibrator together
@classmethod
def load(cls, path: Path) -> "PrecipOccurrenceClassifier"
```

- Wrappers select feature columns from the matrix and drop id/time/label columns
  (`feature_schema_version`, `deployment_id`, `issue_time`, `valid_time`, `label_*`).
  `lead_hour` **is** a feature (ADR-0004).
- Each records the `FEATURE_SCHEMA_VERSION` it trained on (ADR-0012) and refuses a mismatched
  matrix at `predict`.
- **`predict` is row-based** (resolves ADR-0012's deferred open item). `predict(snapshot)`
  is **removed** from the contract; the future inference pipeline owns the `build_features`
  call and reshapes per-row predictions back into the `{lead_hour: value}` forecast.

### 4. `evaluation/metrics.py` (L5, fill the stub)

- **Baseline = raw HRDPS at the target** (CONTEXT.md): for temp, the `nwp_temp_c` column
  already in the matrix; for PoP, HRDPS precip → occurrence via the same threshold.
- Temp: MAE, RMSE, and **skill vs baseline**, aggregated **per lead hour**.
- PoP: Brier score, **Brier Skill Score** vs baseline, and **reliability bins** (for the
  diagram), per lead hour.
- Pure functions over a frame of `(prediction, label, baseline, lead_hour)`. **Shared with
  the future publish gate** — the same skill computation that gate will enforce.

### 5. `notebooks/model_dev.ipynb` (thin)

Flow, with no business logic of its own:

1. Load a `DeploymentConfig`; pick an issue-time range; inject CaSPAr-historical `NWPSource`
   + historical `ObservationSource`s.
2. `rows = assemble_or_load(...)` (cached).
3. **Chronological three-way split** `train | calib | test` (see below).
4. Fit `TemperatureRegressor` on `train+calib`. Fit `PrecipOccurrenceClassifier` on `train`,
   then `calibrate` on the disjoint `calib` slice.
5. `predict` both on `test`; compute `evaluation.metrics`.
6. Plot inline (matplotlib): predicted-vs-actual & residuals by lead; MAE/RMSE & skill-vs-
   baseline by lead; PoP **reliability diagram**; LightGBM feature importances.
7. `save` locally-trained models to a **gitignored** dir.

### 6. Evaluation split (making "trained reasonably" verifiable)

- **Chronological three-way split**: earliest `train`, then `calib`, then latest `test` — no
  shuffling (weather is strongly autocorrelated; a shuffled split leaks).
- Temp trains on `train+calib`, judged on `test`.
- PoP trains on `train`, fits its calibrator on the **disjoint `calib`** slice (so calibration
  is not fit on overconfident in-sample predictions), judged on `test`.
- Reported metrics are **per lead hour**, against the raw-HRDPS baseline.

### 7. Notebook mechanics & path-intact guarantee

- `notebooks/model_dev.ipynb`. New `[dependency-groups]` group `notebook = [jupyter,
  ipykernel, matplotlib]`. **LightGBM is a runtime dependency of `models`** → main deps, not
  the notebook group.
- Locally-trained model artifacts save to a gitignored directory (e.g. `notebooks/_artifacts/`).
- **CI smoke** (fast pytest, not nbconvert): a tiny synthetic/fixture dataset →
  `assemble_training_rows` (with **fake injected sources**) → `fit` → `predict` →
  `metrics`, asserting output shapes and finite values. It exercises the **same shared
  functions** the notebook calls, so notebook bitrot surfaces as a failing test without
  executing the `.ipynb`.

## Documentation updates (same PR(s), per CLAUDE.md)

- **New ADR** — "Local notebook is the model-dev surface; training-data assembly is the
  shared seam": records the thin-notebook + tested-shared-assembly stance, the temporal
  three-way (incl. calibration) split, and deferral of the private store and full training
  pipeline.
- **Amend ADR-0004 and ADR-0012** — row-based `predict`; the inference pipeline owns
  `build_features` and reshapes to `{lead_hour: value}`.
- **CONTEXT.md** — add/confirm: **label attachment / labeler**, **training-data assembly**,
  **calibration slice**, **model-dev notebook**.

## Testing strategy

- `attach_labels` — unit tests: threshold boundary (`==` is occurrence), missing-obs → null
  label, temp join correctness.
- `assemble_training_rows` — tests with deterministic **fake** `NWPSource`/`ObservationSource`
  injected (no network): correct row count (`issue_times × lead_hours`), labels joined at the
  right `valid_time`, single batched future read.
- Model wrappers — fit/predict/save/load round-trip on synthetic rows; predict refuses a
  mismatched `FEATURE_SCHEMA_VERSION`.
- `metrics` — known inputs → known MAE/RMSE/Brier/skill values; reliability binning.
- CI smoke — the end-to-end fast path above.
- All of `uv run ruff check`, `ruff format --check`, `lint-imports`, `pyright`, `pytest`.

## Decomposition (for the implementation plan / issues)

Four vertical slices, each independently testable:

1. **Pure labeler + evaluation metrics** — no models, no network; pure functions + tests.
2. **Assembly + Parquet cache** — orchestrator over `build_snapshot`/`build_features` +
   labeler, with fake sources in tests.
3. **Model wrappers + row-based `predict`** — LightGBM temp & PoP + calibration; ADR-0004/
   0012 amendments + the new ADR.
4. **Notebook + `notebook` dep group + CI smoke** — ties the path together end to end.

## Open items deferred to the plan

- Whether `assemble_or_load` caches **raw snapshots** (ADR-0012-faithful) or **assembled
  rows** (simpler; what the notebook needs). Lean: rows for v1, revisit if the production
  pipeline needs raw.
- Exact LightGBM hyperparameters and early-stopping — sensible defaults in v1; tuning is a
  later concern.
- **Schema-version column reconciliation.** `FEATURE_ROW` carries `feature_schema_version`
  but `TRAINING_ROW` declares `schema_version`. `attach_labels` produces `TRAINING_ROW`, so
  the plan must settle what `schema_version` means there (the snapshot version, the feature
  version, or both columns retained) and align `attach_labels`'s output with the contract —
  surfacing it to the contracts owner rather than silently coercing.
