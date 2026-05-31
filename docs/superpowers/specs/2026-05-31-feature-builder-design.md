# Feature builder — `FeatureSnapshot` → feature matrix

- **Date:** 2026-05-31
- **Status:** Design approved (brainstorming); ready for implementation plan
- **Upstream:** `features.build_snapshot` (ADR-0011), `contracts.snapshot.FeatureSnapshot`
- **Downstream:** `TemperatureRegressor` / `PrecipOccurrenceClassifier` (ADR-0004), training/inference pipelines, `contracts.training_store.TRAINING_ROW`

## Purpose

ADR-0011 closes with an explicit follow-on work item:

> A downstream "build features from the snapshot" step is now required (separate work
> item) to produce the per-lead-hour model-input rows and derived features.

This spec defines that step: a single pure function that turns one `FeatureSnapshot`
(raw, canonicalized values for one issue time spanning all lead hours) into the
**feature matrix** — long-format model-input rows, one per `(issue_time, lead_hour)`,
carrying the derived features named in `CONTEXT.md` (dewpoint depression, pressure
tendency, advection, per-lead-hour encodings).

**Scope (A):** the pure `snapshot → feature matrix` transform only. **No labels**, **no
training store, no pipeline wiring, no model changes.** Label attachment (joining
`label_temp_c` / `label_precip_occurrence` onto rows to form `TRAINING_ROW`) is a separate
downstream concern, noted but not specified here.

## Key decisions (to be recorded in a new ADR)

### Decision 1 — store-raw, read-time transform

The transform runs at **training-read time and at inference time** — never at write time.
The training store persists **raw snapshot values** (+ labels); derived features are
recomputed on read. This is the only option consistent with ADR-0011's "the snapshot is
the boundary; derived features are downstream pure functions." Consequences:

- One shared feature code path for training and inference (the same reason `build_snapshot`
  exists) — train/serve skew is eliminated by construction at this layer too.
- Feature iteration is cheap: adding or changing a derived feature means *retraining only* —
  no re-logging and no store backfill.
- The store is feature-version-independent and smaller (no 48× duplication of the as-of-`t0`
  obs/static/temporal columns, no materialized derived columns).
- This forces the transform to be **deterministic and self-contained**: no fitted state, no
  read-time input beyond `(snapshot, config)`.

### Decision 2 — feature-set versioning

A `FEATURE_SCHEMA_VERSION` constant, **distinct from** `SNAPSHOT_SCHEMA_VERSION`. The raw
snapshot contract and the derived-feature set version independently: the snapshot can be
stable while the derived-feature set evolves. Models record the feature version they were
trained on, so a champion built against a stale feature set is **refused rather than
silently misread** — this matters for champion/challenger (ADR-0006). The transform also
asserts the incoming `snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION` and raises on
mismatch.

### Decision 3 — no statistical scaling

The transform performs **no standardization/normalization** of feature values. LightGBM is
tree-based and gains nothing from scaling. The "normalization" in ADR-0011 means
*canonicalization* (units, variable order), already done by `build_snapshot`; this step
adds derived columns and explodes to rows, nothing more.

## Placement, naming, signature

New pure function in the **`features`** layer — below `models` in the import contract, so
both models and pipelines may call it:

```python
# src/microclimate/features/feature_builder.py

def build_features(snapshot: FeatureSnapshot, config: DeploymentConfig) -> pd.DataFrame:
    """Explode one FeatureSnapshot into the per-(issue_time, lead_hour) feature matrix.

    Pure, deterministic, no fitted state, no network. The single shared transform used by
    both training (over stored raw snapshots) and inference. No labels are attached.
    """
```

- Returns the **feature matrix**: long format, one row per `(issue_time, lead_hour)`, no
  labels.
- Deterministic and self-contained: a pure function of `(snapshot, config)` — no I/O.

### New domain language (`CONTEXT.md`, same PR)

- **Feature matrix** — the long-format, per-`(issue_time, lead_hour)` model-input rows; a
  downstream pure function of the feature snapshot produced by `features.build_features`.
  No labels (those are attached separately).
- **Derived feature** — a feature computed from raw snapshot values (dewpoint depression,
  pressure tendency, advection, per-lead-hour encodings), as opposed to a passthrough of a
  raw snapshot value.
- **Feature schema version** — the version of the derived-feature set
  (`FEATURE_SCHEMA_VERSION`), distinct from the raw-snapshot contract version.

## Output contract & versioning

A new Pandera schema `FEATURE_ROW` in a new module `contracts/feature_matrix.py` (kept
separate from `training_store.py` so the label-free feature contract and the
label-carrying store contract version independently):

- **Identity columns:** `deployment_id` (str), `issue_time` (`datetime64[ns, UTC]`),
  `lead_hour` (int, 1–48), `valid_time` (`datetime64[ns, UTC]`, = `issue_time + lead_hour`).
- **Feature columns:** `strict=False` — the set varies by deployment (neighbor count). This
  is `TRAINING_ROW` minus the label columns; label attachment later joins on
  `(issue_time, lead_hour)`.
- `coerce=True`.

**Column-set determinism.** The full column set and a canonical column order are derived
from `config` (neighbors × physical vars × lags × leads, fixed orders). For a given
deployment, a training-`t0` snapshot and an inference-`t0` snapshot therefore produce
**byte-identical columns** — a guarded train/serve parity invariant.

## Feature catalog

Conventions: `h` = the row's own lead hour; `T` = target station; `Ni` = neighbor `i`;
physical vars in the fixed order from `build_snapshot._PHYSICAL_VARS`. As-of-`t0` features
(obs, static, `t0` temporal) are **broadcast** identically across all lead-hour rows of a
given `t0`.

### NWP (own-lead; `_h{lead}` suffix dropped — `lead_hour` is now a column)

| Column | Definition |
| --- | --- |
| `nwp_temp_c` … `nwp_wind_dir_deg` (8) | passthrough of `nwp_{var}_h{h}` |
| `nwp_dpd` | `nwp_temp_c − nwp_dewpoint_c` (dewpoint depression) |
| `nwp_ptend_3h` | `nwp_pressure_h − nwp_pressure_{h−3}`; **NaN for `lead < 3`** (structural, not missingness) |

NWP is complete-or-fail upstream, so NWP columns are never missingness-NaN; the only NWP
NaN is the structural `ptend` one at `lead < 3`.

### Observations (passthrough + derived; broadcast)

For each station `S ∈ {T, N1…Nm}`, each var, each lag `k ∈ 0…lag_hours`:

| Column | Definition |
| --- | --- |
| `obs_{S}_{var}_lag{k}` | passthrough value (NaN when absent) |
| `obs_{S}_{var}_lag{k}_mask` | passthrough presence mask (bool) — masks pair with LightGBM native NaN handling (ADR-0004) |
| `obs_{S}_dpd_lag{k}` | `temp − dewpoint` at lag `k`; NaN when either input absent |

Target-only tendencies (single as-of-`t0` value, broadcast):

| Column | Definition |
| --- | --- |
| `obs_{T}_ptend_3h` | `pressure_lag0 − pressure_lag3`; NaN if either endpoint missing |
| `obs_{T}_dpd_tend_3h` | `dpd_lag0 − dpd_lag3`; NaN if either endpoint missing (literature: *changes* in DPD carry the heaviest PoP weight) |

### Advection (per neighbor `Ni`)

| Column | Definition |
| --- | --- |
| `adv_{Ni}_temp_grad_lag0` | `obs_{Ni}_temp_c_lag0 − obs_{T}_temp_c_lag0` |
| `adv_{Ni}_dpd_grad_lag0` | `obs_{Ni}_dpd_lag0 − obs_{T}_dpd_lag0` |
| `adv_{Ni}_precip_grad_lag0` | `obs_{Ni}_precip_mm_lag0 − obs_{T}_precip_mm_lag0` |
| `adv_{Ni}_upwind_align` | `cos(bearing(T→Ni) − wind_from_dir_{T}@lag0) × wind_speed_{T}@lag0` |

- `bearing(T→Ni)` is the great-circle initial bearing from target to neighbor, computed from
  **config** coordinates (the snapshot's static block is target-only; neighbor lat/lon live
  in `config.neighbors`). Config is deployment-static, so this is deterministic and
  skew-free.
- `wind_from_dir` uses the meteorological convention (direction wind blows *from*): a
  neighbor lying in the from-direction is upwind, so its air is advecting toward the target →
  positive alignment.
- Any gradient/alignment input absent → that column is NaN.
- Gradients are computed at `lag0` only; the raw neighbor lags (passthrough above) already
  carry the temporal depth the literature calls for.

### Static (passthrough, broadcast)

`static_lat`, `static_lon`, `static_elevation_m` (target only; `static_elevation_m` may be
NaN when unknown).

### Temporal

| Column | Definition |
| --- | --- |
| `t0_hour_sin`, `t0_hour_cos`, `t0_doy_sin`, `t0_doy_cos` | passthrough of snapshot `t0` cyclical encodings (broadcast) |
| `valid_hour_sin`, `valid_hour_cos` | cyclical encoding of `valid_time` hour-of-day (per-lead; dominant diurnal signal) |
| `lead_hour` | the row's lead hour as an integer feature (ADR-0004); doubles as identity column |

A per-lead day-of-year encoding is intentionally omitted (barely moves across a 48 h
horizon; `t0_doy_*` already covers seasonality).

## Error handling & edge cases

- **Schema mismatch:** `snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION` → raise.
- **Missing observations:** passthrough value NaN + `mask=False`; derived obs/advection
  features → NaN when any input absent. The model degrades gracefully (LightGBM native NaN
  handling) rather than crashing.
- **Structural NaN:** `nwp_ptend_3h` at `lead < 3`; tendencies when an endpoint lag is
  absent.
- **Empty neighbor list:** no `adv_*` and no neighbor `obs_*` columns; still deployment-stable.
- **Off-hour `t0` / NWP-only snapshot:** inherited from `build_snapshot` semantics — absent
  obs slots are already NaN+mask-False in the snapshot, so the transform simply propagates.

## Integration note (not redesigned here)

`TemperatureRegressor.predict(snapshot)` / `PrecipOccurrenceClassifier.predict(snapshot)`
cannot build rows without `config`. Recommendation for the **models** spec: have the
**pipeline own the `build_features` call** and pass rows to both `fit(rows)` and a
`predict(rows)`, making training and inference symmetric. Flagged as a consequence; not
solved in this spec.

## Testing

- **Golden fixtures:** hand-constructed `FeatureSnapshot` → expected feature matrix
  (values + column set).
- **Property tests:** column-set determinism from config; mask propagation; `nwp_ptend_3h`
  NaN at `lead < 3`; tendency NaN on missing endpoints; advection sign and upwind-alignment
  correctness (a neighbor directly upwind with steady wind yields positive alignment).
- **Train/serve column-parity test:** a training-`t0` snapshot and an inference-`t0`
  snapshot for the same config produce identical columns in identical order.
- **Architecture/purity test:** the module imports nothing from `connectors` and performs no
  I/O (reinforces the read-time, side-effect-free contract); complements the existing
  import-linter layer contract.

## Deliverables

1. `src/microclimate/features/feature_builder.py` — `build_features`.
2. `src/microclimate/contracts/feature_matrix.py` — `FEATURE_ROW` schema +
   `FEATURE_SCHEMA_VERSION`.
3. `CONTEXT.md` — *feature matrix*, *derived feature*, *feature schema version* terms.
4. New ADR — Decisions 1–3 above (read-time/store-raw, feature-set versioning, no scaling).
5. Tests as in the Testing section.
6. README "Project status" update.
