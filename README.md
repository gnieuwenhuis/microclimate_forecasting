# Microclimate Forecasting

Free, zero-maintenance hourly **temperature** and **probability-of-precipitation** forecasts
for a local station, by downscaling Environment Canada's HRDPS. Designed around Lethbridge,
Alberta; deployable for any microclimate by config.

- **Domain glossary (source of truth for terms):** [CONTEXT.md](CONTEXT.md)
- **Architectural decisions (source of truth):** [docs/adr/](docs/adr/)
- **Data licenses & attribution:** [DATA_LICENSES.md](DATA_LICENSES.md)
- **Scaffolding spec:** [docs/superpowers/specs/2026-05-30-scaffolding-spec.md](docs/superpowers/specs/2026-05-30-scaffolding-spec.md)

`docs/adr/` records *why* the architecture is the way it is, and `CONTEXT.md` defines the
vocabulary the whole project uses. Both are kept current as the project evolves — a change
that alters a decision or introduces a concept updates them in the same PR.

The architecture is enforced mechanically: typed boundaries (Pydantic/Pandera), connector
ABCs, a single feature-snapshot builder, source-eligibility validation, and an
`import-linter` layer contract — all gated in CI.

## Project status

Pre-1.0, built bottom-up. The **data backbone is implemented**: typed contracts, validated
deployment config, the connector framework with live + historical sources (Environment
Canada observations, HRDPS via **Open-Meteo** — live `/v1/forecast` for inference and the
Historical Forecast API for the seed backfill), `features.build_snapshot` — the single as-of
feature path shared by training and inference (ADR-0011) — and `features.build_features` —
the read-time transform from a `FeatureSnapshot` to the feature matrix (derived features +
explode-to-per-lead-hour rows, ADR-0012).

**HRDPS source pivot (ADR-0019):** HRDPS is now sourced exclusively via the pure-HTTP
Open-Meteo connector (`src/microclimate/connectors/sources/openmeteo.py`). The native GRIB2
stack (`nwp_core`, `hrdps_datamart`, `hrdps_caspar`) and the `xarray`/`cfgrib` dependencies
have been removed. Live inference calls `/v1/forecast`; the retrain-time seed backfill
(`pipelines/backfill.py`) calls the Historical Forecast API to populate the public
`training-data` branch. There is an accepted **lead-time skew** (ADR-0019 §1b): the seed
data is short-lead-stitched while live inference serves full leads; the publish gate acts as
the fail-safe.

Additionally implemented: `features.attach_labels` (the pure label-attachment step →
labeled feature matrix), `pipelines.training_data` (training-data assembly + local Parquet
cache + chronological split — the shared seam, ADR-0013), the two **LightGBM model
wrappers** (`models.TemperatureRegressor` and `models.PrecipOccurrenceClassifier` with
isotonic calibration, row-based `predict`), `evaluation.metrics` (per-lead skill vs the
raw-HRDPS baseline + PoP reliability), a thin local **model-dev notebook**
(`notebooks/model_dev.py`), the **`training_store`** — the per-deployment, path-based,
partitioned-Parquet store (raw `FeatureSnapshot` blobs + a separate labels table, ADR-0015),
the **raw-HRDPS baseline forecaster** (`models.baseline`) — temperature passthrough +
threshold PoP, the initial published champion (ADR-0016), `publication.write_forecast` —
atomic forecast-JSON writer, `pipelines.inference.run_inference` — **stateless
publish-only**: builds a snapshot, produces the baseline forecast, and writes the
`ForecastDocument` JSON (the inference logger was removed; the training store is populated
at retrain time via the seed backfill, not during inference), and the **hourly inference
GitHub Action** (`.github/workflows/inference.yml`) that runs it per deployment
(ADR-0017).

**Training pipeline now implemented** (`pipelines/training.py`): monthly retrain runs
seed backfill → train → champion/challenger publish gate (`evaluation.publish_gate`,
ADR-0006) → on promotion, publishes the champion model as a GitHub Release asset and
`registry.json` to gh-pages — with the training store persisted on the public
`training-data` branch (ADR-0017/0018). `evaluation.publish_gate` and
`publication.registry_store` are no longer stubs.

**Remaining gap:** inference still publishes the **baseline** (`{"temp": "baseline", "pop":
"baseline"}`); swapping the inference pipeline to load the registry/champion is the next
slice. The **gh-pages forecast-JSON publish** (the public live-service surface) is also not
yet wired. `acis` is retained but unused (ADR-0010).

## Develop

```bash
uv sync                      # install deps + dev group
uv run ruff check . && uv run ruff format --check .
uv run lint-imports          # enforce the layered architecture
uv run pyright               # strict type check
uv run pytest                # network-marked tests are deselected by default
```

These mirror the CI gate (`.github/workflows/ci.yml`). The package lives in
`src/microclimate`, layered low→high (`contracts` → `config` → `connectors` → `features` →
`evaluation`/`models` → `publication` → `pipelines`); the layer order is enforced by
`import-linter`. Deployment configs live in `config/deployments/`; the thin-client dashboard
in `dashboard/`.
