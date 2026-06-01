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
Canada observations, HRDPS from Datamart and CaSPAr), `features.build_snapshot` — the
single as-of feature path shared by training and inference (ADR-0011) — and
`features.build_features` — the read-time transform from a `FeatureSnapshot` to the feature
matrix (derived features + explode-to-per-lead-hour rows, ADR-0012).

The live MSC Datamart HRDPS connector (`hrdps_datamart`) is verified against real GRIB2
(run 2026-05-31 18Z): date-partitioned URL layout, canonical MSC variable codes, sole-data-var
decode. `nwp_core` solar is de-accumulated from J/m² accumulated to mean W/m² per hour
(ADR-0014), shared by both HRDPS connectors.

Additionally implemented: `features.attach_labels` (the pure label-attachment step →
labeled feature matrix), `pipelines.training_data` (training-data assembly + local Parquet
cache + chronological split — the shared seam, ADR-0013), the two **LightGBM model
wrappers** (`models.TemperatureRegressor` and `models.PrecipOccurrenceClassifier` with
isotonic calibration, row-based `predict`), `evaluation.metrics` (per-lead skill vs the
raw-HRDPS baseline + PoP reliability), a thin local **model-dev notebook**
(`notebooks/model_dev.py`), the **`training_store`** — the per-deployment, path-based,
partitioned-Parquet store (raw `FeatureSnapshot` blobs + a separate labels table) the logger
appends to and training reads (ADR-0015), the **raw-HRDPS baseline forecaster**
(`models.baseline`) — temperature passthrough + threshold PoP, the initial published champion
(ADR-0016), `publication.write_forecast` — atomic forecast-JSON writer, and
`pipelines.inference.run_inference` — builds a snapshot, produces the baseline forecast,
writes the `ForecastDocument` JSON, and appends the snapshot to the training store
(in-process, ADR-0016) — and the **hourly inference GitHub Action**
(`.github/workflows/inference.yml`) that runs it per deployment and **persists the snapshots
to the public `training-data` branch** via `GITHUB_TOKEN` (ADR-0017) — so data collection is
live with no external setup.

**Not yet implemented** (currently stubs): the registry/champion-loading (trained model
promotion via champion/challenger, ADR-0006), the publish gate, the training pipeline
orchestration CLI, and the **gh-pages forecast-JSON publish** (the public live-service surface
— the Action collects data but does not yet publish the forecast). CaSPAr appears unavailable,
so v1 pivots to logger-forward accumulation (cold-start, ADR-0008) rather than a historical
seed; the raw store is **public** (`training-data` branch, ADR-0017 amending ADR-0009).

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
