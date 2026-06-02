# CONTEXT — Microclimate Forecasting

The single-context domain glossary for this repo. When code, issues, ADRs, tests, or
plans name a concept below, use the term **exactly as defined here** — do not drift to
synonyms. If a concept you need isn't here, that's a signal: either you're inventing
language the project doesn't use (reconsider), or there's a real gap (resolve it via
`/grill-with-docs` and add it).

## What this project is

A **free-to-deploy, zero-maintenance system that downscales a public numerical weather
forecast to a single local station**, producing an hourly **temperature** and
**probability-of-precipitation** forecast for the next 48 hours. Designed around
Lethbridge, Alberta, but **deployable for any microclimate** by configuration. The
system trains on free data, runs entirely on free infrastructure (GitHub Actions +
Pages + Releases), and serves thin clients that only read a published JSON file.

The system **does not forecast from scratch** and **does not try to out-forecast a
weather agency**. It corrects an existing official forecast for local bias.

## Glossary

### Core task

- **Downscaling** (a.k.a. **post-processing**) — the project's core task: take an
  existing official forecast and *correct it* for the systematic local bias at a
  specific location. The official forecast carries the physics; this system learns the
  local correction. This is **not** forecasting future weather purely from station
  history.
- **Temperature** — one of the two predicted quantities. A continuous regression target,
  in °C.
- **PoP** (Probability of Precipitation) — the second predicted quantity. The probability
  that **precipitation occurs** in a given hour, as a calibrated value in `[0, 1]`. The
  system predicts **occurrence only**, never amount. "Occurrence" is decided by a
  configured precipitation threshold (see **Precip-occurrence label**).

### Data backbone

- **NWP** (Numerical Weather Prediction) — physics-based forecast model output. The
  system's primary input feature.
- **HRDPS** (High Resolution Deterministic Prediction System) — the NWP backbone:
  Environment Canada's ~2.5 km model, 4 runs/day, hourly lead times 1–48 h. The single
  source of the forecast being downscaled. **Sourced via Open-Meteo** (a reprocessed
  mirror — elevation-aware grid-cell selection, `cell_selection=land`), not raw GRIB2; the
  *same* Open-Meteo product feeds both training and inference (ADR-0019).
- **Open-Meteo** — the free public API that is the **single source of HRDPS**, under
  CC-BY-4.0. Two endpoints behind one connector: **`/v1/forecast`** (live, full 1–48 leads) for
  the *inference pipeline*, and the **Historical Forecast API** (deep archive from ~2024) for the
  *seed backfill*. The deep archive is a **stitched short-lead series**, not full per-run leads, so
  the seed and live feeds differ in lead-time provenance — see **Lead-time skew**. Replaces both
  the dead CaSPAr archive and the native MSC GRIB2 channels (GeoMet/Datamart) (ADR-0019, superseding
  ADR-0007 and ADR-0014).
- **Lead-time skew** *(accepted v1 limitation — ADR-0019 §1b)* — the model **trains on short-lead**
  HRDPS (the deep stitched seed) but is **served full-lead** HRDPS (`/v1/forecast`), a difference
  that grows with lead hour. Accepted for v1 because the local-bias correction a downscaler learns
  is largely lead-stable and the **publish gate fails safe**. True-parity forward capture is a
  deferred fast-follow.
- **CaSPAr** *(retired — see ADR-0019)* — formerly the historical-HRDPS seed (Canadian
  Surface Prediction Archive, from 2017-05-22). **Dead since ~mid-2025** (site offline,
  unmaintained, no successor); replaced by the Open-Meteo Historical Forecast API backfill. Term
  kept only so older ADRs read coherently.
- **Observation** / **obs** — an actual measured reading from a weather station
  (temperature, precipitation, etc.) at a past or present time. Distinct from a forecast.

### Geography

- **Deployment** — one fully-specified instance of the system for one place. Defined by a
  single validated config file (`config/deployments/<id>.yml`). Identified by a
  **`deployment_id`** (e.g. `lethbridge`). Every artifact, model, and output is namespaced
  by `deployment_id`. The system is **multi-deployment by design**. v1 ships **one**
  deployment: `lethbridge` (`seeded`), targeting ECCC Lethbridge CDA (#2265) — retargeted
  from ACIS Demo Farm once ACIS proved to have no ungated live-hourly feed (ADR-0010).
- **Training strategy** — a per-deployment mode. v1 uses only **`seeded`**: all observation
  sources have *deep* historical coverage and training uses the **Open-Meteo Historical Forecast
  API *seed backfill*** (deep stitched short-lead, re-pulled at each retrain, ~2024 onward; see
  *lead-time skew*), trainable from day
  one (ADR-0019). A **`cold_start`** mode (live-only target, not trainable until forward data
  accumulates) is *designed but deferred* — see *cold start*.
- **Cold start** *(deferred — not in v1; needs redesign post-ADR-0019)* — the strategy for a
  target with no free deep history (e.g. a CWOP PWS): labels accumulate forward over time.
  Documented in ADR-0008 as the path for predicting *at Henderson Lake* once a station becomes
  reachable there. Its original forward-accumulation mechanism was the now-removed logger
  (ADR-0019); the cold-start path needs a new forward-capture design when revisited.
- **Target** / **target station** — the single station a deployment predicts *for*. Its
  observations are the training **labels**, and (in v1) also an input feature.
- **Neighbor** / **neighbor station** — a nearby station whose recent observations are
  input features for the target (capturing weather advecting toward the target). The
  neighbor list is per-deployment configuration.
- **Microclimate** — the local climate at the target, distinct from the regional forecast.
  The thing the system exists to capture.

### Data contract

- **Feature snapshot** — the single canonical input object for one prediction, built by
  one builder used by **both** training and inference (this is what prevents train/serve
  skew). For a given **issue time** `t₀` and target, it contains: HRDPS forecast features
  (lead 1–48 h at the target cell), observation features (recent obs from target +
  neighbors, lag-windowed, each with a **missingness mask**), static features (lat/lon,
  elevation), and temporal features (cyclical encodings of `t₀` and lead hour).
  The snapshot is built by `features.build_snapshot`, which is the **normalization / as-of
  boundary**: it stores *raw canonicalized values only* — one object per **issue time** spanning
  all lead hours — and never `fetch_live` (only as-of `fetch_historical` bounded to `t0`), which
  is the train/serve skew guarantee (ADR-0011). Derived features (dewpoint depression, pressure
  tendency, advection, per-lead-hour encodings) and the explode-to-per-lead-hour rows are
  *downstream* pure functions of the snapshot. **Feature-key conventions:** NWP →
  `nwp_{var}_h{lead}` (8 variables × leads `1…horizon_hours`, target cell only); observations →
  `obs_{station_id}_{var}_lag{k}` on a fixed hourly **lag grid** `lag0`(=`t0`) … `lag{lag_hours}`
  (absent slot → `NaN`, mask `False`); static → `static_lat`/`static_lon`/`static_elevation_m`
  (target only); temporal → `t0_hour_sin`/`t0_hour_cos`/`t0_doy_sin`/`t0_doy_cos`.
- **Feature matrix** — the long-format, per-`(issue_time, lead_hour)` model-input rows
  produced by `features.build_features` from a **feature snapshot**. One row per lead hour;
  carries **derived features** plus the as-of-`t0` snapshot values broadcast across rows;
  **label-free** (labels are attached downstream). Built at training-read time and at
  inference by the **same** function, so its column set is identical for train and serve.
- **Labeled feature matrix** — the feature matrix with `label_temp_c` and
  `label_precip_occurrence` attached (`features.attach_labels`). What the models train on;
  distinct from the persisted **Training store** schema (raw snapshot + labels, ADR-0012).
- **Label attachment** / **labeler** — the pure step joining target-station observations at
  `valid_time` onto the feature matrix to form the labeled feature matrix. The future
  (post-`issue_time`) read it depends on is done by training-data assembly, never inference.
- **Derived feature** — a feature computed from raw snapshot values (dewpoint depression,
  pressure tendency, advection, per-lead-hour `valid_hour` encoding), as distinct from a
  passthrough of a raw snapshot value. Derived features are pure functions of the snapshot
  (ADR-0011, ADR-0012).
- **Feature schema version** — `FEATURE_SCHEMA_VERSION`, the version of the **derived
  feature** set, distinct from `SNAPSHOT_SCHEMA_VERSION` (the raw-snapshot contract). A model
  records the feature version it trained on so a stale-feature champion is refused.
- **Issue time** (`t₀`) — the reference time a feature snapshot is built at. The forecast
  predicts `t₀+1 … t₀+48`.
- **As-of reconstruction** — the invariant that a feature snapshot only ever uses
  observations available **at or before `t₀`**. Future data cannot enter, by construction.
  Guarantees no label leakage.
- **Missingness mask** — a per-observation-feature flag marking whether the value was
  present or imputed. Lets a down feed degrade accuracy gracefully instead of crashing.
- **Dual-feed source** — an observation source that provides **both** an hourly
  historical feed (for training) **and** an hourly live feed (for inference), for the same
  physical measurement. **Eligibility requires every observation source to have `deep`
  historical coverage** (see *historical coverage*); daily-only sources (CoCoRaHS) and
  live-only sources (CWOP) are ineligible (ADR-0008).
- **Historical coverage** — a declared, validated capability of each `ObservationSource`:
  `deep` (multi-year, e.g. Environment Canada), `shallow`, or `none` (live-only).
  Depth is explicit and machine-checked, not assumed; only `deep` sources are eligible.
- **Connector** — the abstraction over a data source. An `NWPSource` or an
  `ObservationSource`; observation connectors implement both historical and live fetch (the
  dual-feed contract) and declare their *historical coverage*.
- **Available sources** — the free, no-device source this project uses for v1 observations is
  **Environment Canada** (bulk hourly CSV, deep) — both target and neighbors. **ACIS** was
  evaluated and **dropped**: its only ungated feed is *daily*, and no ungated *live-hourly*
  feed exists for its stations (spike #3 / ADR-0010); the connector is retained but unused.
  Consumer-PWS network APIs (Weather Underground, Tempest, Ambient) are device-gated and
  **not used**; CWOP is live-only and therefore ineligible for the (seeded) v1 (ADR-0008).
- **Precip-occurrence label** — the binary training label for PoP: `1` if observed
  precipitation in the hour meets/exceeds the configured threshold, else `0`.

### Modeling & quality

- **Temp model** — a LightGBM **regressor** predicting temperature. One per deployment.
- **PoP model** — a LightGBM **classifier** predicting precip-occurrence, with an explicit
  **calibration** stage. One per deployment.
- **`lead_hour`** — a model input feature (1–48). A single model spans all lead hours
  rather than one model per hour.
- **Calibration** — the property (and the post-training stage that enforces it) that a
  PoP value means what it says: "30%" should rain ~30% of the time. The PoP deliverable.
- **Calibration slice** — the disjoint chronological slice between train and test on which
  the PoP isotonic calibrator is fit, so calibration is not fit on overconfident in-sample
  predictions.
- **Baseline** / **raw HRDPS** — the unmodified HRDPS forecast at the target. The floor a
  model must beat to justify the project's existence.
- **Baseline forecaster** — the raw-HRDPS forecaster published before a trained model exists
  (ADR-0016): temperature passthrough + threshold PoP. The initial champion and the floor a
  trained model must beat; `model_versions` marks it `"baseline"`.
- **Skill score** — a metric expressing improvement over a baseline (e.g. MAE skill for
  temp, **Brier Skill Score** for PoP), reported **per lead hour**.
- **Champion / challenger** — the model-promotion model: a freshly trained model
  (challenger) is **published only if** it beats both raw HRDPS and the current published
  model (champion) on the temporal holdout. Temp and PoP are promoted **independently**.
- **Publish gate** — the step in the training pipeline that enforces champion/challenger.
  Refuses to publish a model that fails. Makes shipping a worse model impossible.

### Infrastructure

- **Inference pipeline** — the hourly job that builds a feature snapshot from live data,
  runs the champion models, and publishes the **forecast JSON**. It is **stateless** — it no
  longer logs snapshots (the logger was removed in ADR-0019; training data comes from the
  seed backfill).
- **Seed backfill** — the **retrain-time** step that pulls the deep HRDPS history from the
  Open-Meteo **Historical Forecast API** (+ as-of ECCC obs) and assembles labeled rows into the
  training store. **Idempotent and additive** — coalesces by `issue_time`×`lead_hour` and **never
  prunes** existing rows, so the store survives Open-Meteo pruning or outage and accumulates
  monotonically (ADR-0019). This additivity is a durability guarantee — do not add a step that
  deletes rows absent from a backfill.
- **Training pipeline** — the job that, at each retrain, runs the **seed backfill**, reads the
  **training store**, trains temp and PoP models, evaluates them, and runs the publish gate.
- **Training store** — the accumulating per-deployment dataset populated by the **seed
  backfill**: raw **snapshots** (each `FeatureSnapshot` serialized as a blob +
  `SNAPSHOT_SCHEMA_VERSION`) plus a separate **labels** table (per `issue_time`×`lead_hour`).
  One **`data.parquet` per deployment-month** (coalesced **read-modify-write**, write-time
  dedupe; latest `written_at` wins), path-based; persisted to a **public `training-data`
  branch** (ADR-0017, ADR-0018). The store is raw-only — `TRAINING_ROW` is the read-time join
  (snapshot → `build_features` → labels).
- **Training-data assembly** — `pipelines.training_data`: iterates issue-times through
  `build_snapshot` → `build_features`, performs the single training-only future read of
  target observations, and labels the result. The shared seam used by the model-dev notebook
  and (later) the training pipeline; caches assembled rows to local Parquet.
- **Model-dev notebook** — the thin, local-only `notebooks/model_dev.py` for training and
  exploring models. Holds no logic; calls the shared assembly, model, and metric functions.
- **Forecast JSON** — the single published, schema-versioned output document a deployment
  produces. The only thing thin clients read. Carries an **`attribution`** field (data-source
  acknowledgments) and never embeds raw observations — only derived predictions (ADR-0009).
- **Thin client** — a consumer that *only* reads the forecast JSON (never touches HRDPS,
  station feeds, or models). The dashboard and the future Android app are thin clients.
- **Dashboard** — the v1 thin client: static files served from GitHub Pages, reading the
  forecast JSON from the same origin. Lives in this repo, outside the Python import graph.
- **The four homes** — where artifacts live: forecast JSON + dashboard + `registry.json`
  on `gh-pages` (public); model binaries as versioned GitHub **Release** assets (public);
  the raw **training store** on a **public `training-data` branch** (ADR-0017, ADR-0018; the
  ADR-0009 "private repo" stance is superseded — its raw data is now CC-BY-4.0 Open-Meteo
  HRDPS + ECCC obs, redistributable with attribution); source/configs/docs in the main branch
  (public). All four homes are public; the store carries its own attribution notice (ADR-0019).
- **Derived product** — a transformation of the source data (the forecast JSON, the trained
  models). Redistributable under all source licences *with attribution*. Note: with the
  Open-Meteo pivot (ADR-0019), the **raw training store is also publicly redistributed** — its
  data is CC-BY-4.0/ECCC, so it ships publicly *with attribution* rather than being withheld.
- **`registry.json`** — the manifest naming the current champion version per
  `(deployment_id, task)`. Updated by the publish gate; read by the inference pipeline.

## Conventions baked into the system

- **UTC everywhere.** All timestamps are UTC; local time is a display concern only.
- **`schema_version`** is carried on both the forecast JSON and the training store, so
  clients and pipelines can refuse incompatible payloads instead of misreading them.
- **One Open-Meteo request spec.** Training (Historical Forecast API) and inference
  (`/v1/forecast`) must issue **identical** Open-Meteo request parameters — coordinates, model,
  variable set, units, and `cell_selection=land` — so both resolve to the same grid cell and
  unit/variable contract. Enforced by a shared request-spec fitness function, not convention
  (ADR-0019). This pins spatial/variable parity but **not lead-time provenance** — the deep seed
  is short-lead-stitched while live serves full leads (see *lead-time skew*), the one accepted
  v1 divergence. Replaces the old "one HRDPS spec" (which spanned two native feeds).
- **Attribution is mandatory.** Every public, data-bearing artifact carries source
  attribution: **Open-Meteo** (`Weather data by Open-Meteo.com`, CC-BY-4.0, changes
  indicated) and **ECCC** (`Data Source: Environment and Climate Change Canada` — the HRDPS
  model and station observations). The CaSPAr / Mai et al. 2020 citation is **dropped**
  (CaSPAr retired, ADR-0019); ACIS attribution remains not-required for v1 (ADR-0010, licence
  section kept in `DATA_LICENSES.md` for the deferred path). Attribution is enumerated in
  `DATA_LICENSES.md`, embedded in the forecast JSON `attribution` field, shown in the dashboard
  footer, **and carried on the public `training-data` branch** (which redistributes the raw
  CC-BY-4.0 data) — the last enforced by a CI check (ADR-0019).
- **Guardrails over discipline.** Architectural rules are enforced mechanically (types,
  schemas, CI fitness functions), not by convention — see the scaffolding spec in
  `docs/superpowers/specs/`. Bad structural decisions should fail at construction or in CI.
