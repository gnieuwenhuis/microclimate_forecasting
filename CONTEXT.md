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
  source of the forecast being downscaled. Sampled at the target's grid cell.
- **GeoMet** / **Datamart** — Environment Canada's free **live** HRDPS channels, used by
  the **inference pipeline**.
- **CaSPAr** (Canadian Surface Prediction Archive) — the free research archive of
  historical HRDPS (from 2017-05-22). A queued bulk-request archive, **not** an API.
  Used **once**, as the **historical seed** for the training store.
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
  sources have *deep* historical coverage and training uses the CaSPAr *historical seed* +
  the logger (trainable from day one). A **`cold_start`** mode (live-only target, logger as
  the sole label source, not trainable until logged data accumulates) is *designed but
  deferred* — see *cold start*.
- **Cold start** *(deferred — not in v1)* — the strategy for a target with no free deep
  history (e.g. a CWOP PWS): labels accumulate forward via the logger. Documented in
  ADR-0008 as the path for predicting *at Henderson Lake* once a station becomes reachable
  there; not implemented in v1 because no free live station exists at Henderson today.
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
  runs the champion models, publishes the **forecast JSON**, and **logs the snapshot**.
- **Logger** — the role of the inference pipeline as a self-accumulating data source:
  every snapshot it builds is persisted so that, once its valid times pass and obs land, a
  fully labeled training row exists. Decouples ongoing training from CaSPAr.
- **Training pipeline** — the job that reads the **training store**, trains temp and PoP
  models, evaluates them, and runs the publish gate.
- **Training store** — the accumulating per-deployment dataset behind the logger: raw
  **snapshots** (each `FeatureSnapshot` serialized as a blob + `SNAPSHOT_SCHEMA_VERSION`) plus
  a separate **labels** table (per `issue_time`×`lead_hour`, written once obs land). One
  **`data.parquet` per deployment-month** (coalesced **read-modify-write**, write-time dedupe;
  latest `written_at` wins), path-based; persisted to a **public `training-data` branch** whose
  state is **force-pushed** as a single commit by the hourly inference Action (ADR-0017,
  ADR-0018, amending ADR-0009 now that ACIS is dropped). The store is raw-only — `TRAINING_ROW`
  is the read-time join (snapshot → `build_features` → labels).
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
  the raw **training store** in a separate **private repo** (ADR-0009); source/configs/docs
  in the main branch (public). Three public homes carry only derived works or code; the raw
  store is the single private home.
- **Derived product** — a transformation of the source data (the forecast JSON, the trained
  models). Redistributable under all source licences *with attribution*; these are the only
  data-bearing artifacts published publicly. Raw observations/forecasts are not (ADR-0009).
- **`registry.json`** — the manifest naming the current champion version per
  `(deployment_id, task)`. Updated by the publish gate; read by the inference pipeline.

## Conventions baked into the system

- **UTC everywhere.** All timestamps are UTC; local time is a display concern only.
- **`schema_version`** is carried on both the forecast JSON and the training store, so
  clients and pipelines can refuse incompatible payloads instead of misreading them.
- **One HRDPS spec.** HRDPS from CaSPAr (training) and from GeoMet/Datamart (inference)
  must resolve to the identical variable/grid specification, or seed and logged data
  diverge.
- **Attribution is mandatory.** Every public, data-bearing artifact carries source
  attribution: ECCC (`Data Source: Environment and Climate Change Canada`) and CaSPAr
  (cite Mai et al. 2020). (ACIS attribution is no longer required for v1 — ACIS is dropped,
  ADR-0010 — but its licence section is retained in `DATA_LICENSES.md` for the deferred path.)
  Enumerated in `DATA_LICENSES.md`, embedded in the forecast JSON `attribution` field, and
  shown in the dashboard footer (ADR-0009).
- **Guardrails over discipline.** Architectural rules are enforced mechanically (types,
  schemas, CI fitness functions), not by convention — see the scaffolding spec in
  `docs/superpowers/specs/`. Bad structural decisions should fail at construction or in CI.
