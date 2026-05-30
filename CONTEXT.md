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
  by `deployment_id`. The system is **multi-deployment by design**. v1 ships **two**
  deployments: `lethbridge` (seeded, trainable now) and `lethbridge_henderson` (cold-start,
  accumulating) — see *training strategy*.
- **Training strategy** — a per-deployment mode:
  - **`seeded`** — sources have *deep* historical coverage; training uses the CaSPAr
    *historical seed* + the logger; trainable from day one.
  - **`cold_start`** — the target source is live-only (e.g. CWOP); there is **no historical
    seed**, the *logger* is the sole label source, and the deployment publishes no model
    until a configured minimum of logged rows accumulates.
- **Cold start** — the condition of a `cold_start` deployment before enough forward-logged
  labels exist to train. The inference run still logs; the training pipeline reports
  "insufficient data" rather than publishing an untrained model.
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
- **Issue time** (`t₀`) — the reference time a feature snapshot is built at. The forecast
  predicts `t₀+1 … t₀+48`.
- **As-of reconstruction** — the invariant that a feature snapshot only ever uses
  observations available **at or before `t₀`**. Future data cannot enter, by construction.
  Guarantees no label leakage.
- **Missingness mask** — a per-observation-feature flag marking whether the value was
  present or imputed. Lets a down feed degrade accuracy gracefully instead of crashing.
- **Dual-feed source** — an observation source that provides **both** an hourly
  historical feed (for training) **and** an hourly live feed (for inference), for the same
  physical measurement. Daily-only sources such as CoCoRaHS are ineligible. Eligibility is
  **strategy-aware** (see *training strategy*): a `seeded` deployment requires every source
  to have **deep** historical coverage; a `cold_start` deployment may use a live-only target
  source (ADR-0008).
- **Historical coverage** — a declared, validated capability of each `ObservationSource`:
  `deep` (multi-year, e.g. Environment Canada, ACIS), `shallow`, or `none` (live-only, e.g.
  CWOP). Depth is explicit and machine-checked, not assumed.
- **Connector** — the abstraction over a data source. An `NWPSource` or an
  `ObservationSource`; observation connectors implement both historical and live fetch (the
  dual-feed contract) and declare their *historical coverage*. A live-only source
  implements `fetch_historical` as best-effort and declares coverage `none`/`shallow`.
- **Available sources** — the free, no-device sources this project actually uses:
  **Environment Canada** (SWOB live + historical CSV, deep), **ACIS** (current + historical,
  deep), and **CWOP** (live only, no free deep history). Consumer-PWS network APIs
  (Weather Underground, Tempest, Ambient) are device-gated and **not used** (ADR-0008).
- **CWOP** (Citizen Weather Observer Program) — a free, no-device network of volunteer PWS
  observations. Used as a *live-only* source for `cold_start` deployments (it has no
  reliable free multi-year history).
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
- **Baseline** / **raw HRDPS** — the unmodified HRDPS forecast at the target. The floor a
  model must beat to justify the project's existence.
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
- **Training store** — the accumulating dataset of feature snapshots + labels: CaSPAr seed
  plus logged snapshots. Partitioned Parquet, per deployment.
- **Forecast JSON** — the single published, schema-versioned output document a deployment
  produces. The only thing thin clients read.
- **Thin client** — a consumer that *only* reads the forecast JSON (never touches HRDPS,
  station feeds, or models). The dashboard and the future Android app are thin clients.
- **Dashboard** — the v1 thin client: static files served from GitHub Pages, reading the
  forecast JSON from the same origin. Lives in this repo, outside the Python import graph.
- **The four homes** — where artifacts live: forecast JSON + dashboard + `registry.json`
  on `gh-pages`; model binaries as versioned GitHub **Release** assets; training data on a
  `training-data` branch; everything else in the main branch.
- **`registry.json`** — the manifest naming the current champion version per
  `(deployment_id, task)`. Updated by the publish gate; read by the inference pipeline.

## Conventions baked into the system

- **UTC everywhere.** All timestamps are UTC; local time is a display concern only.
- **`schema_version`** is carried on both the forecast JSON and the training store, so
  clients and pipelines can refuse incompatible payloads instead of misreading them.
- **One HRDPS spec.** HRDPS from CaSPAr (training) and from GeoMet/Datamart (inference)
  must resolve to the identical variable/grid specification, or seed and logged data
  diverge.
- **Guardrails over discipline.** Architectural rules are enforced mechanically (types,
  schemas, CI fitness functions), not by convention — see the scaffolding spec in
  `docs/superpowers/specs/`. Bad structural decisions should fail at construction or in CI.
