# Repository Scaffolding Spec — Microclimate Forecasting

- **Date:** 2026-05-30
- **Status:** Approved
- **Scope:** repository skeleton only — directory layout, module boundaries and public
  interfaces (signatures as contracts), guardrail tooling configuration, contract-object
  field definitions, CI/workflow files, and acceptance criteria. **No business logic.**

This spec defines *the structure that must exist before any logic is written*, such that
the architectural decisions in `CONTEXT.md` and ADR-0001…0008 are enforced **mechanically**
(by types, schemas, and CI fitness functions) rather than by discipline. A contributor —
human or agent — should find the wrong move rejected at construction time or in CI.

## Problem

The architecture is decided but not yet expressed in code. Without a guardrailed skeleton,
the first implementation work will quietly erode the decisions: someone bypasses the
single feature builder (reintroducing train/serve skew), adds a one-way data source,
hardcodes "Lethbridge", leaks future observations, or crosses a layer boundary. The
skeleton must make each of those mechanically impossible or CI-failing **before** logic
exists.

## Non-goals

- No function bodies, model code, connector implementations, fetching, or feature math.
  Public interfaces are declared (signatures, ABCs, field schemas) so the *contracts*
  exist; bodies raise `NotImplementedError` or are `...`.
- No real data, no credentials, no live calls.
- The Android client (separate repo, future) is out of scope (ADR-0003).

## Design

### Tooling baseline (fixed choices)

| Concern | Choice |
|---|---|
| Language | Python 3.12 |
| Env / deps / build | `uv` + `pyproject.toml` (PEP 621) |
| Package layout | `src/` layout, package name `microclimate` |
| Boundary types | Pydantic v2 |
| Dataframe schema | Pandera |
| Models | LightGBM (declared dep; not invoked in skeleton) |
| Lint / format | Ruff |
| Type check | Pyright, **strict** mode |
| Layer enforcement | `import-linter` |
| Tests | Pytest |
| CI | GitHub Actions |

### Layer model and the dependency rule

Modules are assigned to layers. **A module may import only from strictly lower layers;
imports upward or across a forbidden boundary fail CI** (enforced by `import-linter`,
§Guardrail tooling). Siblings are independent unless stated.

```
L6  pipelines        (training, inference)         — orchestrators; may import any lower layer
L5  publication      (forecast JSON writer, registry manifest)
L4  models, evaluation
L3  features         (the single feature-snapshot builder)
L2  connectors       (NWP + observation sources, registry)
L1  config           (DeploymentConfig loading/validation)
L0  contracts        (pure Pydantic/Pandera types; no internal imports)
```

`publication` (L5) and `models`/`evaluation` (L4) are mutually independent. `pipelines`
(L6) is the only layer permitted to import across all of L0–L5.

### Repository layout

```
/
├── CONTEXT.md                         # exists (domain glossary)
├── README.md                          # one-paragraph what/why + link to CONTEXT.md
├── DATA_LICENSES.md                   # ECCC / ACIS / CaSPAr attributions (ADR-0009)
├── pyproject.toml                     # uv project, deps, tool configs
├── uv.lock                            # committed lockfile
├── .importlinter                      # layer contracts
├── docs/
│   ├── adr/                           # exists (0001–0007)
│   ├── agents/                        # exists
│   └── superpowers/specs/             # this spec
├── config/
│   ├── deployments/
│   │   └── lethbridge.yml             # the single v1 deployment (seeded; ACIS Demo Farm target)
│   └── README.md                      # how to add a deployment
├── src/microclimate/
│   ├── __init__.py
│   ├── contracts/                     # L0
│   │   ├── __init__.py
│   │   ├── observation.py             # ObservationFrame Pandera schema + record model
│   │   ├── snapshot.py                # FeatureSnapshot model
│   │   ├── forecast.py                # ForecastDocument model (the published JSON)
│   │   ├── registry.py                # RegistryManifest model
│   │   └── training_store.py          # TrainingStore Pandera schema
│   ├── config/                        # L1
│   │   ├── __init__.py
│   │   ├── schema.py                  # DeploymentConfig + nested models
│   │   └── loader.py                  # load_deployment(id) -> DeploymentConfig
│   ├── connectors/                    # L2
│   │   ├── __init__.py
│   │   ├── base.py                    # NWPSource, ObservationSource ABCs
│   │   ├── registry.py                # source registry + factory + eligibility check
│   │   └── sources/
│   │       ├── __init__.py
│   │       ├── hrdps_geomet.py        # NWPSource (live)
│   │       ├── hrdps_caspar.py        # NWPSource (historical seed)
│   │       ├── envcanada.py           # ObservationSource (deep history + live)
│   │       └── acis.py                # ObservationSource (deep history + live)
│   ├── features/                      # L3
│   │   ├── __init__.py
│   │   └── snapshot_builder.py        # build_snapshot(...) — the SOLE feature path
│   ├── models/                        # L4
│   │   ├── __init__.py
│   │   ├── temp_model.py              # TemperatureRegressor
│   │   └── pop_model.py               # PrecipOccurrenceClassifier (+ calibration)
│   ├── evaluation/                    # L4
│   │   ├── __init__.py
│   │   ├── metrics.py                 # per-lead-hour skill metrics
│   │   └── publish_gate.py            # champion/challenger gate
│   ├── publication/                   # L5
│   │   ├── __init__.py
│   │   ├── forecast_writer.py         # ForecastDocument -> JSON file
│   │   └── registry_store.py          # read/update registry.json
│   └── pipelines/                     # L6
│       ├── __init__.py
│       ├── training.py                # CLI entrypoint
│       └── inference.py               # CLI entrypoint (also the logger)
├── tests/                             # tests sit OUTSIDE the layered import graph
│   ├── conftest.py
│   ├── contracts/                     # contract-object validation tests
│   ├── config/
│   │   └── test_deployments_valid.py  # every config loads AND passes source eligibility
│   ├── connectors/
│   │   └── test_connector_contract.py # shared harness, parametrized over registry
│   └── architecture/
│       └── test_layering.py           # invokes import-linter as a test
├── dashboard/                         # static thin client (outside Python import graph)
│   ├── index.html
│   ├── app.js
│   └── README.md
└── .github/workflows/
    ├── ci.yml                         # ruff + pyright + import-linter + pytest
    ├── inference.yml                  # hourly cron + manual dispatch (matrix over deployments)
    └── training.yml                   # monthly cron + dispatch + on config change
```

## Unit specifications

Each unit lists **purpose**, **public interface** (the contract that must exist in the
skeleton; bodies are stubs), **dependencies** (allowed lower-layer imports), and **what it
enforces**.

### L0 — `contracts/`

Pure data definitions. No internal imports; depends only on stdlib, Pydantic, Pandera.

- **`observation.py`**
  - Purpose: the standardized observation schema every `ObservationSource` must emit.
  - Interface: a Pandera `DataFrameSchema` `OBSERVATION_FRAME` with columns
    `station_id: str`, `timestamp: datetime[UTC]`, `temp_c: float (nullable)`,
    `precip_mm: float (nullable)`, plus any other standardized fields, **each paired with a
    `<field>_present: bool` mask column**. A Pydantic `ObservationRecord` mirrors one row.
  - Enforces: connectors physically cannot return off-spec frames (validated on the way
    out); the missingness mask is structural, not optional.
- **`snapshot.py`**
  - Purpose: the single canonical model-input object (CONTEXT: *feature snapshot*).
  - Interface: Pydantic `FeatureSnapshot` with fields `deployment_id: str`,
    `issue_time: datetime[UTC]`, `nwp_features: Mapping[str, float]`,
    `observation_features: Mapping[str, float]`, `observation_masks: Mapping[str, bool]`,
    `static_features: Mapping[str, float]`, `temporal_features: Mapping[str, float]`,
    `lead_hours: tuple[int, ...]`, `schema_version: str`. Construction validates types.
  - Enforces: an ill-formed snapshot cannot be constructed.
- **`forecast.py`**
  - Purpose: the published forecast document (the only thing thin clients read).
  - Interface: Pydantic `ForecastDocument` with `schema_version: str`,
    `deployment_id: str`, `issue_time: datetime[UTC]`, `last_updated: datetime[UTC]`,
    `status: Literal["ok","stale","degraded"]`, `model_versions: {"temp": str, "pop": str}`,
    `attribution: list[str]` (data-source acknowledgments — ADR-0009; ECCC/ACIS/CaSPAr),
    `series: list[ForecastStep]` where `ForecastStep` = `{lead_hour: int,
    valid_time: datetime[UTC], temp_c: float, pop: float in [0,1]}`.
  - Enforces: the JSON contract; `pop` range, non-empty `attribution`, and required fields
    validated. Contains only derived predictions — never raw observations (ADR-0009).
- **`registry.py`**
  - Purpose: champion pointer manifest.
  - Interface: Pydantic `RegistryManifest` mapping `(deployment_id, task)` →
    `{version: str, release_asset_url: str, promoted_at: datetime[UTC],
    holdout_metrics: dict}`; `task: Literal["temp","pop"]`.
- **`training_store.py`**
  - Purpose: schema of the accumulating training store (Parquet).
  - Interface: Pandera `TRAINING_ROW` schema covering the flattened snapshot fields + label
    columns (`label_temp_c`, `label_precip_occurrence: int{0,1}`) + `schema_version`,
    `deployment_id`, `issue_time`, `lead_hour`, `valid_time`.

### L1 — `config/`

- **`schema.py`**
  - Purpose: validated deployment definition (ADR-0006).
  - Interface: Pydantic `DeploymentConfig` with `deployment_id: str`,
    `target: StationRef`, `neighbors: list[StationRef]`, `enabled_sources: list[str]`,
    `nwp: NwpConfig` (grid sampling spec, must match across CaSPAr/GeoMet — ADR-0007),
    `horizon_hours: int = 48`, `lag_hours: int`, `feature_groups: FeatureGroupSwitches`,
    `label: LabelConfig` (`precip_occurrence_threshold_mm: float`),
    `training: TrainingConfig` (`seed: SeedConfig`, `holdout_months: int`),
    `output: OutputConfig`.
    `StationRef = {station_id, connector_key, lat, lon, elevation_m}`. `StationRef`,
    `NwpConfig`, `FeatureGroupSwitches`, `LabelConfig`, `TrainingConfig`, `SeedConfig`, and
    `OutputConfig` are **local Pydantic models defined in this module** (not L0 contracts —
    they exist only to structure `DeploymentConfig`).
  - Enforces: a malformed config raises at load (shape, types, ranges). **Source
    *eligibility* is deliberately not checked here** — that would require importing the
    connectors layer (L2) from config (L1), an upward import the layer rule forbids.
    Eligibility is validated one layer up (see `connectors/registry.validate_config_sources`)
    and in a CI config test (see tests).
- **`loader.py`**
  - Purpose: load and schema-validate one deployment by id.
  - Interface: `load_deployment(deployment_id: str) -> DeploymentConfig`;
    `list_deployments() -> list[str]`.
  - Dependencies: `contracts` only. (No connectors import — keeps config strictly below
    connectors in the layer graph.)
  - Enforces: every loaded config is structurally valid. Source-eligibility is enforced
    separately (connectors layer + CI test), not by the loader.

### L2 — `connectors/`

- **`base.py`**
  - Purpose: the source abstractions and the dual-feed contract (ADR-0002).
  - Interface — ABCs with abstract methods (stub bodies):
    - `NWPSource`: `fetch_forecast(issue_time, lat, lon, lead_hours) -> DataFrame`,
      `is_live: bool`.
    - `ObservationSource`: **both** `fetch_historical(station_id, start, end) ->
      DataFrame[OBSERVATION_FRAME]` **and** `fetch_live(station_id, since) ->
      DataFrame[OBSERVATION_FRAME]` (both abstract), plus a declared
      `historical_coverage: Literal["deep","shallow","none"]` capability (ADR-0008). v1
      eligibility requires `deep`; `shallow`/`none` sources exist only for a future
      cold-start path.
  - Enforces: a half-implemented observation source (one feed missing) cannot be
    instantiated; historical depth is a declared, checkable capability, not an assumption.
- **`registry.py`**
  - Purpose: register sources by key; resolve, assert eligibility, and validate a config's
    named sources.
  - Interface: `@register_source(key)` decorator; `get_source(key) -> Source`;
    `is_registered(key) -> bool`; `registered_keys() -> set[str]`;
    **`validate_config_sources(config: DeploymentConfig) -> None`** — raises if the config
    names an unregistered source, or if any named `ObservationSource` declares
    `historical_coverage != "deep"` (ADR-0008: v1 requires deep history for every source).
  - Dependencies: `contracts`, `config` (importing config from L2 is *downward* — allowed).
  - Enforces: config can only name registered, deep-history sources — checked here (called
    by pipelines at startup) and in a CI config test, never by the config layer itself.
- **`sources/*`**: one stub class per source, decorated with `@register_source`, methods
  raising `NotImplementedError`. Present so the registry is populated and the contract-test
  harness has subjects.

### L3 — `features/`

- **`snapshot_builder.py`**
  - Purpose: **the single, only path** that produces a `FeatureSnapshot` (CONTEXT:
    prevents train/serve skew).
  - Interface: `build_snapshot(config: DeploymentConfig, issue_time: datetime[UTC],
    nwp: NWPSource, observations: Mapping[str, ObservationSource]) -> FeatureSnapshot`.
  - Enforces (by signature, even before bodies exist):
    - **As-of / no-leakage** — the function takes `issue_time` and the only obs entry point
      restricts to `timestamp <= issue_time`; there is no parameter through which future
      data can enter.
    - **Single path** — no other module exposes a snapshot-building function; training and
      inference both import this one.
  - Dependencies: `contracts`, `connectors`, `config`.

### L4 — `models/`, `evaluation/`

- **`temp_model.py` / `pop_model.py`** (ADR-0004)
  - Purpose: the two model wrappers, `lead_hour` as a feature.
  - Interface (stub): `fit(rows: DataFrame[TRAINING_ROW]) -> None`,
    `predict(snapshot: FeatureSnapshot) -> Series` keyed by lead hour; `pop_model` adds a
    `calibrate(...)` step and outputs probabilities in `[0,1]`; both expose `version: str`
    and `save(path)/load(path)`.
- **`evaluation/metrics.py`**: per-lead-hour metric functions — temp MAE/RMSE & MAE-skill;
  PoP Brier, reliability, ROC-AUC/PR-AUC & Brier-skill — all relative to a passed baseline.
- **`evaluation/publish_gate.py`** (CONTEXT: publish gate)
  - Interface: `evaluate_challenger(task, challenger, champion, baseline, holdout) ->
    GateResult` where `GateResult = {promote: bool, reason: str, metrics: dict}`. Promotes
    only if challenger beats **both** raw HRDPS and the incumbent.
  - Enforces: shipping a worse model is impossible — the training pipeline only publishes
    on `promote == True`.

### L5 — `publication/`

- **`forecast_writer.py`**: `write_forecast(doc: ForecastDocument, path) -> None` —
  serializes *only* through the validated model.
- **`registry_store.py`**: `read_registry(path) -> RegistryManifest`;
  `promote(manifest, task, deployment_id, entry) -> RegistryManifest`.

### L6 — `pipelines/`

- **`inference.py`** (ADR-0003, ADR-0007)
  - Purpose: hourly job — load config → `validate_config_sources(config)` →
    build snapshot from live sources → predict with champions → write forecast JSON (public,
    with `attribution`) → **log the snapshot** to the **private** training store (ADR-0009).
  - Interface: `run_inference(deployment_id: str) -> None`; module is CLI-invokable
    (`python -m microclimate.pipelines.inference --deployment <id>`).
  - Dependencies: all lower layers.
- **`training.py`**
  - Purpose: load config → `validate_config_sources(config)` → read the **private** training
    store → train temp & pop → evaluate → run publish gate → update registry / upload
    champions on promotion.
  - Interface: `run_training(deployment_id: str) -> None`; CLI-invokable.

### Dashboard (`dashboard/`)

Static skeleton only: `index.html` + `app.js` that fetch
`forecasts/<deployment_id>.json` from the same origin, render a placeholder, and **show the
JSON's `attribution` strings in the footer** (ADR-0009). Outside the Python import graph (the
import-linter contract does not include it). A `README.md` notes the `schema_version` it
targets.

## Guardrail tooling (the ten guardrails, as files)

1. **Pydantic boundaries** — all L0 contract objects (above).
2. **Pandera obs frame** — `OBSERVATION_FRAME` + `TRAINING_ROW`.
3. **Connector ABCs** — `base.py` abstract dual-feed methods.
4. **Single builder** — only `features/snapshot_builder.py` exposes `build_snapshot`; a
   test (`tests/architecture/`) asserts no other module defines a snapshot-building symbol.
5. **Leakage-proof signature** — `build_snapshot` takes `issue_time`; reviewed by the
   contract harness's no-leakage case.
6. **Source registry + strategy-aware eligibility** — `connectors/registry.py` owns
   `validate_config_sources(config)`, enforcing the ADR-0008 rule (seeded → all sources
   `deep`; cold_start → live-only target allowed). Pipelines call it at startup and
   `tests/config/test_deployments_valid.py` asserts every committed config passes. The
   config layer never imports connectors (preserving the layer graph), so eligibility is
   enforced above config and in CI, not inside the loader.
7. **`.importlinter`** — a `layers` contract encoding the L0→L6 order above, plus an
   `independence` contract between `models` and `evaluation` (sibling L4 modules that must
   not import each other; `publish_gate` receives model objects as arguments, not imports).
   Run in CI and as a pytest case.
8. **Connector contract-test harness** — `tests/connectors/test_connector_contract.py`,
   parametrized over `registry.registered_keys()`, asserting for every source: dual-feed
   methods present, output conforms to `OBSERVATION_FRAME`, historical fetch honors the
   `<= issue_time` boundary, missingness yields masks rather than exceptions, and the
   declared `historical_coverage` is consistent with what `fetch_historical` returns over a
   probe window (a source claiming `"deep"` must actually return multi-year history). Adding
   a source automatically subjects it to all cases.
9. **Strict typing + lint** — `pyproject.toml` configures Pyright `strict` and Ruff; CI
   gates on both.
10. **ADRs required** — `docs/adr/` exists; `CONTEXT.md` is the glossary. (Process guardrail;
    enforced by review, noted in `README.md`/CI description.)

### `.github/workflows/`

- **`ci.yml`** — on PR/push: `uv sync`, then Ruff, Pyright (strict), import-linter, and
  Pytest. Merge-gating.
- **`inference.yml`** — hourly `cron` + `workflow_dispatch`; matrix over
  `config/deployments/*`; runs the inference CLI; reads `DATA_REPO_TOKEN` from Actions
  secrets to push logged snapshots to the private data repo. (Skeleton: workflow file with
  the trigger/matrix wiring; the step invokes the stub CLI.)
- **`training.yml`** — monthly `cron` + `workflow_dispatch` + `push` paths-filter on
  `config/deployments/**`; matrix over deployments; runs the training CLI; reads
  `DATA_REPO_TOKEN` to clone the private data repo.

### The four homes (ADR-0003/0007/0009)

| Artifact | Home | Visibility |
|---|---|---|
| Forecast JSON, dashboard, `registry.json` | `gh-pages` branch (served by Pages) | **public** (derived) |
| Model binaries (versioned) | GitHub **Release** assets, keyed `{deployment_id}/{task}/{version}` | **public** (derived) |
| Raw training store (Parquet) | a **separate private repo**, written via `DATA_REPO_TOKEN` | **private** (raw data) |
| Source, configs, docs, workflows | `main` | **public** (code) |

Only derived works and code are public; the raw store is private for data-licensing reasons
(ADR-0009). Scaffolding creates the public `gh-pages` branch (with a `README.md`); the
private data repo is created separately and its push token stored as the `DATA_REPO_TOKEN`
secret. The orphan `gh-pages` branch and the private data repo each get a `README.md`
stating their purpose and the `schema_version` they carry.

## Acceptance criteria (definition of done for scaffolding)

The skeleton is complete when **all** hold:

1. `uv sync` succeeds; `uv.lock` is committed.
2. `ruff check` and `ruff format --check` pass.
3. `pyright` passes in strict mode against the stubbed package (stubs are fully typed).
4. `import-linter` passes and its contracts encode the L0→L6 layering + the
   `models`/`evaluation` independence rule. (No module in L0–L5 imports a higher layer; in
   particular `config` does not import `connectors`.)
5. `pytest` passes, including: contract-object validation tests, the architecture/layering
   test, the connector contract-test harness running (and passing on the trivial structural
   assertions) against every registered source stub, and the deployments-validity test.
6. `config/deployments/lethbridge.yml` loads via `load_deployment` (schema-valid) **and**
   passes `validate_config_sources` (all sources registered + `deep`), asserted by
   `tests/config/test_deployments_valid.py`. The target is ACIS Demo Farm IMCIN (#9835), a
   genuine microclimate station, not the airport (ADR-0006/0008). A few neighbor coordinates
   and elevations remain marked `# CONFIRM` in the YAML (gated ACIS metadata API).
7. Every module file listed in the layout exists with its declared public interface; bodies
   are `...` / `raise NotImplementedError`. No business logic is present.
8. The three workflow files exist with correct triggers/matrix wiring and invoke the stub
   CLIs; `inference.yml`/`training.yml` reference the `DATA_REPO_TOKEN` secret.
9. The public `gh-pages` branch exists with an explanatory `README.md`; the private data
   repo is documented (not created by this repo's scaffolding) in `DATA_LICENSES.md`.
10. `README.md` at root links to `CONTEXT.md` and `docs/adr/`; **`DATA_LICENSES.md`** exists
    with the ECCC/ACIS/CaSPAr attributions, and the `ForecastDocument` defines an
    `attribution` field rendered by the dashboard.

## Self-review notes

- **Placeholders:** none — the one deliberately deferred value (the default target
  `station_id`) is explicitly flagged as a config-authoring decision, with a hard rule
  (not the airport), not a `TBD`.
- **Consistency:** layer model, layout, unit deps, and the import-linter contract all
  reference the same L0–L6 ordering; no unit declares an upward import (source-eligibility
  lives in the connectors layer + a CI test, never in the config layer); the four homes
  match ADR-0003/0006/0007 and CONTEXT.
- **Scope:** structure-only; one implementation plan can execute it. Logic is explicitly a
  non-goal and is gated behind acceptance criterion 7.
- **Ambiguity:** interfaces are given as concrete signatures/field lists so "what exists"
  is unambiguous; behavior (bodies) is intentionally absent and out of scope.
