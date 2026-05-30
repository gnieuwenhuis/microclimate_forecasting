# Repository Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the guardrailed repository skeleton for the microclimate forecasting system — every module, contract object, and CI fitness function from `docs/superpowers/specs/2026-05-30-scaffolding-spec.md` exists, with domain logic stubbed (`NotImplementedError`) but the architecture mechanically enforced.

**Architecture:** A `src/`-layout Python package `microclimate` with seven import-layers (L0 contracts → L6 pipelines). Boundary objects are Pydantic v2 models and Pandera DataFrame schemas. Data sources are ABC-defined connectors registered in a registry. Layer direction, the single-feature-builder rule, source eligibility, and config validity are enforced by `import-linter` + a pytest suite that runs in CI. The *guardrails* (contracts, config, registry, validators, tests, lint config) are fully implemented; *domain* code (fetching, feature math, model fit/predict, metrics, publication, pipeline bodies) raises `NotImplementedError`.

**Tech Stack:** Python 3.12, `uv`, Pydantic v2, Pandera, pandas, LightGBM (declared only), Ruff, Pyright (strict), import-linter, Pytest, GitHub Actions.

**Conventions for every task:** run all Python tools through `uv run`. UTC-aware datetimes everywhere (Pydantic `AwareDatetime`). `from __future__ import annotations` at the top of every module. Domain stub bodies are exactly `raise NotImplementedError`.

---

### Task 1: Project bootstrap (uv, pyproject, tooling, empty package)

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/microclimate/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "microclimate"
version = "0.0.0"
description = "Free microclimate temperature + PoP forecasting by downscaling HRDPS."
requires-python = ">=3.12"
dependencies = [
  "pydantic>=2.6",
  "pandera[pandas]>=0.20",
  "pandas>=2.2",
  "lightgbm>=4.3",
  "pyyaml>=6.0",
]

[project.scripts]
microclimate-inference = "microclimate.pipelines.inference:main"
microclimate-training = "microclimate.pipelines.training:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/microclimate"]

[dependency-groups]
dev = [
  "ruff>=0.6",
  "pyright>=1.1.380",
  "import-linter>=2.0",
  "pytest>=8.0",
  "pandas-stubs>=2.2",
  "types-PyYAML>=6.0",
]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.12"
typeCheckingMode = "strict"
venvPath = "."
venv = ".venv"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.python-version`**

```
3.12
```

- [ ] **Step 3: Create package + test bootstrap files**

`src/microclimate/__init__.py`:
```python
"""Microclimate forecasting: downscale HRDPS to a local station (temp + PoP)."""
```

`tests/__init__.py`: (empty file)

`tests/conftest.py`:
```python
"""Shared pytest fixtures (none yet)."""
```

- [ ] **Step 4: Sync and verify the empty toolchain passes**

Run: `uv sync`
Expected: creates `.venv` and `uv.lock`, installs deps + dev group, exit 0.

Run: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
Expected: ruff clean; pyright `0 errors`; pytest `no tests ran` (exit 5 is acceptable here — there are no tests yet). Treat ruff+pyright passing as the gate for this task.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version src/microclimate/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: bootstrap uv project, tooling, empty package"
```

---

### Task 2: L0 contract — `OBSERVATION_FRAME` + `ObservationRecord`

**Files:**
- Create: `src/microclimate/contracts/__init__.py`
- Create: `src/microclimate/contracts/observation.py`
- Test: `tests/contracts/__init__.py`, `tests/contracts/test_observation.py`

- [ ] **Step 1: Write the failing test**

`tests/contracts/__init__.py`: (empty)

`tests/contracts/test_observation.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from microclimate.contracts.observation import OBSERVATION_FRAME


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["9835"],
            "timestamp": pd.to_datetime(["2026-05-30T00:00:00Z"]),
            "temp_c": [12.3],
            "temp_c_present": [True],
            "precip_mm": [0.0],
            "precip_mm_present": [True],
        }
    )


def test_valid_observation_frame_passes() -> None:
    OBSERVATION_FRAME.validate(_valid_frame())


def test_missing_mask_column_fails() -> None:
    frame = _valid_frame().drop(columns=["temp_c_present"])
    with pytest.raises(Exception):
        OBSERVATION_FRAME.validate(frame)


def test_extra_column_fails() -> None:
    frame = _valid_frame()
    frame["humidity"] = [50.0]
    with pytest.raises(Exception):
        OBSERVATION_FRAME.validate(frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_observation.py -v`
Expected: FAIL — `ModuleNotFoundError: microclimate.contracts.observation`.

- [ ] **Step 3: Write the contract**

`src/microclimate/contracts/__init__.py`:
```python
"""L0 contracts: pure Pydantic/Pandera boundary types (no internal imports)."""
```

`src/microclimate/contracts/observation.py`:
```python
"""Standardized observation frame + single-record model (L0)."""

from __future__ import annotations

import pandera.pandas as pa
from pydantic import AwareDatetime, BaseModel, ConfigDict

# Every observation source must emit exactly these columns. Each measurement is paired
# with a `<field>_present` mask so a down feed degrades to imputed+masked, never crashes.
OBSERVATION_FRAME = pa.DataFrameSchema(
    {
        "station_id": pa.Column(str),
        "timestamp": pa.Column("datetime64[ns, UTC]"),
        "temp_c": pa.Column(float, nullable=True),
        "temp_c_present": pa.Column(bool),
        "precip_mm": pa.Column(float, nullable=True),
        "precip_mm_present": pa.Column(bool),
    },
    strict=True,
    coerce=True,
)


class ObservationRecord(BaseModel):
    """One row of OBSERVATION_FRAME."""

    model_config = ConfigDict(extra="forbid")

    station_id: str
    timestamp: AwareDatetime
    temp_c: float | None
    temp_c_present: bool
    precip_mm: float | None
    precip_mm_present: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_observation.py -v && uv run pyright`
Expected: 3 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/contracts/__init__.py src/microclimate/contracts/observation.py tests/contracts/
git commit -m "feat(contracts): OBSERVATION_FRAME schema + ObservationRecord"
```

---

### Task 3: L0 contract — `FeatureSnapshot`

**Files:**
- Create: `src/microclimate/contracts/snapshot.py`
- Test: `tests/contracts/test_snapshot.py`

- [ ] **Step 1: Write the failing test**

`tests/contracts/test_snapshot.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from microclimate.contracts.snapshot import FeatureSnapshot


def test_valid_snapshot_constructs() -> None:
    snap = FeatureSnapshot(
        deployment_id="lethbridge",
        issue_time=datetime(2026, 5, 30, tzinfo=timezone.utc),
        nwp_features={"t2m_lead1": 11.0},
        observation_features={"target_temp_lag1": 10.5},
        observation_masks={"target_temp_lag1": True},
        static_features={"lat": 49.68872},
        temporal_features={"hour_sin": 0.0},
        lead_hours=(1, 2, 3),
        schema_version="1",
    )
    assert snap.deployment_id == "lethbridge"
    assert snap.lead_hours == (1, 2, 3)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureSnapshot(
            deployment_id="lethbridge",
            issue_time=datetime(2026, 5, 30),  # naive — no tzinfo
            nwp_features={},
            observation_features={},
            observation_masks={},
            static_features={},
            temporal_features={},
            lead_hours=(1,),
            schema_version="1",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_snapshot.py -v`
Expected: FAIL — `ModuleNotFoundError: microclimate.contracts.snapshot`.

- [ ] **Step 3: Write the contract**

`src/microclimate/contracts/snapshot.py`:
```python
"""The single canonical model-input object (L0)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import AwareDatetime, BaseModel, ConfigDict


class FeatureSnapshot(BaseModel):
    """Inputs for one prediction at issue_time. Built only by features.build_snapshot."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    issue_time: AwareDatetime
    nwp_features: Mapping[str, float]
    observation_features: Mapping[str, float]
    observation_masks: Mapping[str, bool]
    static_features: Mapping[str, float]
    temporal_features: Mapping[str, float]
    lead_hours: tuple[int, ...]
    schema_version: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_snapshot.py -v && uv run pyright`
Expected: 2 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/contracts/snapshot.py tests/contracts/test_snapshot.py
git commit -m "feat(contracts): FeatureSnapshot model"
```

---

### Task 4: L0 contract — `ForecastDocument` + `ForecastStep`

**Files:**
- Create: `src/microclimate/contracts/forecast.py`
- Test: `tests/contracts/test_forecast.py`

- [ ] **Step 1: Write the failing test**

`tests/contracts/test_forecast.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from microclimate.contracts.forecast import ForecastDocument, ForecastStep


def _doc(**overrides: object) -> ForecastDocument:
    base: dict[str, object] = dict(
        schema_version="1",
        deployment_id="lethbridge",
        issue_time=datetime(2026, 5, 30, tzinfo=timezone.utc),
        last_updated=datetime(2026, 5, 30, tzinfo=timezone.utc),
        status="ok",
        model_versions={"temp": "1.0.0", "pop": "1.0.0"},
        attribution=["Data Source: Environment and Climate Change Canada"],
        series=[
            ForecastStep(
                lead_hour=1,
                valid_time=datetime(2026, 5, 30, 1, tzinfo=timezone.utc),
                temp_c=11.0,
                pop=0.2,
            )
        ],
    )
    base.update(overrides)
    return ForecastDocument(**base)  # type: ignore[arg-type]


def test_valid_document() -> None:
    assert _doc().status == "ok"


def test_pop_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastStep(
            lead_hour=1,
            valid_time=datetime(2026, 5, 30, 1, tzinfo=timezone.utc),
            temp_c=11.0,
            pop=1.5,
        )


def test_empty_attribution_rejected() -> None:
    with pytest.raises(ValidationError):
        _doc(attribution=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_forecast.py -v`
Expected: FAIL — `ModuleNotFoundError: microclimate.contracts.forecast`.

- [ ] **Step 3: Write the contract**

`src/microclimate/contracts/forecast.py`:
```python
"""The published forecast document — the only thing thin clients read (L0)."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class ForecastStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lead_hour: int = Field(ge=1, le=48)
    valid_time: AwareDatetime
    temp_c: float
    pop: float = Field(ge=0.0, le=1.0)


class ForecastDocument(BaseModel):
    """Derived predictions only — never raw observations (ADR-0009)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    deployment_id: str
    issue_time: AwareDatetime
    last_updated: AwareDatetime
    status: Literal["ok", "stale", "degraded"]
    model_versions: dict[Literal["temp", "pop"], str]
    attribution: list[str] = Field(min_length=1)
    series: list[ForecastStep]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_forecast.py -v && uv run pyright`
Expected: 3 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/contracts/forecast.py tests/contracts/test_forecast.py
git commit -m "feat(contracts): ForecastDocument + ForecastStep with attribution"
```

---

### Task 5: L0 contracts — `RegistryManifest` + `TRAINING_ROW`

**Files:**
- Create: `src/microclimate/contracts/registry.py`
- Create: `src/microclimate/contracts/training_store.py`
- Test: `tests/contracts/test_registry_manifest.py`, `tests/contracts/test_training_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/contracts/test_registry_manifest.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone

from microclimate.contracts.registry import (
    RegistryEntry,
    RegistryManifest,
    manifest_key,
)


def test_manifest_roundtrip() -> None:
    entry = RegistryEntry(
        version="1.0.0",
        release_asset_url="https://example/asset",
        promoted_at=datetime(2026, 5, 30, tzinfo=timezone.utc),
        holdout_metrics={"mae_skill": 0.12},
    )
    manifest = RegistryManifest(entries={manifest_key("lethbridge", "temp"): entry})
    assert manifest.entries["lethbridge/temp"].version == "1.0.0"
```

`tests/contracts/test_training_store.py`:
```python
from __future__ import annotations

import pandas as pd

from microclimate.contracts.training_store import TRAINING_ROW


def test_training_row_accepts_extra_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "schema_version": ["1"],
            "deployment_id": ["lethbridge"],
            "issue_time": pd.to_datetime(["2026-05-30T00:00:00Z"]),
            "lead_hour": [1],
            "valid_time": pd.to_datetime(["2026-05-30T01:00:00Z"]),
            "label_temp_c": [11.0],
            "label_precip_occurrence": [0],
            "nwp_t2m_lead1": [11.2],  # dynamic feature column — allowed
        }
    )
    TRAINING_ROW.validate(frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/contracts/test_registry_manifest.py tests/contracts/test_training_store.py -v`
Expected: FAIL — both modules missing.

- [ ] **Step 3: Write the contracts**

`src/microclimate/contracts/registry.py`:
```python
"""Champion-pointer manifest (L0)."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

Task = Literal["temp", "pop"]


def manifest_key(deployment_id: str, task: Task) -> str:
    """Canonical manifest key: '{deployment_id}/{task}'."""
    return f"{deployment_id}/{task}"


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    release_asset_url: str
    promoted_at: AwareDatetime
    holdout_metrics: dict[str, float]


class RegistryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: dict[str, RegistryEntry] = {}
```

`src/microclimate/contracts/training_store.py`:
```python
"""Schema of the accumulating training store (L0). strict=False — feature columns vary."""

from __future__ import annotations

import pandera.pandas as pa

TRAINING_ROW = pa.DataFrameSchema(
    {
        "schema_version": pa.Column(str),
        "deployment_id": pa.Column(str),
        "issue_time": pa.Column("datetime64[ns, UTC]"),
        "lead_hour": pa.Column(int, pa.Check.in_range(1, 48)),
        "valid_time": pa.Column("datetime64[ns, UTC]"),
        "label_temp_c": pa.Column(float, nullable=True),
        "label_precip_occurrence": pa.Column(int, pa.Check.isin([0, 1]), nullable=True),
    },
    strict=False,
    coerce=True,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/contracts/test_registry_manifest.py tests/contracts/test_training_store.py -v && uv run pyright`
Expected: 2 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/contracts/registry.py src/microclimate/contracts/training_store.py tests/contracts/test_registry_manifest.py tests/contracts/test_training_store.py
git commit -m "feat(contracts): RegistryManifest + TRAINING_ROW schema"
```

---

### Task 6: L1 config — `DeploymentConfig` schema

**Files:**
- Create: `src/microclimate/config/__init__.py`
- Create: `src/microclimate/config/schema.py`
- Test: `tests/config/__init__.py`, `tests/config/test_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/config/__init__.py`: (empty)

`tests/config/test_schema.py`:
```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from microclimate.config.schema import DeploymentConfig


def _raw() -> dict[str, object]:
    return {
        "deployment_id": "demo",
        "target": {
            "station_id": "9835",
            "connector_key": "acis",
            "lat": 49.68872,
            "lon": -112.74494,
            "elevation_m": 903,
        },
        "neighbors": [
            {
                "station_id": "3033875",
                "connector_key": "envcanada",
                "lat": 49.6303,
                "lon": -112.7989,
                "elevation_m": None,
            }
        ],
        "enabled_sources": ["hrdps_geomet", "hrdps_caspar", "envcanada", "acis"],
        "nwp": {
            "product": "hrdps",
            "live_connector": "hrdps_geomet",
            "historical_connector": "hrdps_caspar",
            "sampling": "nearest_grid_cell",
        },
        "horizon_hours": 48,
        "lag_hours": 6,
        "feature_groups": {"nwp": True, "observations": True},
        "label": {"precip_occurrence_threshold_mm": 0.2},
        "training": {"seed": {"source": "caspar", "start": "2017-05-22"}, "holdout_months": 12},
        "output": {"forecast_json": "forecasts/demo.json"},
    }


def test_valid_config() -> None:
    config = DeploymentConfig.model_validate(_raw())
    assert config.target.connector_key == "acis"
    assert config.neighbors[0].elevation_m is None


def test_unknown_key_rejected() -> None:
    raw = _raw()
    raw["training_strategy"] = "seeded"  # removed field — must be rejected
    with pytest.raises(ValidationError):
        DeploymentConfig.model_validate(raw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: microclimate.config.schema`.

- [ ] **Step 3: Write the schema**

`src/microclimate/config/__init__.py`:
```python
"""L1 config: validated deployment definitions. Imports only contracts."""
```

`src/microclimate/config/schema.py`:
```python
"""DeploymentConfig and nested local models (L1, ADR-0006)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    station_id: str
    connector_key: str
    lat: float
    lon: float
    elevation_m: float | None = None


class NwpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str
    live_connector: str
    historical_connector: str
    sampling: str


class FeatureGroupSwitches(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nwp: bool
    observations: bool


class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precip_occurrence_threshold_mm: float = Field(ge=0.0)


class SeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    start: str


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: SeedConfig
    holdout_months: int = Field(ge=1)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_json: str


class DeploymentConfig(BaseModel):
    """One fully-specified deployment. Everything is namespaced by deployment_id."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    target: StationRef
    neighbors: list[StationRef]
    enabled_sources: list[str]
    nwp: NwpConfig
    horizon_hours: int = 48
    lag_hours: int = Field(ge=0)
    feature_groups: FeatureGroupSwitches
    label: LabelConfig
    training: TrainingConfig
    output: OutputConfig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_schema.py -v && uv run pyright`
Expected: 2 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/config/__init__.py src/microclimate/config/schema.py tests/config/__init__.py tests/config/test_schema.py
git commit -m "feat(config): DeploymentConfig schema (extra=forbid)"
```

---

### Task 7: L1 config — `loader.py` (loads the real `lethbridge.yml`)

**Files:**
- Create: `src/microclimate/config/loader.py`
- Test: `tests/config/test_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/config/test_loader.py`:
```python
from __future__ import annotations

import pytest

from microclimate.config.loader import list_deployments, load_deployment


def test_lethbridge_is_listed() -> None:
    assert "lethbridge" in list_deployments()


def test_load_lethbridge_returns_config() -> None:
    config = load_deployment("lethbridge")
    assert config.deployment_id == "lethbridge"
    assert config.target.connector_key == "acis"
    assert config.target.station_id == "9835"


def test_missing_deployment_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_deployment("does-not-exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: microclimate.config.loader`.

- [ ] **Step 3: Write the loader**

`src/microclimate/config/loader.py`:
```python
"""Load and schema-validate deployment configs (L1). Imports only contracts/schema."""

from __future__ import annotations

from pathlib import Path

import yaml

from microclimate.config.schema import DeploymentConfig

# repo_root/config/deployments — parents[3] from src/microclimate/config/loader.py
DEPLOYMENTS_DIR = Path(__file__).resolve().parents[3] / "config" / "deployments"


def list_deployments(directory: Path = DEPLOYMENTS_DIR) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.yml"))


def load_deployment(deployment_id: str, directory: Path = DEPLOYMENTS_DIR) -> DeploymentConfig:
    path = directory / f"{deployment_id}.yml"
    if not path.exists():
        raise FileNotFoundError(f"No deployment config: {path}")
    data = yaml.safe_load(path.read_text())
    return DeploymentConfig.model_validate(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_loader.py -v && uv run pyright`
Expected: 3 passed; pyright 0 errors. (If it fails on a schema mismatch, the real `config/deployments/lethbridge.yml` and the schema disagree — reconcile them; the YAML is authoritative for field names.)

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/config/loader.py tests/config/test_loader.py
git commit -m "feat(config): load_deployment + list_deployments"
```

---

### Task 8: L2 connectors — `base.py` ABCs

**Files:**
- Create: `src/microclimate/connectors/__init__.py`
- Create: `src/microclimate/connectors/base.py`
- Test: `tests/connectors/__init__.py`, `tests/connectors/test_base.py`

- [ ] **Step 1: Write the failing test**

`tests/connectors/__init__.py`: (empty)

`tests/connectors/test_base.py`:
```python
from __future__ import annotations

import pytest

from microclimate.connectors.base import NWPSource, ObservationSource


def test_incomplete_observation_source_cannot_instantiate() -> None:
    class Broken(ObservationSource):
        @property
        def historical_coverage(self) -> str:  # type: ignore[override]
            return "deep"

        # missing fetch_historical / fetch_live

    with pytest.raises(TypeError):
        Broken()  # type: ignore[abstract]


def test_incomplete_nwp_source_cannot_instantiate() -> None:
    class Broken(NWPSource):
        @property
        def is_live(self) -> bool:
            return True

        # missing fetch_forecast

    with pytest.raises(TypeError):
        Broken()  # type: ignore[abstract]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/connectors/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: microclimate.connectors.base`.

- [ ] **Step 3: Write the ABCs**

`src/microclimate/connectors/__init__.py`:
```python
"""L2 connectors. Importing this package registers all sources as a side effect."""

from microclimate.connectors import sources as sources  # noqa: F401  (populates registry)
```

`src/microclimate/connectors/base.py`:
```python
"""Source abstractions and the dual-feed contract (L2, ADR-0002/0008)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import pandas as pd

HistoricalCoverage = Literal["deep", "shallow", "none"]


class Source(ABC):
    """Common base for every data connector."""


class NWPSource(Source):
    @property
    @abstractmethod
    def is_live(self) -> bool: ...

    @abstractmethod
    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame: ...


class ObservationSource(Source):
    @property
    @abstractmethod
    def historical_coverage(self) -> HistoricalCoverage: ...

    @abstractmethod
    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame: ...
```

> Note: `connectors/__init__.py` imports `sources`, which doesn't exist until Task 10. To keep this task's test green, create an **empty** `src/microclimate/connectors/sources/__init__.py` now:
```python
"""Source stubs. Each module self-registers on import."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/connectors/test_base.py -v && uv run pyright`
Expected: 2 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/connectors/__init__.py src/microclimate/connectors/base.py src/microclimate/connectors/sources/__init__.py tests/connectors/__init__.py tests/connectors/test_base.py
git commit -m "feat(connectors): NWPSource + ObservationSource ABCs"
```

---

### Task 9: L2 connectors — `registry.py` + `validate_config_sources`

**Files:**
- Create: `src/microclimate/connectors/registry.py`
- Test: `tests/connectors/test_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/connectors/test_registry.py`:
```python
from __future__ import annotations

from datetime import datetime
from collections.abc import Sequence

import pandas as pd
import pytest

from microclimate.config.schema import (
    DeploymentConfig,
    FeatureGroupSwitches,
    LabelConfig,
    NwpConfig,
    OutputConfig,
    SeedConfig,
    StationRef,
    TrainingConfig,
)
from microclimate.connectors.base import HistoricalCoverage, NWPSource, ObservationSource
from microclimate.connectors.registry import (
    get_source,
    register_source,
    registered_keys,
    validate_config_sources,
)


@register_source("_test_nwp")
class _TestNwp(NWPSource):
    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        raise NotImplementedError


@register_source("_test_deep")
class _TestDeep(ObservationSource):
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError


@register_source("_test_shallow")
class _TestShallow(ObservationSource):
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "shallow"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError


def _config(target_key: str, sources: list[str]) -> DeploymentConfig:
    return DeploymentConfig(
        deployment_id="t",
        target=StationRef(station_id="s", connector_key=target_key, lat=0.0, lon=0.0),
        neighbors=[],
        enabled_sources=sources,
        nwp=NwpConfig(
            product="hrdps",
            live_connector="_test_nwp",
            historical_connector="_test_nwp",
            sampling="nearest_grid_cell",
        ),
        lag_hours=6,
        feature_groups=FeatureGroupSwitches(nwp=True, observations=True),
        label=LabelConfig(precip_occurrence_threshold_mm=0.2),
        training=TrainingConfig(seed=SeedConfig(source="caspar", start="2017-05-22"), holdout_months=12),
        output=OutputConfig(forecast_json="forecasts/t.json"),
    )


def test_registered_and_lookup() -> None:
    assert "_test_deep" in registered_keys()
    assert isinstance(get_source("_test_deep"), ObservationSource)


def test_duplicate_key_rejected() -> None:
    with pytest.raises(ValueError):

        @register_source("_test_deep")
        class _Dupe(ObservationSource):
            @property
            def historical_coverage(self) -> HistoricalCoverage:
                return "deep"

            def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
                raise NotImplementedError

            def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
                raise NotImplementedError


def test_deep_source_passes_validation() -> None:
    validate_config_sources(_config("_test_deep", ["_test_nwp", "_test_deep"]))


def test_unregistered_source_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_sources(_config("_test_deep", ["_test_nwp", "_test_deep", "ghost"]))


def test_non_deep_target_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_sources(_config("_test_shallow", ["_test_nwp", "_test_shallow"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/connectors/test_registry.py -v`
Expected: FAIL — `ImportError` for `microclimate.connectors.registry`.

- [ ] **Step 3: Write the registry**

`src/microclimate/connectors/registry.py`:
```python
"""Source registry + strategy-aware eligibility (L2). Imports config (downward) + base."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource, Source

_REGISTRY: dict[str, type[Source]] = {}

S = TypeVar("S", bound=Source)


def register_source(key: str) -> Callable[[type[S]], type[S]]:
    def decorator(cls: type[S]) -> type[S]:
        if key in _REGISTRY:
            raise ValueError(f"Duplicate source key: {key!r}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def is_registered(key: str) -> bool:
    return key in _REGISTRY


def registered_keys() -> set[str]:
    return set(_REGISTRY)


def get_source(key: str) -> Source:
    if key not in _REGISTRY:
        raise KeyError(f"Unregistered source: {key!r}")
    return _REGISTRY[key]()


def validate_config_sources(config: DeploymentConfig) -> None:
    """Raise unless every named source is registered and every station source is deep.

    v1 requires deep historical coverage for all observation sources (ADR-0008).
    """
    for key in config.enabled_sources:
        if not is_registered(key):
            raise ValueError(f"enabled_sources names unregistered source: {key!r}")

    for key in (config.nwp.live_connector, config.nwp.historical_connector):
        source = get_source(key)
        if not isinstance(source, NWPSource):
            raise ValueError(f"nwp connector {key!r} is not an NWPSource")

    for station in [config.target, *config.neighbors]:
        source = get_source(station.connector_key)
        if not isinstance(source, ObservationSource):
            raise ValueError(f"station connector {station.connector_key!r} is not an ObservationSource")
        if source.historical_coverage != "deep":
            raise ValueError(
                f"source {station.connector_key!r} coverage "
                f"{source.historical_coverage!r} != 'deep'"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/connectors/test_registry.py -v && uv run pyright`
Expected: 5 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/connectors/registry.py tests/connectors/test_registry.py
git commit -m "feat(connectors): source registry + validate_config_sources"
```

---

### Task 10: L2 connectors — the four source stubs

**Files:**
- Create: `src/microclimate/connectors/sources/hrdps_geomet.py`, `hrdps_caspar.py`, `envcanada.py`, `acis.py`
- Modify: `src/microclimate/connectors/sources/__init__.py`
- Test: `tests/connectors/test_sources_registered.py`

- [ ] **Step 1: Write the failing test**

`tests/connectors/test_sources_registered.py`:
```python
from __future__ import annotations

import microclimate.connectors  # noqa: F401  (populates the registry on import)
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys


def test_all_v1_sources_registered() -> None:
    assert {"hrdps_geomet", "hrdps_caspar", "envcanada", "acis"} <= registered_keys()


def test_nwp_sources_typed() -> None:
    assert isinstance(get_source("hrdps_geomet"), NWPSource)
    assert get_source("hrdps_geomet").is_live is True
    assert get_source("hrdps_caspar").is_live is False


def test_observation_sources_are_deep() -> None:
    for key in ("envcanada", "acis"):
        source = get_source(key)
        assert isinstance(source, ObservationSource)
        assert source.historical_coverage == "deep"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/connectors/test_sources_registered.py -v`
Expected: FAIL — keys not registered (source modules don't exist).

- [ ] **Step 3: Write the source stubs**

`src/microclimate/connectors/sources/hrdps_geomet.py`:
```python
"""HRDPS via MSC GeoMet/Datamart — live NWP source (stub)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from microclimate.connectors.base import NWPSource
from microclimate.connectors.registry import register_source


@register_source("hrdps_geomet")
class HrdpsGeoMetSource(NWPSource):
    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        raise NotImplementedError
```

`src/microclimate/connectors/sources/hrdps_caspar.py`:
```python
"""HRDPS via the CaSPAr archive — historical NWP seed (stub)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from microclimate.connectors.base import NWPSource
from microclimate.connectors.registry import register_source


@register_source("hrdps_caspar")
class HrdpsCasparSource(NWPSource):
    @property
    def is_live(self) -> bool:
        return False

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        raise NotImplementedError
```

`src/microclimate/connectors/sources/envcanada.py`:
```python
"""Environment Canada SWOB (live) + historical climate CSV — deep dual-feed (stub)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, ObservationSource
from microclimate.connectors.registry import register_source


@register_source("envcanada")
class EnvCanadaSource(ObservationSource):
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError
```

`src/microclimate/connectors/sources/acis.py`:
```python
"""Alberta Climate Information Service — deep dual-feed (stub)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, ObservationSource
from microclimate.connectors.registry import register_source


@register_source("acis")
class AcisSource(ObservationSource):
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError
```

Replace `src/microclimate/connectors/sources/__init__.py` with:
```python
"""Source stubs. Importing this package registers all sources."""

from microclimate.connectors.sources import (  # noqa: F401
    acis,
    envcanada,
    hrdps_caspar,
    hrdps_geomet,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/connectors/test_sources_registered.py -v && uv run pyright`
Expected: 3 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/connectors/sources/ tests/connectors/test_sources_registered.py
git commit -m "feat(connectors): HRDPS GeoMet/CaSPAr + EnvCanada + ACIS source stubs"
```

---

### Task 11: L3 features — `snapshot_builder.py` (the sole feature path)

**Files:**
- Create: `src/microclimate/features/__init__.py`
- Create: `src/microclimate/features/snapshot_builder.py`
- Test: `tests/features/__init__.py`, `tests/features/test_snapshot_builder.py`

- [ ] **Step 1: Write the failing test**

`tests/features/__init__.py`: (empty)

`tests/features/test_snapshot_builder.py`:
```python
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from microclimate.features.snapshot_builder import build_snapshot


def test_signature_takes_issue_time() -> None:
    params = inspect.signature(build_snapshot).parameters
    assert "issue_time" in params  # leakage-proof by signature


def test_builder_is_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        build_snapshot(
            config=None,  # type: ignore[arg-type]
            issue_time=datetime(2026, 5, 30, tzinfo=timezone.utc),
            nwp=None,  # type: ignore[arg-type]
            observations={},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the builder stub**

`src/microclimate/features/__init__.py`:
```python
"""L3 features: the single feature-snapshot builder."""
```

`src/microclimate/features/snapshot_builder.py`:
```python
"""The single, only path that produces a FeatureSnapshot (L3).

As-of / no-leakage: this is the only entry point, it takes issue_time, and the only obs
access is bounded to timestamp <= issue_time. There is no parameter for future data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.contracts.snapshot import FeatureSnapshot


def build_snapshot(
    config: DeploymentConfig,
    issue_time: datetime,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
) -> FeatureSnapshot:
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v && uv run pyright`
Expected: 2 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/ tests/features/
git commit -m "feat(features): build_snapshot stub (the sole feature path)"
```

---

### Task 12: L4 models + evaluation stubs

**Files:**
- Create: `src/microclimate/models/__init__.py`, `models/temp_model.py`, `models/pop_model.py`
- Create: `src/microclimate/evaluation/__init__.py`, `evaluation/metrics.py`, `evaluation/publish_gate.py`
- Test: `tests/models/__init__.py`, `tests/models/test_models_stub.py`, `tests/evaluation/__init__.py`, `tests/evaluation/test_publish_gate.py`

- [ ] **Step 1: Write the failing tests**

`tests/models/__init__.py`, `tests/evaluation/__init__.py`: (empty)

`tests/models/test_models_stub.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor


def test_temp_fit_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        TemperatureRegressor().fit(pd.DataFrame())


def test_pop_calibrate_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        PrecipOccurrenceClassifier().calibrate(pd.DataFrame())
```

`tests/evaluation/test_publish_gate.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from microclimate.evaluation.publish_gate import GateResult, evaluate_challenger


def test_gate_result_shape() -> None:
    result = GateResult(promote=False, reason="stub", metrics={})
    assert result.promote is False


def test_evaluate_challenger_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        evaluate_challenger(
            task="temp",
            challenger=object(),
            champion=None,
            baseline=pd.DataFrame(),
            holdout=pd.DataFrame(),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models tests/evaluation -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write the stubs**

`src/microclimate/models/__init__.py`:
```python
"""L4 models: two independent LightGBM wrappers (ADR-0004)."""
```

`src/microclimate/models/temp_model.py`:
```python
"""Temperature regressor wrapper (L4, stub). lead_hour is a feature."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from microclimate.contracts.snapshot import FeatureSnapshot


class TemperatureRegressor:
    version: str = "0.0.0"

    def fit(self, rows: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, snapshot: FeatureSnapshot) -> dict[int, float]:
        """Return {lead_hour: temperature_c}."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> TemperatureRegressor:
        raise NotImplementedError
```

`src/microclimate/models/pop_model.py`:
```python
"""Precipitation-occurrence classifier wrapper with calibration (L4, stub)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from microclimate.contracts.snapshot import FeatureSnapshot


class PrecipOccurrenceClassifier:
    version: str = "0.0.0"

    def fit(self, rows: pd.DataFrame) -> None:
        raise NotImplementedError

    def calibrate(self, rows: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, snapshot: FeatureSnapshot) -> dict[int, float]:
        """Return {lead_hour: calibrated_pop in [0, 1]}."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> PrecipOccurrenceClassifier:
        raise NotImplementedError
```

`src/microclimate/evaluation/__init__.py`:
```python
"""L4 evaluation: per-lead-hour metrics + the publish gate. Independent of models."""
```

`src/microclimate/evaluation/metrics.py`:
```python
"""Per-lead-hour skill metrics relative to a baseline (L4, stub)."""

from __future__ import annotations

import pandas as pd


def mae_skill(predictions: pd.DataFrame, baseline: pd.DataFrame, truth: pd.DataFrame) -> dict[int, float]:
    """Temperature MAE skill vs baseline, keyed by lead_hour."""
    raise NotImplementedError


def brier_skill(predictions: pd.DataFrame, baseline: pd.DataFrame, truth: pd.DataFrame) -> dict[int, float]:
    """PoP Brier skill vs baseline, keyed by lead_hour."""
    raise NotImplementedError
```

`src/microclimate/evaluation/publish_gate.py`:
```python
"""Champion/challenger publish gate (L4, stub). Imports no model classes (independence)."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from microclimate.contracts.registry import Task


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promote: bool
    reason: str
    metrics: dict[str, float]


def evaluate_challenger(
    task: Task,
    challenger: object,
    champion: object | None,
    baseline: pd.DataFrame,
    holdout: pd.DataFrame,
) -> GateResult:
    """Promote only if the challenger beats both raw HRDPS and the incumbent."""
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models tests/evaluation -v && uv run pyright`
Expected: 4 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/models/ src/microclimate/evaluation/ tests/models/ tests/evaluation/
git commit -m "feat(models,evaluation): model wrappers + publish gate stubs"
```

---

### Task 13: L5 publication stubs

**Files:**
- Create: `src/microclimate/publication/__init__.py`, `publication/forecast_writer.py`, `publication/registry_store.py`
- Test: `tests/publication/__init__.py`, `tests/publication/test_publication_stub.py`

- [ ] **Step 1: Write the failing test**

`tests/publication/__init__.py`: (empty)

`tests/publication/test_publication_stub.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from microclimate.contracts.forecast import ForecastDocument, ForecastStep
from microclimate.publication.forecast_writer import write_forecast
from microclimate.publication.registry_store import read_registry


def test_write_forecast_stubbed(tmp_path: Path) -> None:
    doc = ForecastDocument(
        schema_version="1",
        deployment_id="lethbridge",
        issue_time=datetime(2026, 5, 30, tzinfo=timezone.utc),
        last_updated=datetime(2026, 5, 30, tzinfo=timezone.utc),
        status="ok",
        model_versions={"temp": "1.0.0", "pop": "1.0.0"},
        attribution=["Data Source: Environment and Climate Change Canada"],
        series=[
            ForecastStep(
                lead_hour=1,
                valid_time=datetime(2026, 5, 30, 1, tzinfo=timezone.utc),
                temp_c=11.0,
                pop=0.1,
            )
        ],
    )
    with pytest.raises(NotImplementedError):
        write_forecast(doc, tmp_path / "out.json")


def test_read_registry_stubbed(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        read_registry(tmp_path / "registry.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/publication -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write the stubs**

`src/microclimate/publication/__init__.py`:
```python
"""L5 publication: forecast JSON writer + champion registry manifest store."""
```

`src/microclimate/publication/forecast_writer.py`:
```python
"""Write a ForecastDocument to JSON — only through the validated model (L5, stub)."""

from __future__ import annotations

from pathlib import Path

from microclimate.contracts.forecast import ForecastDocument


def write_forecast(doc: ForecastDocument, path: Path) -> None:
    raise NotImplementedError
```

`src/microclimate/publication/registry_store.py`:
```python
"""Read/update the champion registry manifest (L5, stub)."""

from __future__ import annotations

from pathlib import Path

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, Task


def read_registry(path: Path) -> RegistryManifest:
    raise NotImplementedError


def promote(
    manifest: RegistryManifest, task: Task, deployment_id: str, entry: RegistryEntry
) -> RegistryManifest:
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/publication -v && uv run pyright`
Expected: 2 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/publication/ tests/publication/
git commit -m "feat(publication): forecast writer + registry store stubs"
```

---

### Task 14: L6 pipelines — `inference.py` + `training.py` with CLIs

**Files:**
- Create: `src/microclimate/pipelines/__init__.py`, `pipelines/inference.py`, `pipelines/training.py`
- Test: `tests/pipelines/__init__.py`, `tests/pipelines/test_pipelines_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/pipelines/__init__.py`: (empty)

`tests/pipelines/test_pipelines_cli.py`:
```python
from __future__ import annotations

import pytest

from microclimate.pipelines import inference, training


def test_run_inference_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        inference.run_inference("lethbridge")


def test_run_training_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        training.run_training("lethbridge")


def test_inference_cli_requires_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])  # no --deployment
    with pytest.raises(SystemExit):
        inference.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipelines -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write the pipeline stubs**

`src/microclimate/pipelines/__init__.py`:
```python
"""L6 pipelines: inference (hourly + logger) and training. Orchestrators only."""
```

`src/microclimate/pipelines/inference.py`:
```python
"""Hourly inference + logger pipeline (L6, ADR-0003/0007/0009; stub body)."""

from __future__ import annotations

import argparse


def run_inference(deployment_id: str) -> None:
    """Load config -> validate sources -> build snapshot -> predict -> publish JSON ->
    log the snapshot to the private training store."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly inference for a deployment.")
    parser.add_argument("--deployment", required=True)
    args = parser.parse_args()
    run_inference(args.deployment)


if __name__ == "__main__":
    main()
```

`src/microclimate/pipelines/training.py`:
```python
"""Monthly training pipeline (L6; stub body)."""

from __future__ import annotations

import argparse


def run_training(deployment_id: str) -> None:
    """Load config -> validate sources -> read private store -> train temp & pop ->
    evaluate -> publish gate -> update registry / upload champions on promotion."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training for a deployment.")
    parser.add_argument("--deployment", required=True)
    args = parser.parse_args()
    run_training(args.deployment)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipelines -v && uv run pyright`
Expected: 3 passed; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/pipelines/ tests/pipelines/
git commit -m "feat(pipelines): inference + training CLI stubs"
```

---

### Task 15: Guardrail — `.importlinter` + architecture test (layering + single builder)

**Files:**
- Create: `.importlinter`
- Test: `tests/architecture/__init__.py`, `tests/architecture/test_layering.py`

- [ ] **Step 1: Write the failing test**

`tests/architecture/__init__.py`: (empty)

`tests/architecture/test_layering.py`:
```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "microclimate"


def test_import_linter_contracts_pass() -> None:
    exe = shutil.which("lint-imports")
    assert exe is not None, "import-linter not installed (uv sync)"
    result = subprocess.run([exe], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_only_snapshot_builder_defines_build_snapshot() -> None:
    offenders = [
        str(path)
        for path in SRC.rglob("*.py")
        if path.name != "snapshot_builder.py" and "def build_snapshot" in path.read_text()
    ]
    assert not offenders, f"build_snapshot defined outside snapshot_builder: {offenders}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/architecture/test_layering.py -v`
Expected: FAIL — `lint-imports` errors because `.importlinter` doesn't exist yet.

- [ ] **Step 3: Write the import-linter config**

`.importlinter`:
```ini
[importlinter]
root_package = microclimate

[importlinter:contract:layers]
name = Layered architecture (L0 lowest -> L6 highest)
type = layers
layers =
    microclimate.pipelines
    microclimate.publication
    microclimate.models
    microclimate.evaluation
    microclimate.features
    microclimate.connectors
    microclimate.config
    microclimate.contracts

[importlinter:contract:independence]
name = models and evaluation are independent siblings
type = independence
modules =
    microclimate.models
    microclimate.evaluation
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run lint-imports && uv run pytest tests/architecture/test_layering.py -v`
Expected: `lint-imports` reports both contracts KEPT; 2 passed.

- [ ] **Step 5: Commit**

```bash
git add .importlinter tests/architecture/
git commit -m "feat(guardrails): import-linter layers + single-builder architecture test"
```

---

### Task 16: Guardrail — connector contract-test harness

**Files:**
- Create: `tests/connectors/test_connector_contract.py`

- [ ] **Step 1: Write the test (it should pass immediately against the registered stubs)**

`tests/connectors/test_connector_contract.py`:
```python
from __future__ import annotations

import pytest

import microclimate.connectors  # noqa: F401  (populates the registry)
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys

_KEYS = sorted(k for k in registered_keys() if not k.startswith("_"))


@pytest.mark.parametrize("key", _KEYS)
def test_source_conforms_to_contract(key: str) -> None:
    source = get_source(key)
    assert isinstance(source, (NWPSource, ObservationSource))
    if isinstance(source, ObservationSource):
        assert source.historical_coverage in {"deep", "shallow", "none"}
        assert callable(source.fetch_historical)
        assert callable(source.fetch_live)
    else:
        assert isinstance(source.is_live, bool)
        assert callable(source.fetch_forecast)


@pytest.mark.skip(reason="behavioral checks added when fetch_* is implemented")
@pytest.mark.parametrize("key", _KEYS)
def test_source_behavioral_contract(key: str) -> None:
    # Future: assert OBSERVATION_FRAME conformance, the <= issue_time boundary, masks on
    # missing data, and that declared historical_coverage matches a real probe window.
    raise NotImplementedError
```

- [ ] **Step 2: Run test to verify the structural cases pass**

Run: `uv run pytest tests/connectors/test_connector_contract.py -v`
Expected: 4 parametrized cases pass (`hrdps_geomet`, `hrdps_caspar`, `envcanada`, `acis`); 4 behavioral cases skipped.

- [ ] **Step 3: (no implementation needed — the harness exercises existing stubs)**

- [ ] **Step 4: Verify types**

Run: `uv run pyright`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add tests/connectors/test_connector_contract.py
git commit -m "test(connectors): shared contract harness parametrized over the registry"
```

---

### Task 17: Guardrail — deployments-validity test (acceptance criterion 6)

**Files:**
- Create: `tests/config/test_deployments_valid.py`

- [ ] **Step 1: Write the test**

`tests/config/test_deployments_valid.py`:
```python
from __future__ import annotations

import pytest

import microclimate.connectors  # noqa: F401  (populates the registry)
from microclimate.config.loader import list_deployments, load_deployment
from microclimate.connectors.registry import validate_config_sources


@pytest.mark.parametrize("deployment_id", list_deployments())
def test_committed_deployment_is_valid(deployment_id: str) -> None:
    config = load_deployment(deployment_id)
    validate_config_sources(config)  # raises if any source is unregistered or non-deep
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/config/test_deployments_valid.py -v`
Expected: 1 passed (`lethbridge`). If it fails, `lethbridge.yml` names a source not registered as `deep` — reconcile against the source stubs (`acis`, `envcanada` are deep; `hrdps_geomet`/`hrdps_caspar` are NWP).

- [ ] **Step 3: (no implementation — this validates Task 7 + Task 10 together)**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests pass (behavioral connector cases skipped).

- [ ] **Step 5: Commit**

```bash
git add tests/config/test_deployments_valid.py
git commit -m "test(config): every committed deployment loads + passes source eligibility"
```

---

### Task 18: CI + scheduled workflows

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/inference.yml`, `.github/workflows/training.yml`

- [ ] **Step 1: Write `ci.yml`**

`.github/workflows/ci.yml`:
```yaml
name: CI
on:
  push:
  pull_request:

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run lint-imports
      - run: uv run pyright
      - run: uv run pytest
```

- [ ] **Step 2: Write `inference.yml` (hourly, dynamic matrix over deployments)**

`.github/workflows/inference.yml`:
```yaml
name: inference
on:
  schedule:
    - cron: "0 * * * *"
  workflow_dispatch:

jobs:
  discover:
    runs-on: ubuntu-latest
    outputs:
      deployments: ${{ steps.list.outputs.deployments }}
    steps:
      - uses: actions/checkout@v4
      - id: list
        run: |
          ids=$(ls config/deployments/*.yml | xargs -n1 basename | sed 's/\.yml$//' | jq -R . | jq -cs .)
          echo "deployments=$ids" >> "$GITHUB_OUTPUT"

  run:
    needs: discover
    runs-on: ubuntu-latest
    strategy:
      matrix:
        deployment: ${{ fromJson(needs.discover.outputs.deployments) }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run python -m microclimate.pipelines.inference --deployment ${{ matrix.deployment }}
        env:
          DATA_REPO_TOKEN: ${{ secrets.DATA_REPO_TOKEN }}
```

- [ ] **Step 3: Write `training.yml` (monthly + dispatch + on config change)**

`.github/workflows/training.yml`:
```yaml
name: training
on:
  schedule:
    - cron: "0 0 1 * *"
  workflow_dispatch:
  push:
    paths:
      - "config/deployments/**"

jobs:
  discover:
    runs-on: ubuntu-latest
    outputs:
      deployments: ${{ steps.list.outputs.deployments }}
    steps:
      - uses: actions/checkout@v4
      - id: list
        run: |
          ids=$(ls config/deployments/*.yml | xargs -n1 basename | sed 's/\.yml$//' | jq -R . | jq -cs .)
          echo "deployments=$ids" >> "$GITHUB_OUTPUT"

  run:
    needs: discover
    runs-on: ubuntu-latest
    strategy:
      matrix:
        deployment: ${{ fromJson(needs.discover.outputs.deployments) }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run python -m microclimate.pipelines.training --deployment ${{ matrix.deployment }}
        env:
          DATA_REPO_TOKEN: ${{ secrets.DATA_REPO_TOKEN }}
```

- [ ] **Step 4: Validate YAML locally**

Run: `uv run python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('workflows parse OK')"`
Expected: `workflows parse OK`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/
git commit -m "ci: add CI gate + hourly inference + monthly training workflows"
```

---

### Task 19: Docs, dashboard skeleton, and gh-pages branch

**Files:**
- Create: `README.md`, `config/README.md`
- Create: `dashboard/index.html`, `dashboard/app.js`, `dashboard/README.md`
- Create (orphan branch): `gh-pages` branch with `README.md`

- [ ] **Step 1: Write root `README.md`**

`README.md`:
```markdown
# Microclimate Forecasting

Free, zero-maintenance hourly **temperature** and **probability-of-precipitation** forecasts
for a local station, by downscaling Environment Canada's HRDPS. Designed around Lethbridge,
Alberta; deployable for any microclimate by config.

- **Domain glossary:** [CONTEXT.md](CONTEXT.md)
- **Decisions:** [docs/adr/](docs/adr/)
- **Data licenses & attribution:** [DATA_LICENSES.md](DATA_LICENSES.md)
- **Scaffolding spec:** [docs/superpowers/specs/2026-05-30-scaffolding-spec.md](docs/superpowers/specs/2026-05-30-scaffolding-spec.md)

The architecture is enforced mechanically: typed boundaries (Pydantic/Pandera), connector
ABCs, a single feature-snapshot builder, source-eligibility validation, and an
`import-linter` layer contract — all gated in CI.
```

- [ ] **Step 2: Write `config/README.md`**

`config/README.md`:
```markdown
# Deployments

One YAML per deployment in `deployments/`, validated against
`microclimate.config.schema.DeploymentConfig`. Every artifact is namespaced by
`deployment_id`.

To add a microclimate: copy `lethbridge.yml`, set the `deployment_id`, `target` (a station
with a registered **deep**-history connector), `neighbors`, and `output.forecast_json`, then
run training for it. CI (`tests/config/test_deployments_valid.py`) asserts the new config
loads and that every named source is registered and deep.
```

- [ ] **Step 3: Write the dashboard skeleton**

`dashboard/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Microclimate Forecast</title>
  </head>
  <body>
    <main>
      <h1>Microclimate Forecast</h1>
      <p id="status">Loading…</p>
      <pre id="series"></pre>
    </main>
    <footer id="attribution"></footer>
    <script src="app.js"></script>
  </body>
</html>
```

`dashboard/app.js`:
```javascript
// Thin client: read the published forecast JSON and render it. No raw data, no secrets.
const DEPLOYMENT_ID = "lethbridge";
const SCHEMA_VERSION = "1";

async function load() {
  const res = await fetch(`forecasts/${DEPLOYMENT_ID}.json`, { cache: "no-store" });
  if (!res.ok) {
    document.getElementById("status").textContent = "Forecast unavailable.";
    return;
  }
  const doc = await res.json();
  if (doc.schema_version !== SCHEMA_VERSION) {
    document.getElementById("status").textContent =
      `Unsupported schema_version ${doc.schema_version}.`;
    return;
  }
  document.getElementById("status").textContent =
    `${doc.status} — updated ${doc.last_updated}`;
  document.getElementById("series").textContent = JSON.stringify(doc.series, null, 2);
  document.getElementById("attribution").textContent = (doc.attribution || []).join(" · ");
}

load();
```

`dashboard/README.md`:
```markdown
# Dashboard (thin client)

Static files served from the `gh-pages` branch. Reads `forecasts/<deployment_id>.json` from
the same origin and renders it. Targets forecast `schema_version` **1** and shows the JSON's
`attribution` strings in the footer (ADR-0009). No build step, no secrets, no raw data.
```

- [ ] **Step 4: Verify, commit on `main`, then create the orphan `gh-pages` branch**

Run: `uv run pytest && uv run ruff check . && uv run pyright`
Expected: all green.

```bash
git add README.md config/README.md dashboard/
git commit -m "docs: README, config guide, dashboard skeleton"
```

Create the public `gh-pages` branch (returns to `main` afterward):
```bash
git switch --orphan gh-pages
git rm -rf . >/dev/null 2>&1 || true
printf '# gh-pages\n\nPublished forecast JSON (`forecasts/<deployment_id>.json`), the\nregistry manifest (`registry.json`), and the built dashboard. Forecast `schema_version`: 1.\nDerived products only — no raw data (ADR-0009).\n' > README.md
git add README.md
git commit -m "chore: initialize gh-pages (derived artifacts only)"
git switch main
```

- [ ] **Step 5: Final verification against acceptance criteria**

Run: `uv sync && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run lint-imports && uv run pytest`
Expected: every gate green — this is the scaffolding "definition of done" (spec acceptance criteria 1–10). Confirm `DATA_LICENSES.md` exists at root, `git branch` lists `gh-pages`, and `git switch main` is the current branch.

```bash
git switch main
git status
```

---

## Self-review notes

- **Spec coverage:** every layout file maps to a task — contracts (T2–T5), config (T6–T7), connectors base/registry/sources (T8–T10), features (T11), models+evaluation (T12), publication (T13), pipelines (T14); the ten guardrails: Pydantic boundaries (T2–T6), Pandera frames (T2, T5), connector ABCs (T8), single builder (T11+T15), leakage-proof signature (T11), registry+eligibility (T9+T17), import-linter layers/independence (T15), contract-test harness (T16), strict typing+lint+CI (T1, T18), ADRs+DATA_LICENSES (pre-existing + T19). Workflows (T18), four homes incl. gh-pages + DATA_REPO_TOKEN (T18, T19), dashboard attribution footer (T19).
- **Deferred config values:** the `# CONFIRM` neighbor coordinates/elevations in `lethbridge.yml` are intentional and do not block any task — the schema allows `elevation_m: None` and the loader/validity tests pass regardless.
- **Type consistency:** `Task` (Literal) defined once in `contracts/registry.py`, reused by `publish_gate` and `registry_store`; `HistoricalCoverage` defined in `connectors/base.py`, reused by sources and the registry; model `predict` returns `dict[int, float]` consistently (chosen over `pd.Series` to keep Pyright-strict clean); source registry keys (`hrdps_geomet`, `hrdps_caspar`, `envcanada`, `acis`) match `lethbridge.yml` and the validity test.
- **No placeholders:** every code/test step contains complete content; the one `@pytest.mark.skip` (behavioral connector contract) is an explicit, justified deferral tied to future `fetch_*` implementation, not a hidden TODO.
