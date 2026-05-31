# Local Notebook Model Training & Exploration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature-engineering prerequisites (label attachment + training-data assembly), the two LightGBM model wrappers with calibration, the evaluation metrics, and a thin local notebook that trains and explores the models — all driving shared, tested code.

**Architecture:** A pure `attach_labels` turns the label-free feature matrix into a labeled feature matrix; an assembly orchestrator in `pipelines` iterates issue-times through `build_snapshot` → `build_features`, performs the single training-only future read of target observations, and labels the result (with a Parquet cache). Two LightGBM wrappers expose a **row-based** `predict` (resolving ADR-0012's deferred item); PoP adds an isotonic calibration stage. `evaluation.metrics` computes per-lead skill vs the raw-HRDPS baseline and PoP reliability. A thin notebook wires these together over a chronological 3-way split; a fast CI smoke test guards the whole path.

**Tech Stack:** Python 3.12, pandas, LightGBM, scikit-learn (isotonic calibration + joblib persistence), Pandera contracts, pytest, jupytext + matplotlib (notebook group), uv.

---

## Conventions for every task

- TDD: write the failing test, watch it fail, implement minimally, watch it pass, commit.
- Run the full gate before each commit unless a step says otherwise:
  `uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
- LightGBM and scikit-learn are **untyped under pyright strict**. The `# type: ignore[...]` / `# pyright: ignore[...]` comments in the code below are required; run `uv run pyright` and confirm zero errors after each model task.
- Commits go on the current feature branch (main rejects direct pushes; integrate via PR at the end).

## File structure (what gets created / modified)

**Create**
- `src/microclimate/features/labeler.py` — pure `attach_labels`.
- `src/microclimate/models/_columns.py` — shared model-input column selection.
- `src/microclimate/pipelines/training_data.py` — `assemble_training_rows`, `assemble_or_load`, `chronological_split`.
- `tests/fakes.py` — shared fake connectors (moved from `tests/features/conftest.py`).
- `tests/features/test_labeler.py`, `tests/evaluation/test_metrics.py`,
  `tests/models/test_temp_model.py`, `tests/models/test_pop_model.py`,
  `tests/models/test_columns.py`, `tests/pipelines/test_training_data.py`,
  `tests/pipelines/test_smoke_model_dev.py`.
- `notebooks/model_dev.py` — thin jupytext percent-format notebook.
- `docs/adr/0013-notebook-model-dev-and-assembly.md`.

**Modify**
- `src/microclimate/models/temp_model.py`, `src/microclimate/models/pop_model.py` — real wrappers.
- `src/microclimate/evaluation/metrics.py` — fill the stub.
- `src/microclimate/contracts/feature_matrix.py` — correct the TRAINING_ROW docstring.
- `tests/features/conftest.py` — re-export from `tests/fakes.py`.
- `tests/models/test_models_stub.py` — delete (superseded).
- `.importlinter` — add `labeler` to the purity contract.
- `pyproject.toml` — add `scikit-learn` dependency + `notebook` dependency group.
- `.gitignore` — ignore notebook artifacts + generated `.ipynb`.
- `docs/adr/0004-two-lightgbm-models.md`, `docs/adr/0012-feature-builder-read-time-transform.md` — amend for row-based predict.
- `CONTEXT.md` — add domain terms.
- `README.md` — update Project status.

---

# Slice 1 — Pure labeler + evaluation metrics

No models, no network. Pure functions only.

### Task 1.1: `attach_labels` (labeled feature matrix)

**Files:**
- Create: `src/microclimate/features/labeler.py`
- Test: `tests/features/test_labeler.py`
- Modify: `.importlinter`, `src/microclimate/contracts/feature_matrix.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_labeler.py
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.features.labeler import attach_labels

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)


def _matrix(valid_times: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "deployment_id": "test",
            "issue_time": pd.to_datetime([_T0] * len(valid_times), utc=True),
            "lead_hour": list(range(1, len(valid_times) + 1)),
            "valid_time": pd.to_datetime(valid_times, utc=True),
            "nwp_temp_c": 10.0,
        }
    )


def _target_obs(rows: list[tuple[pd.Timestamp, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": "T1",
            "timestamp": pd.to_datetime([r[0] for r in rows], utc=True),
            "temp_c": [r[1] for r in rows],
            "precip_mm": [r[2] for r in rows],
        }
    )


def test_labels_join_at_valid_time_and_threshold() -> None:
    v1 = pd.Timestamp(_T0) + pd.Timedelta(hours=1)
    v2 = pd.Timestamp(_T0) + pd.Timedelta(hours=2)
    matrix = _matrix([v1, v2])
    obs = _target_obs([(v1, 12.5, 0.5), (v2, 9.0, 0.0)])

    out = attach_labels(matrix, obs, threshold_mm=0.2)

    assert list(out["label_temp_c"]) == [12.5, 9.0]
    assert list(out["label_precip_occurrence"]) == [1, 0]
    # original columns preserved (not converted to TRAINING_ROW)
    assert "feature_schema_version" in out.columns
    assert "nwp_temp_c" in out.columns


def test_threshold_is_inclusive() -> None:
    v1 = pd.Timestamp(_T0) + pd.Timedelta(hours=1)
    out = attach_labels(_matrix([v1]), _target_obs([(v1, 5.0, 0.2)]), threshold_mm=0.2)
    assert int(out["label_precip_occurrence"].iloc[0]) == 1


def test_missing_obs_yields_null_labels() -> None:
    v1 = pd.Timestamp(_T0) + pd.Timedelta(hours=1)
    v2 = pd.Timestamp(_T0) + pd.Timedelta(hours=2)
    matrix = _matrix([v1, v2])
    obs = _target_obs([(v1, 12.5, 0.5)])  # nothing for v2

    out = attach_labels(matrix, obs, threshold_mm=0.2)

    assert out["label_temp_c"].iloc[0] == 12.5
    assert pd.isna(out["label_temp_c"].iloc[1])
    assert int(out["label_precip_occurrence"].iloc[0]) == 1
    assert pd.isna(out["label_precip_occurrence"].iloc[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_labeler.py -v`
Expected: FAIL with `ModuleNotFoundError: microclimate.features.labeler`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/microclimate/features/labeler.py
"""Label attachment: label-free feature matrix -> labeled feature matrix (L3, pure).

Pure and connector-free. The *future* read of target observations (values at valid_time,
which are after issue_time) happens in the assembler (pipelines.training_data), where a
training-only future read is legal; this function receives that frame already read. The
output is the **labeled feature matrix** the models train on -- NOT the persisted
TRAINING_ROW (which is raw snapshot + labels in the training store, ADR-0012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_labels(
    matrix: pd.DataFrame,
    target_obs: pd.DataFrame,
    threshold_mm: float,
) -> pd.DataFrame:
    """Add `label_temp_c` and `label_precip_occurrence` by joining target obs at valid_time.

    `label_precip_occurrence` is 1 when observed precip >= threshold_mm, else 0, and <NA>
    where the target observation for that valid_time is missing (ADR-0008 degradation).
    """
    obs = target_obs[["timestamp", "temp_c", "precip_mm"]].copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], utc=True)
    obs = obs.drop_duplicates(subset="timestamp").set_index("timestamp")

    valid = pd.to_datetime(matrix["valid_time"], utc=True)
    temp = valid.map(obs["temp_c"])
    precip = valid.map(obs["precip_mm"])

    # NaN where the target obs is missing, else 1/0 — built via float so the missing
    # entries become <NA> cleanly under the nullable Int64 dtype.
    occurrence = np.where(precip.isna().to_numpy(), np.nan, precip.to_numpy() >= threshold_mm)

    out = matrix.copy()
    out["label_temp_c"] = temp.astype("float64")
    out["label_precip_occurrence"] = pd.array(occurrence, dtype="Int64")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_labeler.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Keep the labeler pure by contract**

Edit `.importlinter` — add `labeler` to the existing purity contract so it can never import connectors:

```ini
[importlinter:contract:feature_builder_purity]
name = build_features and labeler are pure — must not import connectors
type = forbidden
source_modules =
    microclimate.features.feature_builder
    microclimate.features.labeler
forbidden_modules =
    microclimate.connectors
```

Run: `uv run lint-imports`
Expected: All contracts pass.

- [ ] **Step 6: Correct the TRAINING_ROW docstring** (it currently conflates two things)

In `src/microclimate/contracts/feature_matrix.py`, replace the module docstring's second sentence:

```python
"""Schema of the label-free feature matrix (L0). strict=False — feature columns vary.

Produced by features.build_features from a FeatureSnapshot. Attaching labels
(features.attach_labels) yields the LABELED FEATURE MATRIX the models train on. That is
distinct from TRAINING_ROW (training_store.py), the persisted store schema, which per
ADR-0012 holds raw snapshot values + labels and versions independently.
"""
```

- [ ] **Step 7: Commit**

```bash
git add src/microclimate/features/labeler.py tests/features/test_labeler.py .importlinter src/microclimate/contracts/feature_matrix.py
git commit -m "feat(features): attach_labels — labeled feature matrix (pure, connector-free)"
```

### Task 1.2: evaluation metrics (skill vs baseline + reliability)

**Files:**
- Modify: `src/microclimate/evaluation/metrics.py` (currently `NotImplementedError` stub)
- Test: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_metrics.py
from __future__ import annotations

import pandas as pd

from microclimate.evaluation.metrics import (
    pop_skill_by_lead,
    reliability_table,
    temp_skill_by_lead,
)


def test_temp_skill_by_lead() -> None:
    df = pd.DataFrame(
        {
            "lead_hour": [1, 1],
            "pred_temp_c": [1.0, 1.0],
            "label_temp_c": [2.0, 2.0],
            "nwp_temp_c": [0.0, 0.0],  # baseline is twice as wrong
        }
    )
    out = temp_skill_by_lead(df).set_index("lead_hour")
    assert out.loc[1, "mae"] == 1.0
    assert out.loc[1, "rmse"] == 1.0
    assert out.loc[1, "baseline_mae"] == 2.0
    assert out.loc[1, "baseline_rmse"] == 2.0
    assert out.loc[1, "skill"] == 0.5  # MAE skill: 1 - 1/2
    assert out.loc[1, "n"] == 2


def test_pop_skill_by_lead() -> None:
    df = pd.DataFrame(
        {
            "lead_hour": [1, 1],
            "pred_pop": [0.5, 0.5],
            "label_precip_occurrence": [1, 0],
            "baseline_pop": [0.0, 0.0],
        }
    )
    out = pop_skill_by_lead(df).set_index("lead_hour")
    assert out.loc[1, "brier"] == 0.25
    assert out.loc[1, "baseline_brier"] == 0.5
    assert out.loc[1, "bss"] == 0.5


def test_reliability_table_bins() -> None:
    df = pd.DataFrame(
        {"pred_pop": [0.05, 0.95], "label_precip_occurrence": [0, 1]}
    )
    out = reliability_table(df, n_bins=10)
    assert len(out) == 10
    first = out.iloc[0]
    last = out.iloc[-1]
    assert first["count"] == 1 and first["observed_freq"] == 0.0
    assert last["count"] == 1 and last["observed_freq"] == 1.0
    assert abs(first["mean_pred"] - 0.05) < 1e-9
    assert abs(last["mean_pred"] - 0.95) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL (functions raise `NotImplementedError` / are absent).

- [ ] **Step 3: Write the implementation**

```python
# src/microclimate/evaluation/metrics.py
"""Forecast skill metrics vs the raw-HRDPS baseline + PoP reliability (L5, pure).

Per-lead-hour aggregation (CONTEXT.md: metrics reported per lead hour). Shared by the
notebook now and the publish gate later. No models import (sibling independence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temp_skill_by_lead(
    df: pd.DataFrame,
    *,
    pred_col: str = "pred_temp_c",
    label_col: str = "label_temp_c",
    baseline_col: str = "nwp_temp_c",
) -> pd.DataFrame:
    """Per-lead MAE/RMSE and MAE skill vs baseline (CONTEXT.md: "MAE skill for temp").

    skill = 1 - mae / baseline_mae, NaN where the baseline is perfect (baseline_mae == 0).
    rmse / baseline_rmse are reported alongside as diagnostics.
    """
    d = df.dropna(subset=[pred_col, label_col, baseline_col]).copy()
    d["_ae"] = (d[pred_col] - d[label_col]).abs()
    d["_se"] = (d[pred_col] - d[label_col]) ** 2
    d["_bae"] = (d[baseline_col] - d[label_col]).abs()
    d["_bse"] = (d[baseline_col] - d[label_col]) ** 2
    g = d.groupby("lead_hour")
    out = pd.DataFrame(
        {
            "mae": g["_ae"].mean(),
            "rmse": np.sqrt(g["_se"].mean()),
            "baseline_mae": g["_bae"].mean(),
            "baseline_rmse": np.sqrt(g["_bse"].mean()),
            "n": g.size(),
        }
    ).reset_index()
    out["skill"] = 1.0 - out["mae"] / out["baseline_mae"].replace(0.0, np.nan)
    return out


def pop_skill_by_lead(
    df: pd.DataFrame,
    *,
    prob_col: str = "pred_pop",
    label_col: str = "label_precip_occurrence",
    baseline_col: str = "baseline_pop",
) -> pd.DataFrame:
    """Per-lead Brier score and Brier Skill Score vs baseline. bss = 1 - brier/baseline_brier."""
    d = df.dropna(subset=[prob_col, label_col, baseline_col]).copy()
    d[label_col] = d[label_col].astype(float)
    d["_bs"] = (d[prob_col] - d[label_col]) ** 2
    d["_bbs"] = (d[baseline_col] - d[label_col]) ** 2
    g = d.groupby("lead_hour")
    out = pd.DataFrame(
        {"brier": g["_bs"].mean(), "baseline_brier": g["_bbs"].mean(), "n": g.size()}
    ).reset_index()
    out["bss"] = 1.0 - out["brier"] / out["baseline_brier"]
    return out


def reliability_table(
    df: pd.DataFrame,
    *,
    prob_col: str = "pred_pop",
    label_col: str = "label_precip_occurrence",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability-diagram bins: predicted-prob bin vs observed frequency."""
    d = df.dropna(subset=[prob_col, label_col]).copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(d[prob_col].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    d["_bin"] = bins
    records: list[dict[str, float]] = []
    for b in range(n_bins):
        g = d[d["_bin"] == b]
        records.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_pred": float(g[prob_col].mean()) if len(g) else float("nan"),
                "observed_freq": (
                    float(g[label_col].astype(float).mean()) if len(g) else float("nan")
                ),
                "count": float(len(g)),
            }
        )
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat(evaluation): per-lead skill vs baseline + PoP reliability metrics"
```

---

# Slice 2 — Training-data assembly + cache + split

### Task 2.1: Extract shared fakes to `tests/fakes.py` (DRY refactor)

The fake connectors currently live in `tests/features/conftest.py`; the assembly and smoke tests need them too. Move them to an importable module and re-export.

**Files:**
- Create: `tests/fakes.py`
- Modify: `tests/features/conftest.py`

- [ ] **Step 1: Create `tests/fakes.py`** — move the entire contents of `tests/features/conftest.py` (the `PHYS`/`PINNED` constants, `make_forecast_frame`, `make_obs_frame`, `FakeNWP`, `FakeObs`, `make_config`) into `tests/fakes.py` verbatim, keeping its module docstring.

- [ ] **Step 2: Replace `tests/features/conftest.py` with a re-export**

```python
# tests/features/conftest.py
"""Hermetic fixtures for feature tests — re-exported from the shared tests.fakes module."""

from __future__ import annotations

from tests.fakes import (  # noqa: F401
    PHYS,
    PINNED,
    FakeNWP,
    FakeObs,
    make_config,
    make_forecast_frame,
    make_obs_frame,
)
```

- [ ] **Step 3: Verify existing feature tests still pass unchanged**

Run: `uv run pytest tests/features -v`
Expected: PASS (same tests as before).

- [ ] **Step 4: Commit**

```bash
git add tests/fakes.py tests/features/conftest.py
git commit -m "test: extract shared fake connectors to tests/fakes.py"
```

### Task 2.2: `assemble_training_rows`

**Files:**
- Create: `src/microclimate/pipelines/training_data.py`
- Test: `tests/pipelines/test_training_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipelines/test_training_data.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.pipelines.training_data import assemble_training_rows
from tests.fakes import FakeNWP, FakeObs, make_config, make_forecast_frame, make_obs_frame

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
_LEADS = [1, 2, 3]


def _obs_window() -> list[datetime]:
    # lag window (t0, t0-1, t0-2) + future label window (t0+1..t0+3)
    return [
        _T0 - timedelta(hours=2),
        _T0 - timedelta(hours=1),
        _T0,
        _T0 + timedelta(hours=1),
        _T0 + timedelta(hours=2),
        _T0 + timedelta(hours=3),
    ]


def _sources() -> tuple[FakeNWP, dict[str, FakeObs]]:
    ts = _obs_window()
    obs = FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    return FakeNWP(make_forecast_frame(_T0, _LEADS)), {"fake": obs}


def test_assembles_labeled_rows_with_cardinality() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)  # threshold default 0.2
    nwp, obs = _sources()

    rows = assemble_training_rows(config, nwp, obs, [_T0])

    assert len(rows) == 3  # one row per lead
    assert list(rows["lead_hour"]) == [1, 2, 3]
    assert "label_temp_c" in rows.columns
    # PINNED precip 0.5 >= 0.2 -> all occurrence 1
    assert list(rows["label_precip_occurrence"].astype("Int64")) == [1, 1, 1]
    assert rows["label_temp_c"].iloc[0] == 15.0  # PINNED temp_c


def test_threshold_drives_occurrence() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    config = config.model_copy(update={"label": config.label.model_copy(update={"precip_occurrence_threshold_mm": 0.6})})
    nwp, obs = _sources()

    rows = assemble_training_rows(config, nwp, obs, [_T0])
    assert list(rows["label_precip_occurrence"].astype("Int64")) == [0, 0, 0]  # 0.5 < 0.6


def test_multiple_issue_times_concatenate() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    t1 = _T0 + timedelta(hours=6)
    ts = _obs_window() + [t1 + timedelta(hours=h) for h in (-2, -1, 0, 1, 2, 3)]
    obs = FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    nwp = FakeNWP(make_forecast_frame(_T0, _LEADS))  # frame shape reused per issue_time

    rows = assemble_training_rows(config, nwp, obs, [_T0, t1])
    assert len(rows) == 6
    assert set(rows["issue_time"]) == {pd.Timestamp(_T0), pd.Timestamp(t1)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipelines/test_training_data.py -v`
Expected: FAIL with `ModuleNotFoundError: microclimate.pipelines.training_data`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/microclimate/pipelines/training_data.py
"""Training-data assembly: the shared seam reused by the notebook and (later) the training
pipeline (L6).

Iterates issue-times through the shared build_snapshot -> build_features path, performs the
single training-only *future* read of target observations (values at valid_time), and
attaches labels. This future read is legal here (backfill/training) and categorically absent
from build_snapshot/build_features, preserving the ADR-0011 no-leakage guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.features.feature_builder import build_features
from microclimate.features.labeler import attach_labels
from microclimate.features.snapshot_builder import build_snapshot


def assemble_training_rows(
    config: DeploymentConfig,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    issue_times: Iterable[datetime],
) -> pd.DataFrame:
    """Build a labeled feature matrix spanning all given issue_times."""
    matrices: list[pd.DataFrame] = []
    for issue_time in issue_times:
        snapshot = build_snapshot(config, issue_time, nwp, observations)
        matrices.append(build_features(snapshot, config))
    if not matrices:
        raise ValueError("issue_times is empty; nothing to assemble")
    matrix = pd.concat(matrices, ignore_index=True)

    # Single batched future read of the target station across the whole valid-time span.
    target_source = observations[config.target.connector_key]
    start = matrix["valid_time"].min().to_pydatetime()
    end = matrix["valid_time"].max().to_pydatetime()
    target_obs = target_source.fetch_historical(config.target.station_id, start, end)

    return attach_labels(matrix, target_obs, config.label.precip_occurrence_threshold_mm)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipelines/test_training_data.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/pipelines/training_data.py tests/pipelines/test_training_data.py
git commit -m "feat(pipelines): assemble_training_rows — shared training-data assembly seam"
```

### Task 2.3: `assemble_or_load` (Parquet cache)

**Files:**
- Modify: `src/microclimate/pipelines/training_data.py`
- Test: `tests/pipelines/test_training_data.py`

- [ ] **Step 1: Add the failing test** (append to `tests/pipelines/test_training_data.py`)

```python
def test_assemble_or_load_uses_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from microclimate.connectors.base import SourceUnavailable
    from microclimate.pipelines.training_data import assemble_or_load

    config = make_config(horizon_hours=3, lag_hours=2)
    nwp, obs = _sources()
    cache = tmp_path / "rows.parquet"

    first = assemble_or_load(config, nwp, obs, [_T0], cache_path=cache)
    assert cache.exists()

    # Second call with exploding sources must still succeed -> proves it read the cache.
    boom_nwp = FakeNWP(exc=SourceUnavailable("should not be called"))
    boom_obs = {"fake": FakeObs(exc=SourceUnavailable("should not be called"))}
    second = assemble_or_load(config, boom_nwp, boom_obs, [_T0], cache_path=cache)

    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True), check_like=True
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipelines/test_training_data.py::test_assemble_or_load_uses_cache -v`
Expected: FAIL (`assemble_or_load` absent).

- [ ] **Step 3: Implement** (append to `training_data.py`)

```python
def assemble_or_load(
    config: DeploymentConfig,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    issue_times: Iterable[datetime],
    *,
    cache_path: Path,
) -> pd.DataFrame:
    """Read assembled rows from a local Parquet cache, else assemble and write it.

    Local-dev convenience so notebook re-runs don't re-pull CaSPAr. The cache is keyed by
    the caller's chosen path; rotate the path when the issue-time range or snapshot schema
    changes. Derived features are recomputed by build_features on read, so the derived
    FEATURE_SCHEMA_VERSION is intentionally NOT part of the key (ADR-0012).
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    rows = assemble_training_rows(config, nwp, observations, issue_times)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(cache_path, index=False)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipelines/test_training_data.py -v`
Expected: PASS (4 tests). (If `pyarrow` is missing, `to_parquet` will error — pandas pulls it via `pandera[pandas]`; if not, add `pyarrow` to dependencies in pyproject and `uv sync`.)

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/pipelines/training_data.py tests/pipelines/test_training_data.py
git commit -m "feat(pipelines): assemble_or_load — local Parquet cache for assembled rows"
```

### Task 2.4: `chronological_split`

**Files:**
- Modify: `src/microclimate/pipelines/training_data.py`
- Test: `tests/pipelines/test_training_data.py`

- [ ] **Step 1: Add the failing test**

```python
def test_chronological_split_by_issue_time() -> None:
    from microclimate.pipelines.training_data import chronological_split

    issue_times = [pd.Timestamp(_T0) + pd.Timedelta(hours=i) for i in range(10)]
    df = pd.DataFrame(
        {"issue_time": issue_times, "lead_hour": 1, "label_temp_c": 0.0}
    )

    train, calib, test = chronological_split(df, train_frac=0.6, calib_frac=0.2)

    assert list(train["issue_time"]) == issue_times[:6]
    assert list(calib["issue_time"]) == issue_times[6:8]
    assert list(test["issue_time"]) == issue_times[8:]
    # no issue_time leaks across splits
    assert set(train["issue_time"]) & set(test["issue_time"]) == set()
    assert set(train["issue_time"]) & set(calib["issue_time"]) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipelines/test_training_data.py::test_chronological_split_by_issue_time -v`
Expected: FAIL (`chronological_split` absent).

- [ ] **Step 3: Implement** (append to `training_data.py`; add `import numpy as np` at top)

```python
def chronological_split(
    rows: pd.DataFrame,
    *,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into chronological train | calib | test by *whole* issue_time.

    Splits on unique sorted issue_times (never across a single issue_time) so adjacent,
    strongly-correlated rows can't leak between sets. The remainder after train+calib is the
    test holdout. Temp trains on train+calib; PoP trains on train and calibrates on calib.
    """
    if train_frac + calib_frac >= 1.0:
        raise ValueError("train_frac + calib_frac must leave a non-empty test holdout")
    times = pd.Index(sorted(rows["issue_time"].unique()))
    n = len(times)
    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)
    # Guard the degenerate small-n case: int() truncation can zero out a slice (e.g. n<5
    # with the default calib_frac=0.2 → empty calib), which would later make the PoP
    # calibrator fit on no rows. Fail loudly here instead.
    if n_train == 0 or n_calib == 0 or n_train + n_calib >= n:
        raise ValueError(
            f"n={n} unique issue_times is too few to form non-empty train/calib/test "
            f"splits at train_frac={train_frac}, calib_frac={calib_frac}"
        )
    train_times = set(times[:n_train])
    calib_times = set(times[n_train : n_train + n_calib])
    test_times = set(times[n_train + n_calib :])
    return (
        rows[rows["issue_time"].isin(train_times)].copy(),
        rows[rows["issue_time"].isin(calib_times)].copy(),
        rows[rows["issue_time"].isin(test_times)].copy(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipelines/test_training_data.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/pipelines/training_data.py tests/pipelines/test_training_data.py
git commit -m "feat(pipelines): chronological_split — leakage-free train/calib/test by issue_time"
```

---

# Slice 3 — Model wrappers + row-based predict + ADR updates

### Task 3.0: Add scikit-learn dependency

**Files:** Modify `pyproject.toml`

- [ ] **Step 1: Add `scikit-learn` to `[project].dependencies`** (after `lightgbm>=4.3`):

```toml
  "lightgbm>=4.3",
  "scikit-learn>=1.4",
```

- [ ] **Step 2: Sync**

Run: `uv sync`
Expected: scikit-learn (and joblib) installed.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add scikit-learn (isotonic calibration + joblib persistence)"
```

### Task 3.1: Shared model-input column selection

**Files:**
- Create: `src/microclimate/models/_columns.py`
- Test: `tests/models/test_columns.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_columns.py
from __future__ import annotations

import pandas as pd

from microclimate.models._columns import NON_FEATURE_COLUMNS, feature_columns


def test_feature_columns_excludes_ids_times_labels_keeps_lead_hour() -> None:
    df = pd.DataFrame(
        columns=[
            "feature_schema_version",
            "deployment_id",
            "issue_time",
            "valid_time",
            "label_temp_c",
            "label_precip_occurrence",
            "lead_hour",
            "nwp_temp_c",
            "obs_T1_temp_c_lag0",
        ]
    )
    feats = feature_columns(df)
    assert "lead_hour" in feats
    assert "nwp_temp_c" in feats
    assert "obs_T1_temp_c_lag0" in feats
    assert NON_FEATURE_COLUMNS.isdisjoint(feats)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_columns.py -v`
Expected: FAIL (`microclimate.models._columns` absent).

- [ ] **Step 3: Implement**

```python
# src/microclimate/models/_columns.py
"""Model-input column selection shared by both wrappers (L4)."""

from __future__ import annotations

import pandas as pd

# Everything that is metadata/labels, not a model input. lead_hour IS a feature (ADR-0004).
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "feature_schema_version",
        "deployment_id",
        "issue_time",
        "valid_time",
        "label_temp_c",
        "label_precip_occurrence",
    }
)


def feature_columns(rows: pd.DataFrame) -> list[str]:
    """Ordered model-input columns: every column except metadata and labels."""
    return [c for c in rows.columns if c not in NON_FEATURE_COLUMNS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_columns.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/models/_columns.py tests/models/test_columns.py
git commit -m "feat(models): shared model-input column selection"
```

### Task 3.2: `TemperatureRegressor` (row-based)

**Files:**
- Modify: `src/microclimate/models/temp_model.py`
- Test: `tests/models/test_temp_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_temp_model.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.models.temp_model import TemperatureRegressor


def _rows(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    lead = rng.integers(1, 49, size=n)
    return pd.DataFrame(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "deployment_id": "test",
            "issue_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "valid_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "lead_hour": lead,
            "nwp_temp_c": x,
            "label_temp_c": 2.0 * x + 1.0,  # learnable signal
        }
    )


def test_fit_predict_returns_aligned_finite_series() -> None:
    rows = _rows()
    model = TemperatureRegressor()
    model.fit(rows)
    preds = model.predict(rows)
    assert len(preds) == len(rows)
    assert preds.index.equals(rows.index)
    assert np.isfinite(preds.to_numpy()).all()
    # learns the signal: beats predicting the mean
    mae_model = (preds - rows["label_temp_c"]).abs().mean()
    mae_mean = (rows["label_temp_c"].mean() - rows["label_temp_c"]).abs().mean()
    assert mae_model < mae_mean


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        TemperatureRegressor().predict(_rows(5))


def test_predict_rejects_mismatched_feature_version() -> None:
    rows = _rows()
    model = TemperatureRegressor()
    model.fit(rows)
    bad = rows.copy()
    bad["feature_schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="feature_schema_version"):
        model.predict(bad)


def test_save_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = _rows()
    model = TemperatureRegressor()
    model.fit(rows)
    before = model.predict(rows)
    path = tmp_path / "temp.joblib"
    model.save(path)
    reloaded = TemperatureRegressor.load(path)
    after = reloaded.predict(rows)
    pd.testing.assert_series_equal(before, after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_temp_model.py -v`
Expected: FAIL (stub raises `NotImplementedError`).

- [ ] **Step 3: Implement**

```python
# src/microclimate/models/temp_model.py
"""Temperature regressor wrapper (L4). Row-based: fit/predict over the feature matrix.

lead_hour is a feature (ADR-0004). predict takes feature-matrix rows and returns one
prediction per (issue_time, lead_hour); the inference pipeline owns build_features and
reshapes per-row predictions into the published {lead_hour: value} forecast (ADR-0012).
"""

from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore[import-untyped]
import lightgbm as lgb
import pandas as pd

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.models._columns import feature_columns


class TemperatureRegressor:
    version: str = "0.1.0"

    def __init__(self) -> None:
        self._model: lgb.LGBMRegressor | None = None
        self._features: list[str] | None = None
        self._feature_schema_version: str | None = None

    def fit(self, rows: pd.DataFrame) -> None:
        labeled = rows.dropna(subset=["label_temp_c"])
        feats = feature_columns(labeled)
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1
        )
        model.fit(labeled[feats], labeled["label_temp_c"])  # type: ignore[reportUnknownMemberType]
        self._model = model
        self._features = feats
        self._feature_schema_version = str(rows["feature_schema_version"].iloc[0])

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if self._model is None or self._features is None:
            raise RuntimeError("call fit() before predict()")
        got = str(rows["feature_schema_version"].iloc[0])
        if got != self._feature_schema_version:
            raise ValueError(
                f"rows feature_schema_version {got!r} != model's "
                f"{self._feature_schema_version!r}; refusing to predict."
            )
        preds = self._model.predict(rows[self._features])  # type: ignore[reportUnknownMemberType]
        return pd.Series(preds, index=rows.index, name="pred_temp_c")

    def save(self, path: Path) -> None:
        joblib.dump(  # type: ignore[reportUnknownMemberType]
            {
                "model": self._model,
                "features": self._features,
                "feature_schema_version": self._feature_schema_version,
                "version": self.version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TemperatureRegressor:
        state = joblib.load(path)  # type: ignore[reportUnknownMemberType]
        obj = cls()
        obj._model = state["model"]
        obj._features = state["features"]
        obj._feature_schema_version = state["feature_schema_version"]
        return obj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_temp_model.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run pyright** (confirm the type-ignores are correct)

Run: `uv run pyright`
Expected: 0 errors. If a `# type: ignore[...]` reports "unused", switch it to the rule pyright names; if a new unknown surfaces, add the rule pyright reports for that line.

- [ ] **Step 6: Commit**

```bash
git add src/microclimate/models/temp_model.py tests/models/test_temp_model.py
git commit -m "feat(models): TemperatureRegressor — row-based LightGBM regressor"
```

### Task 3.3: `PrecipOccurrenceClassifier` (row-based + calibration)

**Files:**
- Modify: `src/microclimate/models/pop_model.py`
- Test: `tests/models/test_pop_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_pop_model.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.models.pop_model import PrecipOccurrenceClassifier


def _rows(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-x))
    y = (rng.uniform(size=n) < prob).astype(int)  # both classes present
    return pd.DataFrame(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "deployment_id": "test",
            "issue_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "valid_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "lead_hour": rng.integers(1, 49, size=n),
            "nwp_precip_mm": x,
            "label_precip_occurrence": y,
        }
    )


def test_fit_calibrate_predict_in_unit_interval() -> None:
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    model.calibrate(rows)
    preds = model.predict(rows)
    assert len(preds) == len(rows)
    assert preds.index.equals(rows.index)
    arr = preds.to_numpy()
    assert ((arr >= 0.0) & (arr <= 1.0)).all()


def test_predict_requires_calibration() -> None:
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    with pytest.raises(RuntimeError, match="calibrate"):
        model.predict(rows)


def test_predict_rejects_mismatched_feature_version() -> None:
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    model.calibrate(rows)
    bad = rows.copy()
    bad["feature_schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="feature_schema_version"):
        model.predict(bad)


def test_save_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    model.calibrate(rows)
    before = model.predict(rows)
    path = tmp_path / "pop.joblib"
    model.save(path)
    after = PrecipOccurrenceClassifier.load(path).predict(rows)
    pd.testing.assert_series_equal(before, after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_pop_model.py -v`
Expected: FAIL (stub raises `NotImplementedError`).

- [ ] **Step 3: Implement**

```python
# src/microclimate/models/pop_model.py
"""Precipitation-occurrence classifier wrapper with isotonic calibration (L4).

Row-based like the temp model. Calibration (ADR-0004) is required: fit() learns the booster,
calibrate() fits an isotonic map on a disjoint slice, predict() returns the calibrated
probability per row. The fitted calibrator is persisted alongside the booster.
"""

from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore[import-untyped]
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.models._columns import feature_columns


class PrecipOccurrenceClassifier:
    version: str = "0.1.0"

    def __init__(self) -> None:
        self._model: lgb.LGBMClassifier | None = None
        self._calibrator: IsotonicRegression | None = None
        self._features: list[str] | None = None
        self._feature_schema_version: str | None = None

    def fit(self, rows: pd.DataFrame) -> None:
        labeled = rows.dropna(subset=["label_precip_occurrence"])
        feats = feature_columns(labeled)
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1
        )
        y = labeled["label_precip_occurrence"].astype(int)
        model.fit(labeled[feats], y)  # type: ignore[reportUnknownMemberType]
        self._model = model
        self._features = feats
        self._feature_schema_version = str(rows["feature_schema_version"].iloc[0])
        self._calibrator = None

    def calibrate(self, rows: pd.DataFrame) -> None:
        raw = self._raw_proba(rows)
        labeled_idx = rows["label_precip_occurrence"].notna()
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(  # type: ignore[reportUnknownMemberType]
            raw[labeled_idx.to_numpy()],
            rows.loc[labeled_idx, "label_precip_occurrence"].astype(int),
        )
        self._calibrator = calibrator

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if self._calibrator is None:
            raise RuntimeError("call calibrate() before predict()")
        raw = self._raw_proba(rows)
        calibrated = self._calibrator.predict(raw)  # type: ignore[reportUnknownMemberType]
        return pd.Series(np.clip(calibrated, 0.0, 1.0), index=rows.index, name="pred_pop")

    def _raw_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._features is None:
            raise RuntimeError("call fit() before calibrate()/predict()")
        got = str(rows["feature_schema_version"].iloc[0])
        if got != self._feature_schema_version:
            raise ValueError(
                f"rows feature_schema_version {got!r} != model's "
                f"{self._feature_schema_version!r}; refusing to predict."
            )
        proba = self._model.predict_proba(rows[self._features])  # type: ignore[reportUnknownMemberType]
        return np.asarray(proba)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(  # type: ignore[reportUnknownMemberType]
            {
                "model": self._model,
                "calibrator": self._calibrator,
                "features": self._features,
                "feature_schema_version": self._feature_schema_version,
                "version": self.version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> PrecipOccurrenceClassifier:
        state = joblib.load(path)  # type: ignore[reportUnknownMemberType]
        obj = cls()
        obj._model = state["model"]
        obj._calibrator = state["calibrator"]
        obj._features = state["features"]
        obj._feature_schema_version = state["feature_schema_version"]
        return obj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_pop_model.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Delete the superseded stub test + run pyright**

```bash
git rm tests/models/test_models_stub.py
uv run pyright
```
Expected: pyright 0 errors. Fix any reported `# type: ignore` rule mismatches as in Task 3.2 Step 5.

- [ ] **Step 6: Commit**

```bash
git add src/microclimate/models/pop_model.py tests/models/test_pop_model.py
git commit -m "feat(models): PrecipOccurrenceClassifier — row-based LightGBM + isotonic calibration"
```

### Task 3.4: ADRs + CONTEXT.md

**Files:**
- Create: `docs/adr/0013-notebook-model-dev-and-assembly.md`
- Modify: `docs/adr/0004-two-lightgbm-models.md`, `docs/adr/0012-feature-builder-read-time-transform.md`, `CONTEXT.md`

- [ ] **Step 1: Create ADR-0013**

```markdown
# 13. Local notebook is the model-dev surface; assembly is the shared seam

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0004 (two LightGBM models), ADR-0011 (snapshot normalization
  boundary), ADR-0012 (feature builder read-time transform), ADR-0009 (private raw store).

## Context

Model development needs a way to train the temp and PoP models locally and inspect them,
without that path forking from or bit-rotting against production. Reaching a trained model
also surfaced two feature-engineering steps ADR-0012 deferred: label attachment and
training-data assembly over a date range.

## Decision

1. **The notebook is a thin model-dev surface.** It holds no business logic — it calls
   shared, tested functions (`assemble_or_load`, the model wrappers, `evaluation.metrics`)
   and renders plots. A fast CI smoke test exercises that same assemble → fit → predict →
   metrics path on fake sources, so bitrot fails a test rather than waiting to be noticed.
2. **Training-data assembly is a shared seam** (`pipelines.training_data`). It performs the
   single training-only *future* read of target observations and labels the matrix; the
   future read is categorically absent from `build_snapshot`/`build_features` (ADR-0011).
   The same function the notebook calls will back the production training pipeline.
3. **`attach_labels` is pure and produces a labeled feature matrix** — distinct from the
   persisted `TRAINING_ROW` (raw snapshot + labels), which stays deferred with the private
   store. (`feature_matrix.py`'s docstring is corrected to stop conflating the two.)
4. **`predict` is row-based** (resolves ADR-0012's deferred open item; amends ADR-0004): the
   wrappers take feature-matrix rows and return one prediction per row; the inference
   pipeline owns `build_features` and reshapes to `{lead_hour: value}`.
5. **Evaluation uses a chronological three-way split** (`train | calib | test`): temp trains
   on `train+calib`; PoP trains on `train` and fits its isotonic calibrator on the disjoint
   `calib` slice; both are judged on `test` against the raw-HRDPS baseline, per lead hour.
6. **scikit-learn is adopted** for isotonic calibration and joblib model persistence.

## Consequences

- Deferred: the private training-store read/write path, the production training-pipeline
  orchestration / publish gate / publication, and walk-forward CV.
- Locally-trained models are throwaway (gitignored); promotion to a registry is future work.
- The notebook is authored as a jupytext percent-format `.py` (clean diffs, openable as a
  notebook); generated `.ipynb` files are gitignored.
```

- [ ] **Step 2: Amend ADR-0004** — append to its Consequences list:

```markdown
- **`predict` is row-based (added 2026-05-31, see ADR-0013):** the wrappers take feature-
  matrix rows and return one prediction per `(issue_time, lead_hour)` row; the inference
  pipeline reshapes per-row predictions into the published `{lead_hour: value}` forecast.
```

- [ ] **Step 3: Amend ADR-0012** — replace its last Consequence bullet (the `model.predict`
  open item) with:

```markdown
- **Resolved (ADR-0013):** `model.predict` is row-based and the pipeline owns the
  `build_features` call, passing rows to both `fit` and `predict`. A label-attachment step
  (`features.attach_labels`, pure) and a training-data assembler (`pipelines.training_data`)
  now produce the labeled feature matrix; the persisted `TRAINING_ROW` store remains deferred.
```

- [ ] **Step 4: Update CONTEXT.md** — add these bullets under the matching sections (place
  near the existing related terms; the section names already exist in CONTEXT.md):

Under **Features** (near "Feature matrix"):
```markdown
- **Labeled feature matrix** — the feature matrix with `label_temp_c` and
  `label_precip_occurrence` attached (`features.attach_labels`). What the models train on;
  distinct from the persisted **Training store** schema (raw snapshot + labels, ADR-0012).
- **Label attachment** / **labeler** — the pure step joining target-station observations at
  `valid_time` onto the feature matrix to form the labeled feature matrix. The future
  (post-`issue_time`) read it depends on is done by training-data assembly, never inference.
```

Under **Pipelines** (near "Training pipeline" / "Training store"):
```markdown
- **Training-data assembly** — `pipelines.training_data`: iterates issue-times through
  `build_snapshot` → `build_features`, performs the single training-only future read of
  target observations, and labels the result. The shared seam used by the model-dev notebook
  and (later) the training pipeline; caches assembled rows to local Parquet.
- **Model-dev notebook** — the thin, local-only `notebooks/model_dev.py` for training and
  exploring models. Holds no logic; calls the shared assembly, model, and metric functions.
```

Under **Modeling & quality** (near "Calibration"):
```markdown
- **Calibration slice** — the disjoint chronological slice between train and test on which
  the PoP isotonic calibrator is fit, so calibration is not fit on overconfident in-sample
  predictions.
```

- [ ] **Step 5: Commit**

```bash
git add docs/adr/0013-notebook-model-dev-and-assembly.md docs/adr/0004-two-lightgbm-models.md docs/adr/0012-feature-builder-read-time-transform.md CONTEXT.md
git commit -m "docs: ADR-0013 + amend ADR-0004/0012; CONTEXT terms for labeler/assembly/calibration slice"
```

---

# Slice 4 — Notebook + dependency group + CI smoke + status

### Task 4.1: `notebook` dependency group + gitignore

**Files:** Modify `pyproject.toml`, `.gitignore`

- [ ] **Step 1: Add the notebook dependency group** to `pyproject.toml` under `[dependency-groups]`:

```toml
notebook = [
  "jupyter>=1.0",
  "jupytext>=1.16",
  "matplotlib>=3.8",
]
```

- [ ] **Step 2: Ignore notebook artifacts** — append to `.gitignore`:

```gitignore
# Notebook (local model-dev)
notebooks/_artifacts/
notebooks/*.ipynb
```

- [ ] **Step 3: Sync + commit**

```bash
uv sync --group notebook
git add pyproject.toml uv.lock .gitignore
git commit -m "build: add notebook dependency group; ignore notebook artifacts"
```

### Task 4.2: CI smoke test (guards the whole path)

**Files:** Create `tests/pipelines/test_smoke_model_dev.py`

- [ ] **Step 1: Write the smoke test** (it must fail first because it asserts on real output)

```python
# tests/pipelines/test_smoke_model_dev.py
"""Fast smoke of the model-dev path: assemble -> split -> fit -> predict -> metrics.

Uses fake sources (no network). Exercises the SAME shared functions the notebook calls, so
notebook bitrot surfaces here without executing the .ipynb.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, ObservationSource
from microclimate.evaluation.metrics import (
    pop_skill_by_lead,
    reliability_table,
    temp_skill_by_lead,
)
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.pipelines.training_data import assemble_training_rows, chronological_split
from tests.fakes import PINNED, PHYS, FakeNWP, make_config, make_forecast_frame

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
_LEADS = [1, 2, 3]


def _varying_obs_frame(station_id: str, timestamps: list[datetime]) -> pd.DataFrame:
    """OBSERVATION_FRAME with precip alternating by hour parity so PoP has both classes."""
    ts = pd.to_datetime(timestamps, utc=True)
    data: dict[str, object] = {"station_id": [station_id] * len(ts), "timestamp": list(ts)}
    for var in PHYS:
        data[var] = [PINNED[var]] * len(ts)
        data[f"{var}_present"] = [True] * len(ts)
    data["precip_mm"] = [0.5 if t.hour % 2 == 0 else 0.0 for t in ts]
    return pd.DataFrame(data)


class _VaryingObs(ObservationSource):
    def __init__(self, station_ids: list[str], timestamps: list[datetime]) -> None:
        self._frames = {s: _varying_obs_frame(s, timestamps) for s in station_ids}

    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._frames[station_id]

    def fetch_live(self, station_id: str, since: datetime):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def test_model_dev_path_runs_end_to_end() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    issue_times = [_T0 + timedelta(hours=i) for i in range(40)]
    span = [_T0 - timedelta(hours=2) + timedelta(hours=i) for i in range(40 + 3 + 3)]
    obs = {"fake": _VaryingObs(["T1", "N1"], span)}
    nwp = FakeNWP(make_forecast_frame(_T0, _LEADS))

    rows = assemble_training_rows(config, nwp, obs, issue_times)
    assert len(rows) == 40 * 3

    train, calib, test = chronological_split(rows)
    assert len(train) and len(calib) and len(test)

    temp = TemperatureRegressor()
    temp.fit(pd.concat([train, calib], ignore_index=True))
    test = test.copy()
    test["pred_temp_c"] = temp.predict(test).to_numpy()

    pop = PrecipOccurrenceClassifier()
    pop.fit(train)
    pop.calibrate(calib)
    test["pred_pop"] = pop.predict(test).to_numpy()
    test["baseline_pop"] = (test["nwp_precip_mm"] >= config.label.precip_occurrence_threshold_mm).astype(float)

    temp_skill = temp_skill_by_lead(test)
    pop_skill = pop_skill_by_lead(test)
    rel = reliability_table(test)

    assert {"lead_hour", "rmse", "skill"}.issubset(temp_skill.columns)
    assert {"lead_hour", "brier", "bss"}.issubset(pop_skill.columns)
    assert len(rel) == 10
    assert np.isfinite(test["pred_temp_c"].to_numpy()).all()
    assert ((test["pred_pop"] >= 0) & (test["pred_pop"] <= 1)).all()
```

- [ ] **Step 2: Run test to verify it passes** (the modules exist by now; it should pass)

Run: `uv run pytest tests/pipelines/test_smoke_model_dev.py -v`
Expected: PASS. If `nwp_precip_mm` is absent, confirm `make_forecast_frame` includes all `PHYS` vars (it does) and `build_features` emits `nwp_precip_mm`.

- [ ] **Step 3: Commit**

```bash
git add tests/pipelines/test_smoke_model_dev.py
git commit -m "test(pipelines): CI smoke for the model-dev assemble->fit->predict->metrics path"
```

### Task 4.3: The thin notebook

**Files:** Create `notebooks/model_dev.py`

- [ ] **Step 1: Write the notebook (jupytext percent format)**

```python
# notebooks/model_dev.py
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Local model development — train & explore the temp and PoP models
#
# Thin notebook: all logic lives in `microclimate.*` (tested) and is exercised by the CI
# smoke test. This file only orchestrates and plots. Open it as a notebook with jupytext or
# VS Code. Requires the `notebook` dependency group: `uv sync --group notebook`.
# CaSPAr historical access must be configured for the chosen deployment.

# %%
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.connectors.registry import get_source
from microclimate.evaluation.metrics import (
    pop_skill_by_lead,
    reliability_table,
    temp_skill_by_lead,
)
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.pipelines.training_data import assemble_or_load, chronological_split

DEPLOYMENT_ID = "lethbridge"
START = datetime(2024, 1, 1, 0, tzinfo=UTC)
N_ISSUE_TIMES = 24 * 60  # ~60 days of hourly issue times
ARTIFACTS = Path("notebooks/_artifacts")

# %%
config = load_deployment(DEPLOYMENT_ID)
nwp = get_source(config.nwp.historical_connector)
station_keys = {config.target.connector_key, *[n.connector_key for n in config.neighbors]}
observations = {k: get_source(k) for k in station_keys}  # type: ignore[misc]
issue_times = [START + timedelta(hours=i) for i in range(N_ISSUE_TIMES)]

# %%
rows = assemble_or_load(
    config, nwp, observations, issue_times,  # type: ignore[arg-type]
    cache_path=ARTIFACTS / f"{DEPLOYMENT_ID}_rows.parquet",
)
print(f"{len(rows):,} rows  |  {rows['issue_time'].nunique()} issue times")
rows.head()

# %%
train, calib, test = chronological_split(rows, train_frac=0.6, calib_frac=0.2)
print(f"train={len(train):,}  calib={len(calib):,}  test={len(test):,}")

# %%
temp = TemperatureRegressor()
temp.fit(pd.concat([train, calib], ignore_index=True))

pop = PrecipOccurrenceClassifier()
pop.fit(train)
pop.calibrate(calib)

test = test.copy()
test["pred_temp_c"] = temp.predict(test).to_numpy()
test["pred_pop"] = pop.predict(test).to_numpy()
test["baseline_pop"] = (
    test["nwp_precip_mm"] >= config.label.precip_occurrence_threshold_mm
).astype(float)

# %% [markdown]
# ## Temperature: skill vs raw-HRDPS baseline, by lead hour

# %%
ts = temp_skill_by_lead(test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(ts["lead_hour"], ts["rmse"], label="model RMSE")
ax1.plot(ts["lead_hour"], ts["baseline_rmse"], label="HRDPS RMSE")
ax1.set_xlabel("lead hour"); ax1.set_ylabel("°C"); ax1.legend(); ax1.set_title("RMSE")
ax2.axhline(0, color="grey", lw=0.8)
ax2.plot(ts["lead_hour"], ts["skill"])
ax2.set_xlabel("lead hour"); ax2.set_title("RMSE skill (>0 beats HRDPS)")
plt.tight_layout()

# %% [markdown]
# ## Temperature: predicted vs actual & residuals

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(test["label_temp_c"], test["pred_temp_c"], s=4, alpha=0.3)
lims = [test["label_temp_c"].min(), test["label_temp_c"].max()]
ax1.plot(lims, lims, color="red", lw=1)
ax1.set_xlabel("observed °C"); ax1.set_ylabel("predicted °C"); ax1.set_title("pred vs actual")
ax2.scatter(test["pred_temp_c"], test["pred_temp_c"] - test["label_temp_c"], s=4, alpha=0.3)
ax2.axhline(0, color="red", lw=1)
ax2.set_xlabel("predicted °C"); ax2.set_ylabel("residual °C"); ax2.set_title("residuals")
plt.tight_layout()

# %% [markdown]
# ## PoP: Brier Skill Score by lead, and the reliability diagram

# %%
ps = pop_skill_by_lead(test)
rel = reliability_table(test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.axhline(0, color="grey", lw=0.8)
ax1.plot(ps["lead_hour"], ps["bss"])
ax1.set_xlabel("lead hour"); ax1.set_title("Brier Skill Score (>0 beats HRDPS)")
ax2.plot([0, 1], [0, 1], color="grey", lw=1, label="perfect")
ax2.plot(rel["mean_pred"], rel["observed_freq"], marker="o", label="model")
ax2.set_xlabel("predicted PoP"); ax2.set_ylabel("observed frequency")
ax2.set_title("reliability"); ax2.legend()
plt.tight_layout()

# %% [markdown]
# ## Feature importances

# %%
imp = pd.Series(
    temp._model.feature_importances_, index=temp._features  # noqa: SLF001
).sort_values(ascending=False).head(20)
imp.iloc[::-1].plot.barh(figsize=(8, 6), title="Temp model — top feature importances")
plt.tight_layout()

# %% [markdown]
# ## Save the locally-trained models (gitignored)

# %%
ARTIFACTS.mkdir(parents=True, exist_ok=True)
temp.save(ARTIFACTS / f"{DEPLOYMENT_ID}_temp.joblib")
pop.save(ARTIFACTS / f"{DEPLOYMENT_ID}_pop.joblib")
print("saved to", ARTIFACTS)
```

- [ ] **Step 2: Lint the notebook source** (it is a `.py`, so ruff checks it)

Run: `uv run ruff check notebooks/model_dev.py`
Expected: clean. (The `temp._model` / `temp._features` access in the importances cell is
intentional notebook introspection; the `# noqa: SLF001` covers private-access lint. If ruff
flags unused imports because a cell was trimmed, remove them.)

- [ ] **Step 3: Optional local check** — confirm it converts and opens as a notebook:

Run: `uv run --group notebook jupytext --to notebook notebooks/model_dev.py -o /tmp/model_dev.ipynb`
Expected: a valid `.ipynb` is produced (not committed; `notebooks/*.ipynb` is gitignored).

- [ ] **Step 4: Commit**

```bash
git add notebooks/model_dev.py
git commit -m "feat(notebooks): thin model-dev notebook (train + explore, jupytext percent)"
```

### Task 4.4: Update README Project status

**Files:** Modify `README.md`

- [ ] **Step 1: Replace the "Not yet implemented" paragraph** of the Project status section
  (and extend the implemented paragraph) to reflect this work:

```markdown
Additionally implemented: `features.attach_labels` (the pure label-attachment step →
labeled feature matrix), `pipelines.training_data` (training-data assembly + local Parquet
cache + chronological split — the shared seam, ADR-0013), the two **LightGBM model
wrappers** (`models.TemperatureRegressor` and `models.PrecipOccurrenceClassifier` with
isotonic calibration, row-based `predict`), `evaluation.metrics` (per-lead skill vs the
raw-HRDPS baseline + PoP reliability), and a thin local **model-dev notebook**
(`notebooks/model_dev.py`).

**Not yet implemented** (currently stubs): the publish gate, forecast-JSON / registry
publication, the inference/training pipeline orchestration CLIs, and the private
training-store read/write path (local dev backfills from CaSPAr historical instead).
```

- [ ] **Step 2: Full gate** (final, whole suite)

Run: `uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update Project status for labeler/assembly/models/notebook"
```

---

## Final integration

- [ ] Push the branch and open a PR (main rejects direct pushes):

```bash
git push -u origin spec/notebook-model-training
gh pr create --fill --base main
```

- [ ] Confirm CI is green on the PR, then merge per the project's PR workflow.

---

## Self-review notes (already reconciled in this plan)

- **Schema-version reconciliation (spec open item):** resolved by making `attach_labels`
  produce a *labeled feature matrix* (FEATURE_ROW + 2 label columns), explicitly **not**
  `TRAINING_ROW`. The persisted `TRAINING_ROW` store stays deferred; the `feature_matrix.py`
  docstring is corrected (Task 1.1 Step 6) and the distinction recorded in ADR-0013.
- **Cache granularity (spec open item):** `assemble_or_load` caches **assembled rows** (what
  the notebook needs), keyed by the caller's path; documented in the docstring (Task 2.3).
- **Type names consistent across tasks:** `feature_columns`/`NON_FEATURE_COLUMNS` (Task 3.1)
  used by both models; `pred_temp_c`/`pred_pop`/`baseline_pop` column names match between the
  models, `evaluation.metrics` defaults, the smoke test, and the notebook.
- **predict is row-based everywhere** (`predict(rows) -> Series`); no `predict(snapshot)`
  remains. The `models/__init__.py` exports are unchanged (class names identical).
```
