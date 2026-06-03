# Training Pipeline (backfill → train → gate → promote → publish) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full monthly retrain pipeline — pull the seed into the persistent store, train temp + PoP, run the champion/challenger publish gate on a most-recent-12-months holdout, and on promotion publish the champion binary (GitHub Release) + `registry.json` (gh-pages).

**Architecture:** A months-based temporal split feeds two LightGBM models; a pure publish gate promotes a task only if the challenger strictly beats both raw HRDPS and the current champion on the holdout; `registry_store` + a champion-publisher persist the decision locally; `run_training` orchestrates (reusing the tested `backfill_store`/`assemble_from_store`); `training.yml` syncs the store branch and does the external publish via `gh`.

**Tech Stack:** Python 3.12, pandas, LightGBM (existing model wrappers), pydantic (registry contracts), pytest, `uv`, GitHub Actions + `gh` CLI.

**Authoritative docs:** `docs/superpowers/specs/2026-06-03-training-pipeline-publish-gate-design.md`; ADR-0006/0016/0017/0018/0019; `CONTEXT.md`.

**Conventions:** UTC everywhere; injected deps for hermetic tests; run the FULL gate before every commit: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`.

**Key existing APIs (verified):**
- `TemperatureRegressor` / `PrecipOccurrenceClassifier` (`microclimate.models.{temp_model,pop_model}`): `version="0.1.0"`, `fit(rows)`, `pop.calibrate(rows)`, `predict(rows)->pd.Series` (named `pred_temp_c`/`pred_pop`; raises if `rows.feature_schema_version` ≠ model's), `save(Path)`, `load(Path)`.
- `microclimate.evaluation.metrics`: `temp_skill_by_lead(df, baseline_col="nwp_temp_c")`, `nwp_pop_baseline(df, threshold, precip_col="nwp_precip_mm")->pd.Series`, `pop_skill_by_lead(df, baseline_col="baseline_pop")`.
- Feature-matrix rows carry: `issue_time, lead_hour, feature_schema_version, nwp_temp_c, nwp_precip_mm, label_temp_c, label_precip_occurrence` + feature cols.
- `microclimate.contracts.registry`: `Task=Literal["temp","pop"]`, `manifest_key(dep,task)`, `RegistryEntry{version, release_asset_url, promoted_at, holdout_metrics}`, `RegistryManifest{entries: dict[str,RegistryEntry]}`.
- `microclimate.pipelines.backfill`: `hrdps_issue_times(start,end)`, `backfill_store(config,*,nwp,observations,store,issue_times,pause_s=...)`.
- `microclimate.pipelines.training_data`: `assemble_from_store(config, store)`.
- `microclimate.connectors.http.http_get_bytes(url,*,params=None)->bytes`.
- `microclimate.connectors.registry.get_source`, `validate_config_sources`.

---

## File Structure

- Modify: `src/microclimate/pipelines/training_data.py` — add `temporal_split`.
- Modify: `src/microclimate/evaluation/publish_gate.py` — implement `evaluate_challenger`.
- Modify: `src/microclimate/publication/registry_store.py` — implement `read_registry`, `promote`, add `write_registry`.
- Create: `src/microclimate/publication/champion_publisher.py` — version/tag/url/filename + `save_champion`.
- Modify: `src/microclimate/pipelines/training.py` — `load_champion` + `run_training` orchestration + `main`.
- Modify: `.github/workflows/training.yml` — store-sync + run + publish steps.
- Modify: `docs/adr/0016-baseline-champion-pre-model-publishing.md`, `README.md` — status updates.
- Tests: `tests/pipelines/test_temporal_split.py`, `tests/evaluation/test_publish_gate.py`, `tests/publication/test_registry_store.py`, `tests/publication/test_champion_publisher.py`, `tests/pipelines/test_training.py`.

---

## Task 1: Months-based temporal split

**Files:**
- Modify: `src/microclimate/pipelines/training_data.py`
- Test: `tests/pipelines/test_temporal_split.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipelines/test_temporal_split.py`:
```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest


def _rows(n_issue: int, step_h: int = 6, leads: int = 2) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    recs = []
    for i in range(n_issue):
        it = start + timedelta(hours=step_h * i)
        for lh in range(1, leads + 1):
            recs.append({"issue_time": pd.Timestamp(it), "lead_hour": lh})
    return pd.DataFrame(recs)


def test_temporal_split_test_is_recent_holdout_calib_disjoint() -> None:
    from microclimate.pipelines.training_data import temporal_split

    # ~12 months of 6-hourly issue times so all three slices are non-empty.
    rows = _rows(n_issue=4 * 365)
    train, calib, test = temporal_split(rows, holdout_months=3, calib_months=1)

    last = rows["issue_time"].max()
    test_lo = test["issue_time"].min()
    calib_hi, calib_lo = calib["issue_time"].max(), calib["issue_time"].min()
    # test is the most recent window; calib sits strictly before test; train before calib.
    assert test["issue_time"].max() == last
    assert calib_hi < test_lo
    assert train["issue_time"].max() < calib_lo
    # disjoint issue_times across the three sets
    its = lambda d: set(d["issue_time"].unique())  # noqa: E731
    assert its(train).isdisjoint(its(calib)) and its(calib).isdisjoint(its(test))
    assert len(train) and len(calib) and len(test)


def test_temporal_split_raises_when_a_slice_is_empty() -> None:
    from microclimate.pipelines.training_data import temporal_split

    rows = _rows(n_issue=4)  # ~1 day — far too little for a 3-month holdout + calib
    with pytest.raises(ValueError, match="too little history|empty"):
        temporal_split(rows, holdout_months=3, calib_months=1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipelines/test_temporal_split.py -v`
Expected: FAIL — `ImportError` (`temporal_split` not defined).

- [ ] **Step 3: Implement `temporal_split`** (append to `src/microclimate/pipelines/training_data.py`)

```python
def temporal_split(
    rows: pd.DataFrame,
    *,
    holdout_months: int,
    calib_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by whole issue_time into train | calib | test using calendar-month cutoffs.

    test  = issue_time within the most recent ``holdout_months`` (the evaluation holdout).
    calib = the ``calib_months`` immediately before test (disjoint PoP calibration slice).
    train = everything before calib.
    Raises ValueError if any slice is empty (too little history).
    """
    if rows.empty:
        raise ValueError("rows is empty; nothing to split")
    issue = pd.to_datetime(rows["issue_time"], utc=True)
    last = issue.max()
    test_cut = last - pd.DateOffset(months=holdout_months)
    calib_cut = test_cut - pd.DateOffset(months=calib_months)

    train = rows[issue <= calib_cut]
    calib = rows[(issue > calib_cut) & (issue <= test_cut)]
    test = rows[issue > test_cut]
    if train.empty or calib.empty or test.empty:
        raise ValueError(
            f"too little history for holdout_months={holdout_months}, "
            f"calib_months={calib_months}: train={len(train)} calib={len(calib)} test={len(test)}"
        )
    return train.copy(), calib.copy(), test.copy()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipelines/test_temporal_split.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Full gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
```bash
git add src/microclimate/pipelines/training_data.py tests/pipelines/test_temporal_split.py
git commit -m "feat(training-data): months-based temporal_split (recent holdout + disjoint calib)"
```

---

## Task 2: Publish gate

**Files:**
- Modify: `src/microclimate/evaluation/publish_gate.py`
- Test: `tests/evaluation/test_publish_gate.py`

The gate predicts via duck-typed models and compares overall holdout error (temp → MAE, pop → Brier); promote only if the challenger is strictly lower than **both** the baseline and the champion. `champion=None` means the current champion is the baseline (champion error = baseline error).

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_publish_gate.py`:
```python
from __future__ import annotations

import pandas as pd

from microclimate.evaluation.publish_gate import evaluate_challenger


class _ConstTemp:
    """Model stub: predicts a constant temperature for every row."""

    def __init__(self, value: float) -> None:
        self._v = value

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series([self._v] * len(rows), index=rows.index, name="pred_temp_c")


def _temp_holdout() -> tuple[pd.DataFrame, pd.Series]:
    # label = 10 everywhere; baseline (raw HRDPS) predicts 12 (MAE 2.0).
    holdout = pd.DataFrame(
        {"lead_hour": [1, 2, 3, 4], "label_temp_c": [10.0, 10.0, 10.0, 10.0]}
    )
    baseline = pd.Series([12.0, 12.0, 12.0, 12.0], index=holdout.index)
    return holdout, baseline


def test_temp_promotes_when_strictly_beats_baseline_and_champion() -> None:
    holdout, baseline = _temp_holdout()
    challenger = _ConstTemp(10.5)  # MAE 0.5
    champion = _ConstTemp(11.0)  # MAE 1.0
    res = evaluate_challenger("temp", challenger, champion, baseline, holdout)
    assert res.promote is True
    assert res.metrics["mae"] < res.metrics["champion_mae"] < res.metrics["baseline_mae"]


def test_temp_no_promote_when_worse_than_champion() -> None:
    holdout, baseline = _temp_holdout()
    challenger = _ConstTemp(11.5)  # MAE 1.5 — beats baseline (2.0) but not champion (1.0)
    champion = _ConstTemp(11.0)  # MAE 1.0
    res = evaluate_challenger("temp", challenger, champion, baseline, holdout)
    assert res.promote is False


def test_temp_no_promote_on_tie() -> None:
    holdout, baseline = _temp_holdout()
    challenger = _ConstTemp(12.0)  # MAE 2.0 == baseline
    res = evaluate_challenger("temp", challenger, None, baseline, holdout)
    assert res.promote is False


def test_temp_promotes_off_baseline_when_champion_none() -> None:
    holdout, baseline = _temp_holdout()
    challenger = _ConstTemp(10.0)  # MAE 0.0
    res = evaluate_challenger("temp", challenger, None, baseline, holdout)
    assert res.promote is True


class _ConstPop:
    def __init__(self, p: float) -> None:
        self._p = p

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series([self._p] * len(rows), index=rows.index, name="pred_pop")


def test_pop_promotes_when_brier_beats_baseline_and_champion() -> None:
    holdout = pd.DataFrame(
        {"lead_hour": [1, 2, 3, 4], "label_precip_occurrence": [1, 0, 1, 0]}
    )
    baseline = pd.Series([0.5, 0.5, 0.5, 0.5], index=holdout.index)  # Brier 0.25
    challenger = _ConstPop(0.6)  # closer on the 1s → lower Brier than 0.5? compute: ((.6-1)^2*2+(.6-0)^2*2)/4 = (0.32+0.72)/4=0.26 -> NOT better
    res = evaluate_challenger("pop", challenger, None, baseline, holdout)
    # 0.26 > 0.25 → no promote (guards against an over-eager gate)
    assert res.promote is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_publish_gate.py -v`
Expected: FAIL — `evaluate_challenger` raises `NotImplementedError`.

- [ ] **Step 3: Implement the gate** (replace the body in `src/microclimate/evaluation/publish_gate.py`)

```python
"""Champion/challenger publish gate (L4). Imports no model classes (independence)."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from microclimate.contracts.registry import Task


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promote: bool
    reason: str
    metrics: dict[str, float]


def _error(task: Task, pred: pd.Series, holdout: pd.DataFrame) -> float:
    """Overall holdout error for the task: temp -> MAE, pop -> Brier (lower is better)."""
    if task == "temp":
        label = holdout["label_temp_c"]
        keep = label.notna() & pred.notna()
        return float((pred[keep] - label[keep]).abs().mean())
    label = holdout["label_precip_occurrence"].astype("float64")
    keep = label.notna() & pred.notna()
    return float(((pred[keep] - label[keep]) ** 2).mean())


def evaluate_challenger(
    task: Task,
    challenger: object,
    champion: object | None,
    baseline: pd.Series,
    holdout: pd.DataFrame,
) -> GateResult:
    """Promote only if the challenger strictly beats both raw HRDPS and the incumbent.

    ``challenger``/``champion`` are fitted models exposing ``.predict(holdout) -> pd.Series``;
    ``champion`` is None when the current champion is the baseline. ``baseline`` is the
    raw-HRDPS prediction per holdout row (caller-supplied). Lower error wins (temp MAE, pop Brier).
    """
    challenger_pred = challenger.predict(holdout)  # type: ignore[attr-defined]
    champion_pred = champion.predict(holdout) if champion is not None else baseline  # type: ignore[attr-defined]

    m_chal = _error(task, challenger_pred, holdout)
    m_base = _error(task, baseline, holdout)
    m_champ = _error(task, champion_pred, holdout)

    key = "mae" if task == "temp" else "brier"
    metrics = {key: m_chal, f"baseline_{key}": m_base, f"champion_{key}": m_champ}
    # skill vs HRDPS for visibility (1 - err/baseline_err; lower err -> higher skill)
    metrics["skill" if task == "temp" else "bss"] = (
        1.0 - m_chal / m_base if m_base > 0 else float("nan")
    )

    promote = m_chal < m_base and m_chal < m_champ
    reason = (
        f"{key}={m_chal:.4f} vs baseline={m_base:.4f}, champion={m_champ:.4f} -> "
        + ("PROMOTE" if promote else "keep champion")
    )
    return GateResult(promote=promote, reason=reason, metrics=metrics)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evaluation/test_publish_gate.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Full gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
```bash
git add src/microclimate/evaluation/publish_gate.py tests/evaluation/test_publish_gate.py
git commit -m "feat(publish-gate): strictly-beats-both champion/challenger gate (temp MAE / pop Brier)"
```

---

## Task 3: Registry store

**Files:**
- Modify: `src/microclimate/publication/registry_store.py`
- Test: `tests/publication/test_registry_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/publication/test_registry_store.py`:
```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, manifest_key
from microclimate.publication.registry_store import promote, read_registry, write_registry


def _entry(v: str) -> RegistryEntry:
    return RegistryEntry(
        version=v,
        release_asset_url=f"https://example/{v}.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={"mae": 1.0},
    )


def test_read_missing_returns_empty_manifest(tmp_path: Path) -> None:
    assert read_registry(tmp_path / "nope.json").entries == {}


def test_promote_then_roundtrip(tmp_path: Path) -> None:
    m = RegistryManifest()
    m2 = promote(m, "temp", "lethbridge", _entry("v1"))
    # immutability: original unchanged
    assert m.entries == {}
    assert m2.entries[manifest_key("lethbridge", "temp")].version == "v1"

    path = tmp_path / "registry.json"
    write_registry(m2, path)
    back = read_registry(path)
    assert back.entries[manifest_key("lethbridge", "temp")].version == "v1"

    # promoting another task keeps the first
    m3 = promote(back, "pop", "lethbridge", _entry("p1"))
    assert set(m3.entries) == {manifest_key("lethbridge", "temp"), manifest_key("lethbridge", "pop")}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/publication/test_registry_store.py -v`
Expected: FAIL — `read_registry`/`promote` raise `NotImplementedError`; `write_registry` undefined.

- [ ] **Step 3: Implement** (replace the body of `src/microclimate/publication/registry_store.py`)

```python
"""Read/update the champion registry manifest (L5)."""

from __future__ import annotations

from pathlib import Path

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, Task, manifest_key


def read_registry(path: Path) -> RegistryManifest:
    """Parse registry.json, or an empty manifest if the file is absent."""
    if not path.exists():
        return RegistryManifest()
    return RegistryManifest.model_validate_json(path.read_text())


def promote(
    manifest: RegistryManifest, task: Task, deployment_id: str, entry: RegistryEntry
) -> RegistryManifest:
    """Return a new manifest with the (deployment_id, task) entry set (immutable update)."""
    entries = dict(manifest.entries)
    entries[manifest_key(deployment_id, task)] = entry
    return RegistryManifest(entries=entries)


def write_registry(manifest: RegistryManifest, path: Path) -> None:
    """Serialize the manifest to registry.json (pretty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/publication/test_registry_store.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Full gate + commit**

```bash
git add src/microclimate/publication/registry_store.py tests/publication/test_registry_store.py
git commit -m "feat(registry-store): read/promote/write registry.json"
```

---

## Task 4: Champion publisher (deterministic version/URL + save)

**Files:**
- Create: `src/microclimate/publication/champion_publisher.py`
- Test: `tests/publication/test_champion_publisher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/publication/test_champion_publisher.py`:
```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from microclimate.models.temp_model import TemperatureRegressor
from microclimate.publication.champion_publisher import (
    asset_filename,
    champion_version,
    release_asset_url,
    release_tag,
    save_champion,
)


def test_version_and_url_are_deterministic() -> None:
    t = datetime(2026, 6, 3, 14, 5, tzinfo=UTC)
    v = champion_version("lethbridge", "temp", t)
    assert v == "lethbridge-temp-20260603T1405Z"
    assert release_tag(v) == "champion-lethbridge-temp-20260603T1405Z"
    assert asset_filename(v) == "lethbridge-temp-20260603T1405Z.joblib"
    url = release_asset_url("gnieuwenhuis/microclimate_forecasting", v)
    assert url == (
        "https://github.com/gnieuwenhuis/microclimate_forecasting/releases/download/"
        "champion-lethbridge-temp-20260603T1405Z/lethbridge-temp-20260603T1405Z.joblib"
    )


def test_save_champion_writes_loadable_file(tmp_path: Path) -> None:
    import pandas as pd

    # minimal fitted temp model (one feature col + label + schema version)
    rows = pd.DataFrame(
        {
            "feature_schema_version": ["1.0.0"] * 4,
            "lead_hour": [1, 2, 3, 4],
            "label_temp_c": [1.0, 2.0, 3.0, 4.0],
            "nwp_temp_c_h1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    model = TemperatureRegressor()
    model.fit(rows)
    path = save_champion(model, tmp_path, "lethbridge-temp-20260603T1405Z")
    assert path == tmp_path / "lethbridge-temp-20260603T1405Z.joblib"
    TemperatureRegressor.load(path)  # must round-trip
```

(Note: `TemperatureRegressor.fit` calls `feature_columns(rows)`; passing one `nwp_*` feature column plus the required `feature_schema_version`/`label_temp_c` is enough for a tiny fit. If `feature_columns` selects differently, adjust the fixture columns to whatever `microclimate.models._columns.feature_columns` treats as features — read that module if the fit fails.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/publication/test_champion_publisher.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `src/microclimate/publication/champion_publisher.py`

```python
"""Champion model publishing helpers (L5): deterministic naming + local staging.

The deterministic version/tag/asset names let registry.json reference the Release asset URL
*before* the upload happens; the workflow then uploads to exactly that tag/filename. The actual
`gh release upload` + gh-pages push live in the training workflow (this module is offline).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from microclimate.contracts.registry import Task

_TAG_PREFIX = "champion-"


def champion_version(deployment_id: str, task: Task, run_time: datetime) -> str:
    """Deterministic version string from the run time (UTC), e.g. lethbridge-temp-20260603T1405Z."""
    return f"{deployment_id}-{task}-{run_time:%Y%m%dT%H%M}Z"


def release_tag(version: str) -> str:
    return f"{_TAG_PREFIX}{version}"


def asset_filename(version: str) -> str:
    return f"{version}.joblib"


def release_asset_url(repo: str, version: str) -> str:
    """Public download URL for the asset the workflow will upload to release_tag(version)."""
    return (
        f"https://github.com/{repo}/releases/download/"
        f"{release_tag(version)}/{asset_filename(version)}"
    )


def save_champion(model: object, out_dir: Path, version: str) -> Path:
    """Persist the fitted model to out_dir/<version>.joblib (the workflow uploads from here)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / asset_filename(version)
    model.save(path)  # type: ignore[attr-defined]
    return path
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/publication/test_champion_publisher.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Full gate + commit**

```bash
git add src/microclimate/publication/champion_publisher.py tests/publication/test_champion_publisher.py
git commit -m "feat(champion-publisher): deterministic version/tag/url + local model staging"
```

---

## Task 5: `run_training` orchestration + `load_champion`

**Files:**
- Modify: `src/microclimate/pipelines/training.py`
- Test: `tests/pipelines/test_training.py`

This ties everything together. Injected deps make it hermetic (no network/publish in tests).

- [ ] **Step 1: Write the failing test**

Create `tests/pipelines/test_training.py`:
```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.connectors.base import HistoricalCoverage, NWPSource, ObservationSource
from microclimate.contracts.forecast_frame import FORECAST_FRAME
from microclimate.contracts.observation import OBSERVATION_FRAME
from microclimate.contracts.physical_vars import PHYSICAL_VARS
from microclimate.contracts.registry import manifest_key
from microclimate.publication.registry_store import read_registry
from microclimate.training_store.store import TrainingStore

_PINNED = {
    "temp_c": 10.0, "dewpoint_c": 5.0, "surface_pressure_hpa": 900.0, "precip_mm": 0.0,
    "cloud_cover_fraction": 0.5, "solar_radiation_wm2": 100.0, "wind_speed_ms": 3.0, "wind_dir_deg": 180.0,
}


class _FakeNWP(NWPSource):
    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(self, issue_time, lat, lon, lead_hours):  # type: ignore[override]
        rows = []
        for h in lead_hours:
            r = {"issue_time": pd.Timestamp(issue_time), "lead_hour": int(h),
                 "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=int(h))}
            r.update(_PINNED)
            rows.append(r)
        return FORECAST_FRAME.validate(pd.DataFrame(rows))


class _FakeObs(ObservationSource):
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id, start, end):  # type: ignore[override]
        # hourly rows across [start,end]; precip alternates by hour so PoP has both classes.
        ts = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end).ceil("h"), freq="h", tz="UTC")
        data: dict[str, object] = {"station_id": [station_id] * len(ts), "timestamp": list(ts)}
        for v in PHYSICAL_VARS:
            data[v] = [_PINNED[v]] * len(ts)
            data[f"{v}_present"] = [True] * len(ts)
        data["precip_mm"] = [0.5 if t.hour % 2 == 0 else 0.0 for t in ts]
        return OBSERVATION_FRAME.validate(pd.DataFrame(data))

    def fetch_live(self, station_id, since):  # type: ignore[override]
        raise NotImplementedError


def test_run_training_promotes_off_baseline_and_writes_registry(tmp_path: Path) -> None:
    from microclimate.pipelines.training import run_training

    config = load_deployment("lethbridge")
    store = TrainingStore(tmp_path / "store")
    obs = {config.target.connector_key: _FakeObs()}
    out_dir = tmp_path / "out"
    registry_path = tmp_path / "registry.json"

    # ~5 months of 6-hourly issue times so temporal_split (holdout 3 / calib 1) is non-empty.
    start = datetime(2024, 1, 1, tzinfo=UTC)
    now = start + timedelta(days=150)

    summary = run_training(
        "lethbridge",
        nwp=_FakeNWP(),
        observations=obs,
        store=store,
        output_dir=out_dir,
        registry_path=registry_path,
        now=now,
        start=start,
        holdout_months=3,
        calib_months=1,
    )

    assert summary["rows"] > 0
    # the trained models trivially beat the constant baseline on these synthetic labels
    assert "temp" in summary["promoted"]
    manifest = read_registry(registry_path)
    assert manifest_key("lethbridge", "temp") in manifest.entries
    entry = manifest.entries[manifest_key("lethbridge", "temp")]
    assert entry.release_asset_url.endswith(".joblib")
    assert (out_dir / f"{entry.version}.joblib").exists()
```

(If the synthetic labels make a task un-promotable — e.g. PoP where the constant baseline already nails alternating precip — assert only on `temp`; the point is to exercise the promote→registry→save path. Adjust the asserted task to whichever the synthetic data lets win, but keep at least one promotion.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipelines/test_training.py -v`
Expected: FAIL — `run_training` raises `NotImplementedError` / signature mismatch.

- [ ] **Step 3: Implement** `src/microclimate/pipelines/training.py`

```python
"""Monthly training pipeline (L6): backfill -> train -> gate -> promote -> publish (ADR-0016)."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.http import http_get_bytes
from microclimate.connectors.registry import get_source, validate_config_sources
from microclimate.contracts.registry import RegistryEntry, Task, manifest_key
from microclimate.evaluation.metrics import nwp_pop_baseline
from microclimate.evaluation.publish_gate import GateResult, evaluate_challenger
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.pipelines.backfill import backfill_store, hrdps_issue_times
from microclimate.pipelines.training_data import assemble_from_store, temporal_split
from microclimate.publication import champion_publisher as cp
from microclimate.publication.registry_store import promote, read_registry, write_registry
from microclimate.training_store.store import TrainingStore

_REPO = os.environ.get("GITHUB_REPOSITORY", "gnieuwenhuis/microclimate_forecasting")


def load_champion(
    config: DeploymentConfig,
    registry_path: Path,
    task: Task,
    work_dir: Path,
    *,
    fetch_bytes: Callable[[str], bytes] = lambda url: http_get_bytes(url),
) -> object | None:
    """Load the current champion model for a task, or None when it's the baseline."""
    manifest = read_registry(registry_path)
    entry = manifest.entries.get(manifest_key(config.deployment_id, task))
    if entry is None or entry.version == "baseline":
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / cp.asset_filename(entry.version)
    local.write_bytes(fetch_bytes(entry.release_asset_url))
    return (
        TemperatureRegressor.load(local) if task == "temp" else PrecipOccurrenceClassifier.load(local)
    )


def run_training(
    deployment_id: str,
    *,
    nwp: NWPSource | None = None,
    observations: Mapping[str, ObservationSource] | None = None,
    store: TrainingStore | None = None,
    output_dir: Path | None = None,
    registry_path: Path | None = None,
    now: datetime | None = None,
    start: datetime | None = None,
    holdout_months: int | None = None,
    calib_months: int = 3,
    do_backfill: bool = True,
) -> dict[str, object]:
    """Backfill -> train -> gate -> promote -> write registry + champion binaries.

    Returns a summary dict. Promotion of zero tasks is a normal, successful outcome.
    """
    config = load_deployment(deployment_id)
    validate_config_sources(config)
    now = now or datetime.now(UTC)
    start = start or datetime.fromisoformat(config.training.seed.start).replace(tzinfo=UTC)
    holdout_months = holdout_months or config.training.holdout_months
    output_dir = output_dir or Path(os.environ.get("CHAMPION_OUTPUT_DIR", "champions"))
    registry_path = registry_path or Path(os.environ.get("REGISTRY_PATH", "registry.json"))
    store = store or TrainingStore(Path(os.environ.get("TRAINING_STORE_ROOT", "training-store")))
    nwp = nwp or cast(NWPSource, get_source(config.nwp.historical_connector))
    if observations is None:
        keys = {config.target.connector_key, *(n.connector_key for n in config.neighbors)}
        observations = {k: cast(ObservationSource, get_source(k)) for k in keys}

    if do_backfill:
        n = backfill_store(
            config, nwp=nwp, observations=observations, store=store,
            issue_times=hrdps_issue_times(start, now),
        )
        print(f"backfill: +{n} new runs")

    rows = assemble_from_store(config, store)
    train, calib, test = temporal_split(rows, holdout_months=holdout_months, calib_months=calib_months)
    print(f"rows={len(rows):,} train={len(train):,} calib={len(calib):,} test={len(test):,}")

    manifest = read_registry(registry_path)
    promoted: list[str] = []
    results: dict[str, GateResult] = {}

    # --- temp ---
    temp = TemperatureRegressor()
    temp.fit(pd.concat([train, calib], ignore_index=True))
    temp_baseline = test["nwp_temp_c"]
    temp_champion = load_champion(config, registry_path, "temp", output_dir / "_champion")
    res_t = evaluate_challenger("temp", temp, temp_champion, temp_baseline, test)
    results["temp"] = res_t
    print(f"temp gate: {res_t.reason}")
    if res_t.promote:
        manifest = _do_promote(manifest, "temp", config, temp, res_t, output_dir, now)
        promoted.append("temp")

    # --- pop ---
    pop = PrecipOccurrenceClassifier()
    pop.fit(train)
    pop.calibrate(calib)
    pop_baseline = nwp_pop_baseline(test, config.label.precip_occurrence_threshold_mm)
    pop_champion = load_champion(config, registry_path, "pop", output_dir / "_champion")
    res_p = evaluate_challenger("pop", pop, pop_champion, pop_baseline, test)
    results["pop"] = res_p
    print(f"pop gate: {res_p.reason}")
    if res_p.promote:
        manifest = _do_promote(manifest, "pop", config, pop, res_p, output_dir, now)
        promoted.append("pop")

    if promoted:
        write_registry(manifest, registry_path)
    print(f"promoted: {promoted or 'none'}")
    return {"rows": len(rows), "promoted": promoted, "results": results, "registry_path": registry_path}


def _do_promote(manifest, task: Task, config, model, result: GateResult, output_dir: Path, now: datetime):  # noqa: ANN001
    version = cp.champion_version(config.deployment_id, task, now)
    cp.save_champion(model, output_dir, version)
    entry = RegistryEntry(
        version=version,
        release_asset_url=cp.release_asset_url(_REPO, version),
        promoted_at=now,
        holdout_metrics=result.metrics,
    )
    return promote(manifest, task, config.deployment_id, entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training for a deployment.")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--no-backfill", action="store_true")
    args = parser.parse_args()
    run_training(args.deployment, do_backfill=not args.no_backfill)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipelines/test_training.py -v`
Expected: PASS. If `pop` doesn't promote on the synthetic data, the assert targets `temp` only (per the test note) — that's fine.

- [ ] **Step 5: Full gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
(`lint-imports` must still pass: training.py is L6 and may import evaluation/publication/models/connectors/pipelines.)
```bash
git add src/microclimate/pipelines/training.py tests/pipelines/test_training.py
git commit -m "feat(training): run_training orchestration — backfill, train, gate, promote, registry"
```

---

## Task 6: CI wiring — `training.yml`

**Files:**
- Modify: `.github/workflows/training.yml`

- [ ] **Step 1: Read the current workflow**

Run: `cat .github/workflows/training.yml`
Note the `discover` job (lists deployments) and the `run` matrix job.

- [ ] **Step 2: Rewrite the `run` job** to sync the store, run training, and publish on promotion. Replace the `run` job's steps with:

```yaml
  run:
    needs: discover
    runs-on: ubuntu-latest
    permissions:
      contents: write          # push training-data branch, gh-pages, and create Releases
    concurrency:
      group: training-${{ matrix.deployment }}   # never two pushes to the store at once
    strategy:
      matrix:
        deployment: ${{ fromJson(needs.discover.outputs.deployments) }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync

      - name: Check out (or bootstrap) the public training-data store
        env:
          STORE_URL: https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git
        run: |
          if git clone --depth 1 --branch training-data "$STORE_URL" store 2>/dev/null; then
            echo "Cloned existing training-data branch."
          else
            echo "Bootstrapping empty training-data store."
            mkdir store
          fi

      - name: Train + gate + stage champions
        env:
          TRAINING_STORE_ROOT: store
          CHAMPION_OUTPUT_DIR: champions
          REGISTRY_PATH: registry.json
          GITHUB_REPOSITORY: ${{ github.repository }}
        run: uv run python -m microclimate.pipelines.training --deployment ${{ matrix.deployment }}

      - name: Force-push the store state back to training-data
        env:
          STORE_URL: https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git
        run: |
          cd store
          git init -q 2>/dev/null || true
          git checkout -q -B training-data
          git add -A
          git -c user.name=ci -c user.email=ci@local commit -q -m "store: $(date -u +%FT%TZ)" || { echo "no changes"; exit 0; }
          git push -f "$STORE_URL" training-data

      - name: Publish promoted champions (Release assets) + registry.json (gh-pages)
        if: hashFiles('champions/*.joblib') != ''
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          for f in champions/*.joblib; do
            base=$(basename "$f" .joblib)            # == version
            gh release create "champion-$base" "$f" --title "champion-$base" --notes "Promoted champion $base" 2>/dev/null \
              || gh release upload "champion-$base" "$f" --clobber
          done
          # publish registry.json to gh-pages
          git fetch origin gh-pages --depth 1 || true
          tmp=$(mktemp -d); cp registry.json "$tmp/registry.json"
          git worktree add -f gp gh-pages 2>/dev/null || git worktree add -f -b gh-pages gp
          cp "$tmp/registry.json" gp/registry.json
          cd gp && git add registry.json \
            && git -c user.name=ci -c user.email=ci@local commit -q -m "registry: $(date -u +%FT%TZ)" \
            && git push origin gh-pages || echo "registry unchanged"
```

Also at the top, change the triggers: keep `schedule` (monthly cron) and `workflow_dispatch`; **remove the `push: paths: config/deployments/**` trigger** (per spec). Leave the `discover` job unchanged.

- [ ] **Step 3: Validate the YAML**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/training.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/training.yml
git commit -m "ci(training): store-sync + train/gate + publish champions (Release) and registry.json (gh-pages)"
```

---

## Task 7: ADR + README updates

**Files:**
- Modify: `docs/adr/0016-baseline-champion-pre-model-publishing.md`
- Modify: `README.md`

- [ ] **Step 1: Update ADR-0016 consequences**

In the Consequences/last bullet of `docs/adr/0016-baseline-champion-pre-model-publishing.md` (which says "the registry/champion-loading and the … gh-pages git sync are separate follow-on specs"), append a note:
> **Update (2026-06-03):** the training-side of this is now implemented — `pipelines.training.run_training` runs the champion/challenger publish gate and publishes promoted champions (GitHub Release assets) + `registry.json` (gh-pages), with the training store persisted on the `training-data` branch. The **inference side still reads the baseline**; swapping inference to load the registry/champion remains a separate slice.

- [ ] **Step 2: Update README "Project status"**

Edit the status section: the training pipeline (`pipelines/training.py`) is now implemented — monthly retrain does backfill → train → publish gate → promote → publish champion (Release) + `registry.json` (gh-pages). Note `publish_gate` and `registry_store` are no longer stubs. The remaining stub of note: inference still publishes the baseline (doesn't yet read the registry); `acis` retained-but-unused. Run `grep -rl NotImplementedError src/microclimate` and reflect the result.

- [ ] **Step 3: Verify no stale "stub" claims + final gate**

Run: `grep -rni "publish gate.*stub\|registry.*stub\|training pipeline.*stub" README.md docs/adr/0016*.md || echo "(clean)"`
Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add docs/adr/0016-baseline-champion-pre-model-publishing.md README.md
git commit -m "docs: training pipeline + publish gate implemented (ADR-0016 follow-on); README status"
```

---

## Self-Review

**Spec coverage:** temporal split (Task 1 ↔ spec §1), publish gate (Task 2 ↔ §2), registry store (Task 3 ↔ §3), champion publisher (Task 4 ↔ §4), champion load + orchestration (Task 5 ↔ §5/§6), CI store-sync + publish + trigger change (Task 6 ↔ §7), ADR/README (Task 7 ↔ spec "ADR note"). All covered.

**Placeholder scan:** every code step has full code; no TBD/TODO. Two tests carry an explicit "adjust the asserted task/columns if the synthetic data differs" note — these are real fallbacks, not placeholders (the assertion still verifies the promote→registry→save path).

**Type consistency:** `evaluate_challenger(task, challenger, champion, baseline: pd.Series, holdout)` returns `GateResult{promote,reason,metrics}` — used consistently in Task 5. `champion_version`/`release_tag`/`asset_filename`/`release_asset_url`/`save_champion` signatures match between Task 4 and Task 5. `temporal_split(rows,*,holdout_months,calib_months)` consistent (Task 1 ↔ Task 5). `read_registry/promote/write_registry` consistent (Task 3 ↔ Task 5). `RegistryEntry` fields match the contract.

**Known dependencies to verify during execution (not placeholders):** `microclimate.models._columns.feature_columns` column selection (Task 4 fixture may need an extra feature column for the tiny fit); the exact set of feature-matrix columns the models require for a degenerate fit. Both flagged inline with "read that module if the fit fails."

**Out of scope (deferred, per spec):** inference reading the registry/champion; the §1b forward-capture.
