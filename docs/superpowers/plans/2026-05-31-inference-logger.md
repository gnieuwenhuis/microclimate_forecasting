# Inference Logger (first slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An in-process `run_inference` that builds a snapshot, produces a raw-HRDPS **baseline** forecast, writes the `ForecastDocument` JSON, and appends the snapshot to the training store — the logger-first MVP, all to local paths.

**Architecture:** `models.baseline.baseline_predictions` (temp passthrough + 0/1 PoP) feeds `pipelines.inference.run_inference(config, *, nwp, observations, store, forecast_path, issue_time)`, which reuses `build_snapshot`/`build_features`, assembles a `ForecastDocument`, calls `publication.write_forecast` (atomic JSON), and `store.append_snapshot`. `main()` wires live sources for prod. Registry/champion-loading and the private-repo/gh-pages git sync are out of scope (follow-on specs).

**Tech Stack:** Python 3.12, pandas, Pydantic (`ForecastDocument`), pytest, pyright strict, ruff, uv. Reuses the live Datamart connector, `build_snapshot`/`build_features`, and `TrainingStore` (all on `main`).

---

## Conventions for every task

- TDD: failing test → confirm fail → implement → confirm pass → commit.
- Full gate before each commit: `uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`. Network tests deselected by default.
- pandas/pyarrow are untyped under pyright strict; suppress with the narrow `# type: ignore[reportUnknownMemberType]` / `reportUnknownArgumentType` exactly where pyright reports (the snippets mark common ones).
- Commit on branch `spec/inference-logger` (main is PR-only); push only at Final Integration.

## File structure

**Create**
- `src/microclimate/models/baseline.py` — `baseline_predictions`, `BASELINE_VERSION`.
- `tests/models/test_baseline.py`, `tests/publication/test_forecast_writer.py`, `tests/pipelines/test_inference.py`.
- `docs/adr/0016-baseline-champion-pre-model-publishing.md`.

**Modify**
- `src/microclimate/contracts/forecast.py` — add `FORECAST_SCHEMA_VERSION`.
- `src/microclimate/publication/forecast_writer.py` — fill `write_forecast`.
- `src/microclimate/pipelines/inference.py` — fill `run_inference` + `_assemble_forecast` + `main()` wiring.
- `tests/pipelines/test_pipelines_cli.py` — remove the now-stale `test_run_inference_stubbed`.
- `README.md`, `CONTEXT.md`.

---

### Task 1: `baseline_predictions`

**Files:**
- Create: `src/microclimate/models/baseline.py`, `tests/models/test_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_baseline.py
from __future__ import annotations

import pandas as pd

from microclimate.models.baseline import BASELINE_VERSION, baseline_predictions


def test_baseline_temp_passthrough_and_pop_threshold() -> None:
    rows = pd.DataFrame(
        {
            "lead_hour": [1, 2, 3],
            "nwp_temp_c": [10.0, 11.0, 12.0],
            "nwp_precip_mm": [0.0, 0.2, 0.5],  # threshold 0.2: [no, yes(inclusive), yes]
        }
    )
    out = baseline_predictions(rows, threshold_mm=0.2)
    assert list(out["pred_temp_c"]) == [10.0, 11.0, 12.0]
    assert list(out["pred_pop"]) == [0.0, 1.0, 1.0]
    # original columns preserved (reshape needs lead_hour/valid_time downstream)
    assert "lead_hour" in out.columns


def test_baseline_version_constant() -> None:
    assert BASELINE_VERSION == "baseline"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_baseline.py -v`
Expected: FAIL — `microclimate.models.baseline` absent.

- [ ] **Step 3: Implement**

```python
# src/microclimate/models/baseline.py
"""Raw-HRDPS baseline forecaster (L4).

The initial published champion (ADR-0016) and the floor a trained model must beat: temperature
is the HRDPS 2 m passthrough; PoP is the raw-HRDPS occurrence call (precip ≥ threshold → 1).
Pure; no I/O. Self-contained — does NOT import `evaluation` (models/evaluation are import-linter
siblings); the one-line occurrence rule is duplicated, matching `evaluation.nwp_pop_baseline`.
"""

from __future__ import annotations

import pandas as pd

BASELINE_VERSION = "baseline"


def baseline_predictions(rows: pd.DataFrame, threshold_mm: float) -> pd.DataFrame:
    """Add `pred_temp_c` (= nwp_temp_c) and `pred_pop` (= 1.0 if nwp_precip_mm ≥ threshold)."""
    out = rows.copy()
    out["pred_temp_c"] = rows["nwp_temp_c"]
    out["pred_pop"] = (rows["nwp_precip_mm"] >= threshold_mm).astype(float)
    return out
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/models/test_baseline.py -v` → PASS (2 tests).

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/models/baseline.py tests/models/test_baseline.py
git commit -m "feat(models): raw-HRDPS baseline forecaster (temp passthrough + threshold PoP)"
```

---

### Task 2: `FORECAST_SCHEMA_VERSION` + `write_forecast`

**Files:**
- Modify: `src/microclimate/contracts/forecast.py`, `src/microclimate/publication/forecast_writer.py`
- Create: `tests/publication/test_forecast_writer.py`

- [ ] **Step 1: Add the schema-version constant** to `src/microclimate/contracts/forecast.py` — insert after the imports, before `class ForecastStep`:

```python
# Version of the published forecast-document contract; the cross-client compatibility key
# (ADR-0003). Bump when the document's shape/meaning changes.
FORECAST_SCHEMA_VERSION: str = "1.0.0"
```

- [ ] **Step 2: Write the failing writer test**

```python
# tests/publication/test_forecast_writer.py
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from microclimate.contracts.forecast import (
    FORECAST_SCHEMA_VERSION,
    ForecastDocument,
    ForecastStep,
)
from microclimate.publication.forecast_writer import write_forecast


def _doc() -> ForecastDocument:
    t0 = datetime(2026, 6, 1, 0, tzinfo=UTC)
    return ForecastDocument(
        schema_version=FORECAST_SCHEMA_VERSION,
        deployment_id="lethbridge",
        issue_time=t0,
        last_updated=t0,
        status="ok",
        model_versions={"temp": "baseline", "pop": "baseline"},
        attribution=["Data Source: Environment and Climate Change Canada (ECCC)"],
        series=[ForecastStep(lead_hour=1, valid_time=t0, temp_c=10.0, pop=0.0)],
    )


def test_write_forecast_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "lethbridge.json"  # parent dir does not exist yet
    doc = _doc()
    write_forecast(doc, path)
    assert path.exists()
    assert ForecastDocument.model_validate_json(path.read_text()) == doc


def test_write_forecast_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "lethbridge.json"
    write_forecast(_doc(), path)
    assert list(tmp_path.glob(".*.tmp")) == []
    assert path.exists()
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/publication/test_forecast_writer.py -v` → FAIL (`write_forecast` raises `NotImplementedError` / `FORECAST_SCHEMA_VERSION` absent).

- [ ] **Step 4: Implement** `src/microclimate/publication/forecast_writer.py`:

```python
"""Write a ForecastDocument to JSON — only through the validated model (L5)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from microclimate.contracts.forecast import ForecastDocument


def write_forecast(doc: ForecastDocument, path: Path) -> None:
    """Atomically write the forecast document as JSON (temp file + os.replace).

    The ForecastDocument is schema-valid by construction (Pydantic), so dumping the model is
    the validation boundary. Atomic so a crashed run never leaves a half-written JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{uuid.uuid4().hex}.tmp"
    tmp.write_text(doc.model_dump_json(indent=2))
    os.replace(tmp, path)
```

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/publication/test_forecast_writer.py -v` → PASS (2 tests).

- [ ] **Step 6: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/contracts/forecast.py src/microclimate/publication/forecast_writer.py tests/publication/test_forecast_writer.py
git commit -m "feat(publication): write_forecast — atomic ForecastDocument JSON + FORECAST_SCHEMA_VERSION"
```

---

### Task 3: `run_inference` + `_assemble_forecast` + `main()`

**Files:**
- Modify: `src/microclimate/pipelines/inference.py`, `tests/pipelines/test_pipelines_cli.py`
- Create: `tests/pipelines/test_inference.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/pipelines/test_inference.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from microclimate.contracts.forecast import ForecastDocument
from microclimate.pipelines.inference import run_inference
from microclimate.training_store import TrainingStore
from tests.fakes import FakeNWP, FakeObs, make_config, make_forecast_frame, make_obs_frame

_T0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def test_run_inference_publishes_baseline_and_logs_snapshot(tmp_path: Path) -> None:
    config = make_config(horizon_hours=3, lag_hours=2)  # target T1 + neighbor N1, key "fake"
    leads = [1, 2, 3]
    nwp = FakeNWP(make_forecast_frame(_T0, leads))
    ts = [_T0 - timedelta(hours=h) for h in (2, 1, 0)]
    observations = {"fake": FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})}
    store = TrainingStore(tmp_path / "store")
    forecast_path = tmp_path / "forecasts" / "test.json"

    doc = run_inference(
        config,
        nwp=nwp,
        observations=observations,
        store=store,
        forecast_path=forecast_path,
        issue_time=_T0,
    )

    # forecast written + valid + equal to the returned doc
    assert ForecastDocument.model_validate_json(forecast_path.read_text()) == doc
    assert len(doc.series) == 3
    assert [s.lead_hour for s in doc.series] == [1, 2, 3]
    assert doc.status == "ok"
    assert doc.model_versions == {"temp": "baseline", "pop": "baseline"}
    assert doc.attribution  # non-empty (ADR-0009)
    # PINNED temp_c=15.0 → passthrough; PINNED precip 0.5 ≥ threshold 0.2 → pop 1.0
    assert all(s.temp_c == 15.0 for s in doc.series)
    assert all(s.pop == 1.0 for s in doc.series)
    # snapshot logged to the store
    logged = store.read_snapshots(config.deployment_id)
    assert len(logged) == 1
    assert logged[0].issue_time == _T0
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/pipelines/test_inference.py -v` → FAIL (`run_inference` is the old stub with the wrong signature / raises `NotImplementedError`).

- [ ] **Step 3: Rewrite `src/microclimate/pipelines/inference.py`:**

```python
"""Hourly inference + logger pipeline (L6, ADR-0003/0007/0009/0016).

Builds the snapshot, produces the raw-HRDPS baseline forecast, writes the ForecastDocument
JSON, and appends the snapshot to the training store. Registry/champion-loading and the
private-repo/gh-pages git sync are out of scope (separate specs); this writes to local paths.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source
from microclimate.contracts.forecast import FORECAST_SCHEMA_VERSION, ForecastDocument, ForecastStep
from microclimate.features.feature_builder import build_features
from microclimate.features.snapshot_builder import build_snapshot
from microclimate.models.baseline import BASELINE_VERSION, baseline_predictions
from microclimate.publication.forecast_writer import write_forecast
from microclimate.training_store import TrainingStore

_ATTRIBUTION = ["Data Source: Environment and Climate Change Canada (ECCC)"]


def _assemble_forecast(
    config: DeploymentConfig,
    preds: pd.DataFrame,
    issue_time: datetime,
    last_updated: datetime,
) -> ForecastDocument:
    """Reshape per-(lead) baseline predictions into a ForecastDocument (ADR-0012: pipeline owns this)."""
    sdf = preds.sort_values("lead_hour")
    series = [
        ForecastStep(
            lead_hour=int(lh),  # type: ignore[reportUnknownArgumentType]
            valid_time=pd.Timestamp(vt).to_pydatetime(),  # type: ignore[reportUnknownArgumentType]
            temp_c=float(tc),  # type: ignore[reportUnknownArgumentType]
            pop=min(1.0, max(0.0, float(pp))),  # type: ignore[reportUnknownArgumentType]
        )
        for lh, vt, tc, pp in zip(
            sdf["lead_hour"], sdf["valid_time"], sdf["pred_temp_c"], sdf["pred_pop"], strict=True
        )
    ]
    return ForecastDocument(
        schema_version=FORECAST_SCHEMA_VERSION,
        deployment_id=config.deployment_id,
        issue_time=issue_time,
        last_updated=last_updated,
        status="ok",
        model_versions={"temp": BASELINE_VERSION, "pop": BASELINE_VERSION},
        attribution=_ATTRIBUTION,
        series=series,
    )


def run_inference(
    config: DeploymentConfig,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    store: TrainingStore,
    forecast_path: Path,
    issue_time: datetime,
) -> ForecastDocument:
    """Build a snapshot → baseline forecast → write JSON → log the snapshot. Returns the doc."""
    snapshot = build_snapshot(config, issue_time, nwp, observations)
    matrix = build_features(snapshot, config)
    preds = baseline_predictions(matrix, config.label.precip_occurrence_threshold_mm)
    doc = _assemble_forecast(config, preds, issue_time, last_updated=issue_time)
    write_forecast(doc, forecast_path)
    store.append_snapshot(snapshot)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly inference for a deployment.")
    parser.add_argument("--deployment", required=True)
    args = parser.parse_args()

    config = load_deployment(args.deployment)
    nwp = cast(NWPSource, get_source(config.nwp.live_connector))
    station_keys = {config.target.connector_key, *(n.connector_key for n in config.neighbors)}
    observations = {k: cast(ObservationSource, get_source(k)) for k in station_keys}
    store = TrainingStore(Path(os.environ.get("TRAINING_STORE_ROOT", "training-store")))
    issue_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

    run_inference(
        config,
        nwp=nwp,
        observations=observations,
        store=store,
        forecast_path=Path(config.output.forecast_json),
        issue_time=issue_time,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Remove the stale stub test.** In `tests/pipelines/test_pipelines_cli.py`, delete `test_run_inference_stubbed` (run_inference is implemented now and has a new signature). Keep `test_run_training_stubbed` and `test_inference_cli_requires_deployment`.

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/pipelines/test_inference.py tests/pipelines/test_pipelines_cli.py -v` → PASS. Then `uv run pyright`; add any narrow `# type: ignore[...]` pyright reports on the pandas-iteration lines beyond those already marked.

- [ ] **Step 6: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/pipelines/inference.py tests/pipelines/test_inference.py tests/pipelines/test_pipelines_cli.py
git commit -m "feat(pipelines): run_inference — baseline forecast publish + snapshot logging (in-process)"
```

---

### Task 4: ADR-0016 + README + CONTEXT

**Files:**
- Create: `docs/adr/0016-baseline-champion-pre-model-publishing.md`
- Modify: `README.md`, `CONTEXT.md`

- [ ] **Step 1: Write ADR-0016** (0016 is the next free number on `main` — highest is 0015):

```markdown
# 16. Baseline raw-HRDPS forecaster is the initial published champion

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0003 (server-side inference, published JSON), ADR-0004 (two models),
  ADR-0007/0008 (logger-first pivot), ADR-0014 (raw-HRDPS baseline / nwp_pop_baseline).

## Context

CaSPAr is unavailable, so there is no historical seed and no trained model at launch
(ADR-0008 logger-first pivot). The service must still go live (the project's live-hourly
constraint) and start logging snapshots forward.

## Decision

The inference pipeline publishes a **raw-HRDPS baseline forecast** as the initial champion
until a trained model is promoted: temperature is the HRDPS 2 m passthrough; PoP is the raw
occurrence call (`nwp_precip_mm ≥ config.label.precip_occurrence_threshold_mm` → 1.0/0.0,
identical to `evaluation.nwp_pop_baseline` — the floor the trained model must beat). The
published `ForecastDocument.model_versions` is `{"temp": "baseline", "pop": "baseline"}` so
clients and the (future) registry can see the forecast is un-downscaled. Each run also logs
its snapshot to the training store (ADR-0007/0015), accumulating labels forward.

## Consequences

- The service is live and verifiable from day one, before any training; the trained model
  later swaps in via champion/challenger (ADR-0006) once it beats this baseline.
- `status` is `"ok"` for a successful baseline run; the `degraded`/`stale` signals (obs-source
  failure once trained models depend on obs; run freshness) are deferred to later work.
- This first slice runs in-process to local paths; the registry/champion-loading and the
  private-repo + gh-pages git sync (the GitHub Action, ADR-0009) are separate follow-on specs.
```

- [ ] **Step 2: Update `README.md` Project status** — add the inference logger to the implemented list and adjust the not-yet-implemented list. Read the current section and edit coherently: note that `pipelines.inference.run_inference` publishes the baseline forecast JSON and logs snapshots to the training store (in-process, ADR-0016); the registry/champion-loading, the publish gate, and the private-repo/gh-pages Action remain.

- [ ] **Step 3: Add a CONTEXT.md "Baseline forecaster" term** under Modeling & quality (near "Baseline / raw HRDPS"):

```markdown
- **Baseline forecaster** — the raw-HRDPS forecaster published before a trained model exists
  (ADR-0016): temperature passthrough + threshold PoP. The initial champion and the floor a
  trained model must beat; `model_versions` marks it `"baseline"`.
```

- [ ] **Step 4: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add docs/adr/0016-baseline-champion-pre-model-publishing.md README.md CONTEXT.md
git commit -m "docs: ADR-0016 baseline champion / pre-model publishing; README + CONTEXT"
```

---

## Final Integration

- [ ] Push and open a PR (main is PR-only):

```bash
git push -u origin spec/inference-logger
gh pr create --fill --base main
```

- [ ] After automated review + CI, address feedback and merge.

---

## Self-review notes

- **Spec coverage:** `baseline_predictions` (Task 1) ✓; `FORECAST_SCHEMA_VERSION` + `write_forecast` (Task 2) ✓; `run_inference` + `_assemble_forecast` + `main()` (Task 3) ✓; ADR-0016 + README + CONTEXT (Task 4) ✓. `status="ok"` deferral and out-of-scope (registry/champion-loading, git sync) honored.
- **Spec refinement (deliberate):** `run_inference` takes an injected `DeploymentConfig` rather than a `deployment_id` + internal `load_deployment` — cleaner to test (fake config + fake sources) and consistent with `build_snapshot`/`build_features`. `main()` does `load_deployment`. Resolves the spec's `_assemble_forecast`-placement and config open items (private helper; config injected).
- **Stale test handled:** `test_run_inference_stubbed` is removed (Task 3 Step 4) since `run_inference` is no longer a stub and its signature changed; the real integration test replaces it. `test_inference_cli_requires_deployment` still passes (argparse `SystemExit` fires before source wiring).
- **Type consistency:** `baseline_predictions(rows, threshold_mm)` → adds `pred_temp_c`/`pred_pop`, consumed by `_assemble_forecast` via the same names; `BASELINE_VERSION = "baseline"` used in `model_versions`; `run_inference` signature matches its test + `main()` call site.
- **Open items resolved:** `_assemble_forecast` is a private helper in `inference.py`; `main()` uses `config.nwp.live_connector` (Datamart) + `issue_time = now(UTC)` floored to the hour + `TRAINING_STORE_ROOT` env (default `training-store/`, finalized by the Action spec); `baseline_predictions` returns a copy with added columns (keeps `lead_hour`/`valid_time` for the reshape).
