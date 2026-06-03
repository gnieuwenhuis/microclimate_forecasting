# Inference Serves the Champion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make hourly inference load and serve the promoted champion per task (temp/pop) from `registry.json`, fall back to the raw-HRDPS baseline (marking `status="degraded"` only when an *expected* champion is unusable), stamp per-task `model_versions`, and publish the forecast JSON to gh-pages.

**Architecture:** Extract `load_champion` to a shared `publication` module (L5, above `models` — legal downward import), used by both training and inference. `run_inference` computes the baseline once, then per task serves the champion if the registry names a usable one (else baseline), composes the prediction frame, and assembles a `ForecastDocument` with per-task versions + status. The inference workflow clones gh-pages, reads `registry.json` from it, and pushes the forecast JSON back.

**Tech Stack:** Python 3.12, pandas, pydantic (forecast/registry contracts), the existing LightGBM model wrappers, pytest, `uv`, GitHub Actions + `gh`.

**Authoritative docs:** `docs/superpowers/specs/2026-06-03-inference-serves-champion-design.md`; ADR-0016/0006/0009.

**Verified interfaces:**
- Layer order (`.importlinter`): `pipelines > training_store > publication > models > evaluation > features > connectors > config > contracts`. So `publication` MAY import `models`/`connectors`/`contracts` (downward); `pipelines` may import `publication`.
- `TemperatureRegressor`/`PrecipOccurrenceClassifier` (`microclimate.models.*`): `predict(rows) -> pd.Series` (named `pred_temp_c`/`pred_pop`; **raises `ValueError`** if `rows["feature_schema_version"].iloc[0]` ≠ the model's), `save(Path)`, `classmethod load(Path)`. PoP needs `fit` then `calibrate` before `predict`.
- `microclimate.models.baseline.baseline_predictions(rows, threshold_mm) -> pd.DataFrame` adds `pred_temp_c`(=`nwp_temp_c`) and `pred_pop`(=1.0 if `nwp_precip_mm`≥threshold). `BASELINE_VERSION = "baseline"`.
- `microclimate.publication.registry_store.read_registry(path) -> RegistryManifest` (**raises** on corrupt JSON; empty manifest if file absent).
- `microclimate.contracts.registry`: `Task=Literal["temp","pop"]`, `manifest_key(dep,task)`, `RegistryManifest.entries: dict[str,RegistryEntry]`, `RegistryEntry{version, release_asset_url, ...}`.
- `microclimate.publication.champion_publisher.asset_filename(version) -> "<version>.joblib"`.
- `microclimate.connectors.http.http_get_bytes(url,*,params=None) -> bytes`.
- `microclimate.publication.forecast_writer.write_forecast(doc, path)`.
- `ForecastDocument{schema_version, deployment_id, issue_time, last_updated, status: Literal["ok","stale","degraded"], model_versions: dict[Literal["temp","pop"],str], attribution, series}`; `ForecastStep{lead_hour, valid_time, temp_c, pop}`.
- Current `load_champion` lives in `src/microclimate/pipelines/training.py` taking `config: DeploymentConfig`; we move it to `publication` and change its first param to `deployment_id: str`.
- `config.output.forecast_json == "forecasts/lethbridge.json"`.

**Conventions:** UTC everywhere; injected deps for hermetic tests; run the FULL gate before each commit: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`.

---

## File Structure

- Create: `src/microclimate/publication/champion_loader.py` — `load_champion` + `_Predictor` Protocol (shared by training + inference).
- Modify: `src/microclimate/pipelines/training.py` — import `load_champion` from `publication`; drop the local copy; pass `config.deployment_id`.
- Modify: `src/microclimate/pipelines/inference.py` — `_serve_task`, `run_inference` (champion-or-baseline), `_assemble_forecast` signature, `main`.
- Modify: `.github/workflows/inference.yml` — gh-pages checkout/read/publish.
- Modify: `README.md` — status (inference now serves the champion).
- Tests: `tests/publication/test_champion_loader.py`, `tests/pipelines/test_inference.py` (extend).

---

## Task 1: Extract `load_champion` to `publication/champion_loader.py`

**Files:**
- Create: `src/microclimate/publication/champion_loader.py`
- Modify: `src/microclimate/pipelines/training.py`
- Test: `tests/publication/test_champion_loader.py`

- [ ] **Step 1: Write the failing test** — create `tests/publication/test_champion_loader.py`

```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, manifest_key
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.publication import champion_publisher as cp
from microclimate.publication.champion_loader import load_champion
from microclimate.publication.registry_store import write_registry


def _fit_tiny_temp() -> TemperatureRegressor:
    rows = pd.DataFrame(
        {
            "feature_schema_version": ["1.0.0"] * 4,
            "lead_hour": [1, 2, 3, 4],
            "label_temp_c": [1.0, 2.0, 3.0, 4.0],
            "nwp_temp_c": [1.0, 2.0, 3.0, 4.0],
        }
    )
    m = TemperatureRegressor()
    m.fit(rows)
    return m


def test_load_champion_none_when_no_entry(tmp_path: Path) -> None:
    write_registry(RegistryManifest(), tmp_path / "registry.json")
    assert load_champion("lethbridge", tmp_path / "registry.json", "temp", tmp_path / "wd") is None


def test_load_champion_none_when_baseline_entry(tmp_path: Path) -> None:
    entry = RegistryEntry(
        version="baseline", release_asset_url="x", promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={},
    )
    m = RegistryManifest(entries={manifest_key("lethbridge", "temp"): entry})
    write_registry(m, tmp_path / "registry.json")
    assert load_champion("lethbridge", tmp_path / "registry.json", "temp", tmp_path / "wd") is None


def test_load_champion_downloads_and_loads(tmp_path: Path) -> None:
    # stage a real champion .joblib, then serve its bytes via an injected fetch_bytes
    version = "lethbridge-temp-20260603T0000Z"
    asset = cp.save_champion(_fit_tiny_temp(), tmp_path / "stage", version)
    raw = asset.read_bytes()
    entry = RegistryEntry(
        version=version, release_asset_url=f"https://example/{version}.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC), holdout_metrics={"mae": 1.0},
    )
    m = RegistryManifest(entries={manifest_key("lethbridge", "temp"): entry})
    write_registry(m, tmp_path / "registry.json")

    loaded = load_champion(
        "lethbridge", tmp_path / "registry.json", "temp", tmp_path / "wd",
        fetch_bytes=lambda _url: raw,
    )
    assert loaded is not None
    assert isinstance(loaded, TemperatureRegressor)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/publication/test_champion_loader.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `src/microclimate/publication/champion_loader.py`**

```python
"""Load the current champion model named by the registry, or None for the baseline (L5).

Shared by the training pipeline (re-evaluate the champion on the holdout) and the inference
pipeline (serve it). `publication` sits above `models` in the layer order, so importing the
model classes here is a legal downward import.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

from microclimate.connectors.http import http_get_bytes
from microclimate.contracts.registry import Task, manifest_key
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.publication import champion_publisher as cp
from microclimate.publication.registry_store import read_registry


class _Predictor(Protocol):
    """Duck-type contract for a fitted model that scores feature-matrix rows."""

    def predict(self, rows: pd.DataFrame) -> pd.Series: ...


def load_champion(
    deployment_id: str,
    registry_path: Path,
    task: Task,
    work_dir: Path,
    *,
    fetch_bytes: Callable[[str], bytes] = lambda url: http_get_bytes(url),
) -> _Predictor | None:
    """Load the registry's current champion model for a task, or None when it's the baseline.

    None when there is no entry or the entry is the ``"baseline"`` sentinel. Otherwise downloads
    the entry's ``release_asset_url`` (via ``fetch_bytes``) and loads the task's model class.
    Raises on download/load failure (the caller decides fallback).
    """
    manifest = read_registry(registry_path)
    entry = manifest.entries.get(manifest_key(deployment_id, task))
    if entry is None or entry.version == "baseline":
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / cp.asset_filename(entry.version)
    local.write_bytes(fetch_bytes(entry.release_asset_url))
    if task == "temp":
        return TemperatureRegressor.load(local)
    return PrecipOccurrenceClassifier.load(local)
```

- [ ] **Step 4: Refactor `src/microclimate/pipelines/training.py` to use the shared loader**

Remove the local `load_champion` function and the now-unused imports it needed *only* for loading (`http_get_bytes`, the local `_Predictor` Protocol if it's only used by `load_champion` — check; `cp`/`read_registry`/`manifest_key` may still be used by `_do_promote`/orchestration, keep those). Add:
```python
from microclimate.publication.champion_loader import load_champion
```
Update the two call sites in `run_training` from `load_champion(config, registry_path, "temp", champ_dir)` to `load_champion(config.deployment_id, registry_path, "temp", champ_dir)` (and likewise `"pop"`). Read `training.py` to confirm exactly which imports become unused after the deletion and remove only those (ruff will flag unused imports).

- [ ] **Step 5: Run to verify the loader tests pass + training still works**

Run: `uv run pytest tests/publication/test_champion_loader.py tests/pipelines/test_training.py -v`
Expected: loader tests pass; the existing training orchestration test still passes (uses `load_champion` via `run_training`).

- [ ] **Step 6: FULL gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
(`lint-imports` MUST stay green — confirm `publication.champion_loader` importing `models` is legal, and `pipelines.training` importing `publication` is downward.)
```bash
git add src/microclimate/publication/champion_loader.py src/microclimate/pipelines/training.py tests/publication/test_champion_loader.py
git commit -m "refactor(publication): extract shared load_champion (L5); training uses it"
```

---

## Task 2: `run_inference` serves the champion (fallback to baseline)

**Files:**
- Modify: `src/microclimate/pipelines/inference.py`
- Test: `tests/pipelines/test_inference.py`

- [ ] **Step 1: Write the failing tests** — replace/extend `tests/pipelines/test_inference.py`

Read the current `tests/pipelines/test_inference.py` first to reuse its fake NWP/obs fixtures (it already drives `run_inference` hermetically from the ADR-0019 stateless-inference work). Add a champion fixture helper and the serving cases. Target tests:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.contracts.registry import RegistryEntry, RegistryManifest, manifest_key
from microclimate.pipelines.inference import run_inference
from microclimate.publication import champion_publisher as cp
from microclimate.publication.registry_store import write_registry

# Reuse the module's existing _FakeNWP / _FakeObs (or import from tests.fakes). They must yield a
# feature matrix at the current FEATURE_SCHEMA_VERSION so a champion fit on the same matrix is
# compatible. See the existing test for the fakes; the helpers below assume `make_fakes()` returns
# (config, nwp, observations) for "lethbridge".


def _registry_with(tmp_path: Path, entries: dict[str, RegistryEntry]) -> Path:
    p = tmp_path / "registry.json"
    write_registry(RegistryManifest(entries=entries), p)
    return p


def _real_temp_champion(tmp_path, config, nwp, observations, issue_time):
    """Fit a temp champion on the SAME feature matrix inference will build, save it, return
    (entry, raw_bytes) so an injected fetch_bytes can serve it."""
    from microclimate.features.feature_builder import build_features
    from microclimate.features.snapshot_builder import build_snapshot
    from microclimate.models.temp_model import TemperatureRegressor

    snap = build_snapshot(config, issue_time, nwp, observations)
    matrix = build_features(snap, config)
    matrix = matrix.copy()
    matrix["label_temp_c"] = matrix["nwp_temp_c"] + 1.0  # learnable offset
    model = TemperatureRegressor()
    model.fit(matrix)
    version = "lethbridge-temp-20260603T0000Z"
    raw = cp.save_champion(model, tmp_path / "stage", version).read_bytes()
    entry = RegistryEntry(
        version=version, release_asset_url=f"https://example/{version}.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC), holdout_metrics={"mae": 1.0},
    )
    return entry, raw


def test_no_registry_serves_baseline_ok(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    doc = run_inference(
        config, nwp=nwp, observations=obs, forecast_path=tmp_path / "f.json", issue_time=it,
        registry_path=tmp_path / "absent.json", work_dir=tmp_path / "wd",
    )
    assert doc.status == "ok"
    assert doc.model_versions == {"temp": "baseline", "pop": "baseline"}
    assert (tmp_path / "f.json").exists()


def test_real_temp_champion_served(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    entry, raw = _real_temp_champion(tmp_path, config, nwp, obs, it)
    reg = _registry_with(tmp_path, {manifest_key("lethbridge", "temp"): entry})
    doc = run_inference(
        config, nwp=nwp, observations=obs, forecast_path=tmp_path / "f.json", issue_time=it,
        registry_path=reg, work_dir=tmp_path / "wd", fetch_bytes=lambda _u: raw,
    )
    assert doc.status == "ok"
    assert doc.model_versions["temp"] == entry.version
    assert doc.model_versions["pop"] == "baseline"  # no pop entry


def test_expected_champion_download_fails_is_degraded(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    entry = RegistryEntry(
        version="lethbridge-temp-x", release_asset_url="https://example/x.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC), holdout_metrics={},
    )
    reg = _registry_with(tmp_path, {manifest_key("lethbridge", "temp"): entry})

    def _boom(_u: str) -> bytes:
        raise RuntimeError("download failed")

    doc = run_inference(
        config, nwp=nwp, observations=obs, forecast_path=tmp_path / "f.json", issue_time=it,
        registry_path=reg, work_dir=tmp_path / "wd", fetch_bytes=_boom,
    )
    assert doc.status == "degraded"
    assert doc.model_versions["temp"] == "baseline"


def test_corrupt_registry_treated_as_empty(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    bad = tmp_path / "registry.json"
    bad.write_text("{ not valid json")
    doc = run_inference(
        config, nwp=nwp, observations=obs, forecast_path=tmp_path / "f.json", issue_time=it,
        registry_path=bad, work_dir=tmp_path / "wd",
    )
    assert doc.status == "ok" and doc.model_versions["temp"] == "baseline"
```

Provide `_make_fakes()` by reusing the existing fakes in this test module (the ADR-0019 inference test already defines `_FakeNWP`/`_FakeObs` and loads the `lethbridge` config); factor a small `_make_fakes()` returning `(config, _FakeNWP(), {target_key: _FakeObs()})`. If the existing test used different fixture names, adapt — keep the existing passing tests and ADD these. (A stale-schema case is covered indirectly; optionally add one by fitting a champion on rows with `feature_schema_version="0.0.1"` and asserting `degraded`.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipelines/test_inference.py -v`
Expected: FAIL — `run_inference` doesn't accept `registry_path`/`work_dir`/`fetch_bytes` yet.

- [ ] **Step 3: Rewrite the serving logic in `src/microclimate/pipelines/inference.py`**

Update imports (add):
```python
from collections.abc import Callable
from microclimate.connectors.base import ConnectorError
from microclimate.connectors.http import http_get_bytes
from microclimate.contracts.registry import RegistryManifest, Task, manifest_key
from microclimate.publication.champion_loader import load_champion
from microclimate.publication.registry_store import read_registry
```
Add a safe registry read, `_serve_task`, and rewrite `run_inference` + `_assemble_forecast`.
`_serve_task` delegates the load to the shared `load_champion` (passing the same `registry_path`;
the extra cheap re-read keeps one code path rather than duplicating the download/load logic):
```python
def _read_registry_safe(registry_path: Path) -> RegistryManifest:
    """Read the registry; a missing OR corrupt file degrades to an empty manifest (never dark)."""
    try:
        return read_registry(registry_path)
    except Exception as exc:  # noqa: BLE001 — a bad registry must not stop the hourly product
        print(f"inference: registry unreadable ({type(exc).__name__}: {exc}); using baseline")
        return RegistryManifest()

def _serve_task(
    task: Task,
    manifest: RegistryManifest,
    matrix: pd.DataFrame,
    base: pd.DataFrame,
    config: DeploymentConfig,
    registry_path: Path,
    work_dir: Path,
    fetch_bytes: Callable[[str], bytes],
) -> tuple[str, pd.Series, bool]:
    base_col = "pred_temp_c" if task == "temp" else "pred_pop"
    entry = manifest.entries.get(manifest_key(config.deployment_id, task))
    if entry is None or entry.version == "baseline":
        return BASELINE_VERSION, base[base_col], False
    try:
        champion = load_champion(
            config.deployment_id, registry_path, task, work_dir, fetch_bytes=fetch_bytes
        )
        if champion is None:  # entry vanished between reads — treat as baseline, not degraded
            return BASELINE_VERSION, base[base_col], False
        preds = champion.predict(matrix)  # raises ValueError on stale feature_schema_version
        return entry.version, preds, False
    except (ConnectorError, ValueError, OSError) as exc:
        print(f"inference: champion '{entry.version}' unusable for {task} ({type(exc).__name__}: {exc}); baseline")
        return BASELINE_VERSION, base[base_col], True


def run_inference(
    config: DeploymentConfig,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    forecast_path: Path,
    issue_time: datetime,
    registry_path: Path,
    work_dir: Path,
    fetch_bytes: Callable[[str], bytes] = lambda url: http_get_bytes(url),
) -> ForecastDocument:
    """Build snapshot -> serve champion-or-baseline per task -> write forecast JSON."""
    snapshot = build_snapshot(config, issue_time, nwp, observations)
    matrix = build_features(snapshot, config)
    base = baseline_predictions(matrix, config.label.precip_occurrence_threshold_mm)
    manifest = _read_registry_safe(registry_path)

    tver, tpreds, tdeg = _serve_task("temp", manifest, matrix, base, config, registry_path, work_dir, fetch_bytes)
    pver, ppreds, pdeg = _serve_task("pop", manifest, matrix, base, config, registry_path, work_dir, fetch_bytes)

    frame = base.copy()
    frame["pred_temp_c"] = tpreds
    frame["pred_pop"] = ppreds
    status = "degraded" if (tdeg or pdeg) else "ok"
    doc = _assemble_forecast(
        config, frame, snapshot.issue_time, last_updated=snapshot.issue_time,
        model_versions={"temp": tver, "pop": pver}, status=status,
    )
    write_forecast(doc, forecast_path)
    return doc
```
Change `_assemble_forecast`'s signature to accept `model_versions: dict[Literal["temp","pop"], str]` and `status: Literal["ok", "stale", "degraded"]`, and use them instead of the hardcoded `{"temp": BASELINE_VERSION, ...}` / `status="ok"` (import `Literal` from `typing`). Keep the per-`ForecastStep` reshape identical.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipelines/test_inference.py -v`
Expected: the new + existing tests pass.

- [ ] **Step 5: FULL gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
```bash
git add src/microclimate/pipelines/inference.py tests/pipelines/test_inference.py
git commit -m "feat(inference): serve champion per task with baseline fallback + degraded status"
```

---

## Task 3: `main` + `inference.yml` gh-pages publish

**Files:**
- Modify: `src/microclimate/pipelines/inference.py` (`main`)
- Modify: `.github/workflows/inference.yml`

- [ ] **Step 1: Update `main()` in `inference.py`** to resolve the new params from env + write under a gh-pages root

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly inference for a deployment.")
    parser.add_argument("--deployment", required=True)
    args = parser.parse_args()

    config = load_deployment(args.deployment)
    nwp = cast(NWPSource, get_source(config.nwp.live_connector))
    station_keys = {config.target.connector_key, *(n.connector_key for n in config.neighbors)}
    observations = {k: cast(ObservationSource, get_source(k)) for k in station_keys}
    issue_time = _latest_hrdps_issue_time(datetime.now(UTC))

    root = Path(os.environ.get("FORECAST_OUTPUT_ROOT", "."))
    registry_path = Path(os.environ.get("REGISTRY_PATH", str(root / "registry.json")))
    work_dir = Path(os.environ.get("CHAMPION_CACHE_DIR", ".champion-cache"))

    run_inference(
        config,
        nwp=nwp,
        observations=observations,
        forecast_path=root / config.output.forecast_json,
        issue_time=issue_time,
        registry_path=registry_path,
        work_dir=work_dir,
    )
```
Add `import os` to the imports if not present.

- [ ] **Step 2: Edit `.github/workflows/inference.yml`** — clone gh-pages, run inference into it, push

Replace the `run` job's steps so it checks out gh-pages, runs inference per deployment writing into `gp/`, and pushes:
```yaml
  run:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    concurrency:
      group: gh-pages-publish      # gh-pages is shared (also written by training.yml)
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync

      - name: Check out (or bootstrap) gh-pages
        env:
          PAGES_URL: https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git
        run: |
          if git clone --depth 1 --branch gh-pages "$PAGES_URL" gp 2>/dev/null; then
            echo "Cloned gh-pages."
          else
            echo "Bootstrapping empty gh-pages."
            mkdir gp
          fi

      - name: Run inference for each deployment (into gh-pages worktree)
        env:
          FORECAST_OUTPUT_ROOT: gp
          REGISTRY_PATH: gp/registry.json
        run: |
          set -euo pipefail
          for f in config/deployments/*.yml; do
            id="$(basename "$f" .yml)"
            echo "::group::inference $id"
            uv run python -m microclimate.pipelines.inference --deployment "$id"
            echo "::endgroup::"
          done

      - name: Publish forecast JSON to gh-pages
        env:
          PAGES_URL: https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git
        run: |
          cd gp
          git init -q 2>/dev/null || true
          git add -A
          git -c user.name=ci -c user.email=ci@local commit -q -m "forecast: $(date -u +%FT%TZ)" || { echo "no changes"; exit 0; }
          git push -f "$PAGES_URL" HEAD:gh-pages
```
(The inference forecast write is the only thing this job changes on gh-pages; the `git push -f HEAD:gh-pages` from a fresh clone is safe for the forecast files. If `gp` was a fresh `git clone`, it already has the gh-pages history so the commit is a normal fast-forward; the `-f` covers the bootstrap case.)

Keep the existing `on: schedule (cron "23 * * * *") + workflow_dispatch` triggers.

- [ ] **Step 3: Validate YAML + full gate**

Run: `python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/inference.yml')); print('jobs', list(d['jobs']))"`
Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest -q`
Expected: YAML loads (`jobs ['run']`); gate green.

- [ ] **Step 4: Commit**

```bash
git add src/microclimate/pipelines/inference.py .github/workflows/inference.yml
git commit -m "feat(inference): main reads registry + publishes forecast JSON to gh-pages"
```

---

## Task 4: README status

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the "Project status" section**

State that inference now **serves the promoted champion** per task (loading it from `registry.json` via the shared `publication.champion_loader`), falls back to the baseline with `status="degraded"` when an expected champion is unusable (stale-schema champion refused), and publishes the forecast JSON to gh-pages. The training→publish→serve loop is now closed. Note the remaining intentional gaps only (e.g. `acis` retained-but-unused; the `stale` run-freshness status still deferred). Run `grep -rl NotImplementedError src/microclimate` and keep the list accurate.

- [ ] **Step 2: Final full gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
```bash
git add README.md
git commit -m "docs(readme): inference serves the champion + publishes forecast to gh-pages"
```

---

## Self-Review

**Spec coverage:** shared `load_champion` (Task 1 ↔ spec §1); `run_inference` champion-or-baseline + `_serve_task` + `_assemble_forecast` + corrupt-registry catch (Task 2 ↔ §2/§3 + error-handling); `main` + gh-pages publish (Task 3 ↔ §4); README (Task 4). Per-task `model_versions`, `degraded`-only-when-expected, stale-schema-refused-via-predict-ValueError, never-dark — all in Task 2. ✓

**Placeholder scan:** Step 3 of Task 2 shows a deliberately-discarded first sketch (clearly labeled "placeholder; replaced below") immediately followed by the real `_serve_task` — the engineer implements the real one. No TBD/TODO elsewhere; every code step is complete.

**Type consistency:** `_serve_task(task, manifest, matrix, base, config, registry_path, work_dir, fetch_bytes) -> (str, pd.Series, bool)` used consistently by `run_inference`. `load_champion(deployment_id, registry_path, task, work_dir, *, fetch_bytes)` matches Task 1's definition and Task 1's training-side call-site update. `_assemble_forecast` gains `model_versions`/`status` and Task 2 calls it with exactly those. Env names (`FORECAST_OUTPUT_ROOT`, `REGISTRY_PATH`, `CHAMPION_CACHE_DIR`) consistent between `main` and `inference.yml`.

**Execution-time checks (not placeholders):** Task 1 Step 4 says "read training.py to confirm which imports become unused" (ruff flags them); Task 2 Step 1 says reuse the existing `_FakeNWP`/`_FakeObs` in `test_inference.py` (their exact names are in that file). Both are real "read-then-edit," not guesses.

**Out of scope (per spec):** champion/challenger gate, registry format, persistent champion cache, the `stale` status, cross-workflow gh-pages lock.
