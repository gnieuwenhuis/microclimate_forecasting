"""Stateless inference pipeline (ADR-0019): snapshot → champion/baseline forecast → write JSON.

No snapshot logging; training data is collected via a separate retrain-time backfill.
Registry/champion-loading: load_champion per task; fall back to baseline on failure; mark
status="degraded" only when an expected champion (a real registry entry) can't be served.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.http import http_get_bytes
from microclimate.connectors.registry import get_source
from microclimate.contracts.forecast import FORECAST_SCHEMA_VERSION, ForecastDocument, ForecastStep
from microclimate.contracts.registry import RegistryManifest, Task, manifest_key
from microclimate.features.feature_builder import build_features
from microclimate.features.snapshot_builder import build_snapshot
from microclimate.models.baseline import BASELINE_VERSION, baseline_predictions
from microclimate.publication.champion_loader import load_champion
from microclimate.publication.forecast_writer import write_forecast
from microclimate.publication.registry_store import read_registry

_ATTRIBUTION = [
    "Weather data by Open-Meteo.com (CC-BY 4.0)",
    "Data Source: Environment and Climate Change Canada (ECCC)",
]

_HRDPS_PUBLISH_LAG = timedelta(
    hours=4
)  # heuristic: Open-Meteo makes a run available ~3-4 h after init


def _latest_hrdps_issue_time(now: datetime) -> datetime:
    """Most recent HRDPS run init time (00/06/12/18 UTC) likely available via Open-Meteo by ``now``.

    HRDPS runs four times daily; Open-Meteo typically makes each run available ~3-4 h after its
    init time. Subtracting the publish lag then flooring to the 6-hourly cycle yields a run that
    should be available. If the chosen run is not yet available, ``build_snapshot`` propagates the
    connector error (``ForecastUnavailable`` or ``SourceUnavailable``) and the next hourly Action
    run retries — fail-safe.
    """
    t = (
        now.astimezone(UTC) if now.tzinfo is not None else now.replace(tzinfo=UTC)
    ) - _HRDPS_PUBLISH_LAG
    run_hour = (t.hour // 6) * 6
    return t.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def _read_registry_safe(registry_path: Path) -> RegistryManifest:
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
) -> tuple[str, pd.Series, bool]:  # type: ignore[type-arg]
    """Return (version, predictions, degraded) for a single task.

    degraded=True only when a real registry entry exists but can't be loaded/predicted.
    """
    base_col = "pred_temp_c" if task == "temp" else "pred_pop"
    entry = manifest.entries.get(manifest_key(config.deployment_id, task))
    if entry is None or entry.version == "baseline":
        return BASELINE_VERSION, base[base_col], False

    # Loading an expected champion is downloading + deserializing an external, untrusted asset:
    # ANY failure (network, truncated/corrupt joblib -> EOFError/UnpicklingError/KeyError, etc.)
    # must degrade to baseline, never dark the hourly product.
    try:
        champion = load_champion(
            config.deployment_id, registry_path, task, work_dir, fetch_bytes=fetch_bytes
        )
    except Exception as exc:  # noqa: BLE001 — expected-champion load failure must fall back
        print(
            f"inference: champion '{entry.version}' failed to load for {task} "
            f"({type(exc).__name__}: {exc}); using baseline"
        )
        return BASELINE_VERSION, base[base_col], True
    if champion is None:  # entry vanished between reads — not degraded
        return BASELINE_VERSION, base[base_col], False

    # Prediction: a stale feature_schema_version is refused (ValueError) -> degraded baseline.
    # Other errors (e.g. KeyError from a missing feature column) are real bugs -> propagate loud.
    try:
        preds = champion.predict(matrix)
    except ValueError as exc:
        print(
            f"inference: champion '{entry.version}' refused for {task} "
            f"({type(exc).__name__}: {exc}); using baseline"
        )
        return BASELINE_VERSION, base[base_col], True
    return entry.version, preds, False


def _assemble_forecast(
    config: DeploymentConfig,
    preds: pd.DataFrame,
    issue_time: datetime,
    last_updated: datetime,
    model_versions: dict[Literal["temp", "pop"], str],
    status: Literal["ok", "stale", "degraded"],
) -> ForecastDocument:
    """Reshape per-(lead) predictions into a ForecastDocument.

    ADR-0012: the pipeline owns this reshape.
    """
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
        status=status,
        model_versions=model_versions,
        attribution=_ATTRIBUTION,
        series=series,
    )


def run_inference(
    config: DeploymentConfig,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    forecast_path: Path,
    issue_time: datetime,
    last_updated: datetime | None = None,
    registry_path: Path | None = None,
    work_dir: Path | None = None,
    fetch_bytes: Callable[[str], bytes] = lambda url: http_get_bytes(url),
) -> ForecastDocument:
    """Build a snapshot → champion/baseline forecast → write JSON. Stateless (ADR-0019).

    ``issue_time`` is the HRDPS model-cycle init time the forecast is built from; ``last_updated``
    is when *this document* was (re)generated — the wall-clock publish time, defaulting to
    ``datetime.now(UTC)`` (injectable for deterministic tests). They differ: an hourly re-run of
    the same cycle keeps ``issue_time`` but advances ``last_updated``.

    When ``registry_path`` is None (or absent/unreadable), serves baseline for all tasks.
    Status precedence: ``degraded`` (an expected champion — a real registry entry — failed to
    load/predict) > ``stale`` (the snapshot horizon was truncated below ``horizon_hours``) >
    ``ok``. So even a baseline-only run is ``stale`` when truncated, and ``ok`` otherwise.
    """
    published_at = last_updated if last_updated is not None else datetime.now(UTC)
    snapshot = build_snapshot(config, issue_time, nwp, observations)
    truncated = len(snapshot.lead_hours) < config.horizon_hours
    matrix = build_features(snapshot, config)
    base = baseline_predictions(matrix, config.label.precip_occurrence_threshold_mm)

    if registry_path is None:
        doc = _assemble_forecast(
            config,
            base,
            snapshot.issue_time,
            last_updated=published_at,
            model_versions={"temp": BASELINE_VERSION, "pop": BASELINE_VERSION},
            status="stale" if truncated else "ok",
        )
        write_forecast(doc, forecast_path)
        return doc

    manifest = _read_registry_safe(registry_path)
    _work_dir: Path = work_dir if work_dir is not None else forecast_path.parent / ".champion_cache"

    tver, tpreds, tdeg = _serve_task(
        "temp", manifest, matrix, base, config, registry_path, _work_dir, fetch_bytes
    )
    pver, ppreds, pdeg = _serve_task(
        "pop", manifest, matrix, base, config, registry_path, _work_dir, fetch_bytes
    )

    frame = base.copy()
    frame["pred_temp_c"] = tpreds
    frame["pred_pop"] = ppreds
    status: Literal["ok", "stale", "degraded"]
    if tdeg or pdeg:
        status = "degraded"
    elif truncated:
        status = "stale"
    else:
        status = "ok"
    doc = _assemble_forecast(
        config,
        frame,
        snapshot.issue_time,
        last_updated=published_at,
        model_versions={"temp": tver, "pop": pver},
        status=status,
    )
    write_forecast(doc, forecast_path)
    return doc


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


if __name__ == "__main__":
    main()
