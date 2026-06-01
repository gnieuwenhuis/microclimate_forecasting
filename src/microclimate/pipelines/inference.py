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
    """Reshape per-(lead) baseline predictions into a ForecastDocument.

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
