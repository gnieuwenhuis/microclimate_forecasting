from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.connectors.base import HistoricalCoverage, NWPSource, ObservationSource
from microclimate.contracts.forecast_frame import FORECAST_FRAME
from microclimate.contracts.observation import OBSERVATION_FRAME
from microclimate.contracts.physical_vars import PHYSICAL_VARS
from microclimate.contracts.registry import manifest_key
from microclimate.publication.registry_store import read_registry
from microclimate.training_store.store import TrainingStore

_PINNED: dict[str, object] = {
    "temp_c": 10.0,
    "dewpoint_c": 5.0,
    "surface_pressure_hpa": 900.0,
    "precip_mm": 0.0,
    "cloud_cover_fraction": 0.5,
    "solar_radiation_wm2": 100.0,
    "wind_speed_ms": 3.0,
    "wind_dir_deg": 180.0,
}


class _FakeNWP(NWPSource):
    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(self, issue_time, lat, lon, lead_hours):  # type: ignore[override]
        rows: list[dict[str, object]] = []
        for h in lead_hours:
            # Introduce a systematic lead-hour-proportional warm bias (+0.5 °C/12 h) so
            # the temp regressor can learn to remove it and achieve lower MAE than raw NWP.
            r: dict[str, object] = {
                "issue_time": pd.Timestamp(issue_time),
                "lead_hour": int(h),
                "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=int(h)),
                **_PINNED,
                "temp_c": cast(float, _PINNED["temp_c"]) + int(h) * 0.5 / 12.0,
            }
            rows.append(r)
        return FORECAST_FRAME.validate(pd.DataFrame(rows))


class _FakeObs(ObservationSource):
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id, start, end):  # type: ignore[override]
        ts = pd.date_range(
            pd.Timestamp(start).floor("h"), pd.Timestamp(end).ceil("h"), freq="h", tz="UTC"
        )
        data: dict[str, object] = {"station_id": [station_id] * len(ts), "timestamp": list(ts)}
        for v in PHYSICAL_VARS:
            data[v] = [_PINNED[v]] * len(ts)
            data[f"{v}_present"] = [True] * len(ts)
        # precip alternates so the PoP labels carry both classes
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

    start = datetime(2024, 1, 1, tzinfo=UTC)
    now = start + timedelta(
        days=150
    )  # ~5 months so temporal_split(holdout 3 / calib 1) is non-empty

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

    assert cast(int, summary["rows"]) > 0
    promoted = cast(list[str], summary["promoted"])
    assert "temp" in promoted  # trained temp beats the constant baseline on synthetic labels
    manifest = read_registry(registry_path)
    key = manifest_key("lethbridge", "temp")
    assert key in manifest.entries
    entry = manifest.entries[key]
    assert entry.release_asset_url.endswith(".joblib")
    assert (out_dir / f"{entry.version}.joblib").exists()
