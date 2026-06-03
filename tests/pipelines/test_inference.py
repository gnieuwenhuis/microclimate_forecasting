# tests/pipelines/test_inference.py
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource, SourceUnavailable
from microclimate.contracts.forecast import ForecastDocument
from microclimate.contracts.registry import RegistryEntry, RegistryManifest, manifest_key
from microclimate.pipelines.inference import run_inference
from microclimate.publication import champion_publisher as cp
from microclimate.publication.registry_store import write_registry
from tests.fakes import FakeNWP, FakeObs, make_config, make_forecast_frame, make_obs_frame

_T0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _make_fakes() -> tuple[DeploymentConfig, NWPSource, Mapping[str, ObservationSource]]:
    """Return (config, nwp, observations) for deployment_id='test', horizon=3, lag=2."""
    config = make_config(horizon_hours=3, lag_hours=2)
    leads = [1, 2, 3]
    nwp = FakeNWP(make_forecast_frame(_T0, leads))
    ts = [_T0 - timedelta(hours=h) for h in (2, 1, 0)]
    observations: dict[str, ObservationSource] = {
        "fake": FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    }
    return config, nwp, observations


def _registry_with(tmp_path: Path, entries: dict[str, RegistryEntry]) -> Path:
    p = tmp_path / "registry.json"
    write_registry(RegistryManifest(entries=entries), p)
    return p


def _real_temp_champion(
    tmp_path: Path,
    config: DeploymentConfig,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    issue_time: datetime,
) -> tuple[RegistryEntry, bytes]:
    from microclimate.features.feature_builder import build_features
    from microclimate.features.snapshot_builder import build_snapshot
    from microclimate.models.temp_model import TemperatureRegressor

    matrix = build_features(build_snapshot(config, issue_time, nwp, observations), config).copy()
    matrix["label_temp_c"] = matrix["nwp_temp_c"] + 1.0
    model = TemperatureRegressor()
    model.fit(matrix)
    version = "test-temp-20260603T0000Z"
    raw = cp.save_champion(model, tmp_path / "stage", version).read_bytes()
    entry = RegistryEntry(
        version=version,
        release_asset_url=f"https://example/{version}.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={"mae": 1.0},
    )
    return entry, raw


def test_run_inference_publishes_baseline(tmp_path: Path) -> None:
    config = make_config(horizon_hours=3, lag_hours=2)  # target T1 + neighbor N1, key "fake"
    leads = [1, 2, 3]
    nwp = FakeNWP(make_forecast_frame(_T0, leads))
    ts = [_T0 - timedelta(hours=h) for h in (2, 1, 0)]
    observations = {
        "fake": FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    }
    forecast_path = tmp_path / "forecasts" / "test.json"

    doc = run_inference(
        config,
        nwp=nwp,
        observations=observations,
        forecast_path=forecast_path,
        issue_time=_T0,
    )

    # forecast written + valid + equal to the returned doc
    assert doc is not None
    assert ForecastDocument.model_validate_json(forecast_path.read_text()) == doc
    assert len(doc.series) == 3
    assert [s.lead_hour for s in doc.series] == [1, 2, 3]
    assert doc.status == "ok"
    assert doc.model_versions == {"temp": "baseline", "pop": "baseline"}
    assert doc.attribution  # non-empty (ADR-0009)
    # PINNED temp_c=15.0 → passthrough; PINNED precip 0.5 ≥ threshold 0.2 → pop 1.0
    assert all(s.temp_c == 15.0 for s in doc.series)
    assert all(s.pop == 1.0 for s in doc.series)


def test_run_inference_normalizes_naive_issue_time_to_utc(tmp_path: Path) -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    leads = [1, 2, 3]
    nwp = FakeNWP(make_forecast_frame(_T0, leads))
    ts = [_T0 - timedelta(hours=h) for h in (2, 1, 0)]
    observations = {
        "fake": FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    }
    naive = datetime(2026, 6, 1, 0, 0)  # tz-naive → build_snapshot normalizes to _T0 UTC

    doc = run_inference(
        config,
        nwp=nwp,
        observations=observations,
        forecast_path=tmp_path / "f.json",
        issue_time=naive,
    )

    # Document issue_time is the normalized UTC value (would fail AwareDatetime if left naive),
    # consistent with the UTC valid_times.
    assert doc.issue_time == _T0
    assert doc.issue_time.tzinfo is not None
    assert doc.series[0].valid_time == _T0 + timedelta(hours=1)


def test_no_registry_serves_baseline_ok(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    doc = run_inference(
        config,
        nwp=nwp,
        observations=obs,
        forecast_path=tmp_path / "f.json",
        issue_time=it,
        registry_path=tmp_path / "absent.json",
        work_dir=tmp_path / "wd",
    )
    assert doc.status == "ok"
    assert doc.model_versions == {"temp": "baseline", "pop": "baseline"}
    assert (tmp_path / "f.json").exists()


def test_real_temp_champion_served(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    entry, raw = _real_temp_champion(tmp_path, config, nwp, obs, it)
    reg = _registry_with(tmp_path, {manifest_key("test", "temp"): entry})
    doc = run_inference(
        config,
        nwp=nwp,
        observations=obs,
        forecast_path=tmp_path / "f.json",
        issue_time=it,
        registry_path=reg,
        work_dir=tmp_path / "wd",
        fetch_bytes=lambda _u: raw,
    )
    assert doc.status == "ok"
    assert doc.model_versions["temp"] == entry.version
    assert doc.model_versions["pop"] == "baseline"
    # the served champion predicts nwp_temp_c + 1.0, so temps must differ from the 15.0 baseline
    assert any(abs(step.temp_c - 15.0) > 0.1 for step in doc.series)


def test_expected_champion_download_fails_is_degraded(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    entry = RegistryEntry(
        version="test-temp-x",
        release_asset_url="https://example/x.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={},
    )
    reg = _registry_with(tmp_path, {manifest_key("test", "temp"): entry})

    def _boom(_u: str) -> bytes:
        raise SourceUnavailable("download failed")

    doc = run_inference(
        config,
        nwp=nwp,
        observations=obs,
        forecast_path=tmp_path / "f.json",
        issue_time=it,
        registry_path=reg,
        work_dir=tmp_path / "wd",
        fetch_bytes=_boom,
    )
    assert doc.status == "degraded"
    assert doc.model_versions["temp"] == "baseline"


def test_corrupt_registry_treated_as_empty(tmp_path: Path) -> None:
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    bad = tmp_path / "registry.json"
    bad.write_text("{ not valid json")
    doc = run_inference(
        config,
        nwp=nwp,
        observations=obs,
        forecast_path=tmp_path / "f.json",
        issue_time=it,
        registry_path=bad,
        work_dir=tmp_path / "wd",
    )
    assert doc.status == "ok"
    assert doc.model_versions["temp"] == "baseline"


def test_stale_schema_champion_is_degraded(tmp_path: Path) -> None:
    """Champion trained on schema '0.0.1' raises ValueError on predict → status degraded."""
    config, nwp, obs = _make_fakes()
    it = datetime(2026, 6, 1, 0, tzinfo=UTC)

    from microclimate.features.feature_builder import build_features
    from microclimate.features.snapshot_builder import build_snapshot
    from microclimate.models.temp_model import TemperatureRegressor

    # Build a real matrix but stamp a fake (old) schema version so the saved model thinks
    # it was trained on "0.0.1".  We manipulate _feature_schema_version directly after fit so
    # that predict() raises ValueError when it sees the actual version "1.0.0" in the matrix.
    matrix = build_features(build_snapshot(config, it, nwp, obs), config).copy()
    matrix["label_temp_c"] = matrix["nwp_temp_c"] + 1.0
    model = TemperatureRegressor()
    model.fit(matrix)
    model._feature_schema_version = "0.0.1"  # type: ignore[reportPrivateUsage]  # simulate stale

    version = "test-temp-stale"
    raw = cp.save_champion(model, tmp_path / "stage", version).read_bytes()
    entry = RegistryEntry(
        version=version,
        release_asset_url=f"https://example/{version}.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={"mae": 2.0},
    )
    reg = _registry_with(tmp_path, {manifest_key("test", "temp"): entry})
    doc = run_inference(
        config,
        nwp=nwp,
        observations=obs,
        forecast_path=tmp_path / "f.json",
        issue_time=it,
        registry_path=reg,
        work_dir=tmp_path / "wd",
        fetch_bytes=lambda _u: raw,
    )
    assert doc.status == "degraded"
    assert doc.model_versions["temp"] == "baseline"


def test_latest_hrdps_issue_time_floors_to_published_6h_cycle() -> None:
    from microclimate.pipelines.inference import (
        _latest_hrdps_issue_time,  # type: ignore[reportPrivateUsage]
    )

    # 14:00Z minus ~4h publish lag = 10:00Z → floor to the 06Z run
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 14, 0, tzinfo=UTC)) == datetime(
        2026, 6, 1, 6, 0, tzinfo=UTC
    )
    # 16:30Z − 4h = 12:30Z → 12Z run
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 16, 30, tzinfo=UTC)) == datetime(
        2026, 6, 1, 12, 0, tzinfo=UTC
    )
    # 03:00Z − 4h = previous day 23:00Z → 18Z run (day rollover)
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 3, 0, tzinfo=UTC)) == datetime(
        2026, 5, 31, 18, 0, tzinfo=UTC
    )
    # naive input is treated as UTC
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 14, 0)) == datetime(
        2026, 6, 1, 6, 0, tzinfo=UTC
    )
