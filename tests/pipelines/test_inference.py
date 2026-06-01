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
    observations = {
        "fake": FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    }
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
        store=TrainingStore(tmp_path / "store"),
        forecast_path=tmp_path / "f.json",
        issue_time=naive,
    )

    # Document issue_time is the normalized UTC value (would fail AwareDatetime if left naive),
    # consistent with the UTC valid_times and the logged snapshot.
    assert doc.issue_time == _T0
    assert doc.issue_time.tzinfo is not None
    assert doc.series[0].valid_time == _T0 + timedelta(hours=1)
