"""Unit tests for the pure helpers behind build_snapshot."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from microclimate.features.snapshot_builder import (
    _temporal_features,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
)


def test_temporal_features_keys_and_values() -> None:
    # 2026-01-01 06:00 UTC → day-of-year 1, hour 6.
    t0 = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    feats = _temporal_features(t0)

    assert set(feats) == {"t0_hour_sin", "t0_hour_cos", "t0_doy_sin", "t0_doy_cos"}
    assert feats["t0_hour_sin"] == math.sin(2 * math.pi * 6 / 24.0)
    assert feats["t0_hour_cos"] == math.cos(2 * math.pi * 6 / 24.0)
    assert feats["t0_doy_sin"] == math.sin(2 * math.pi * 1 / 365.25)
    assert feats["t0_doy_cos"] == math.cos(2 * math.pi * 1 / 365.25)
