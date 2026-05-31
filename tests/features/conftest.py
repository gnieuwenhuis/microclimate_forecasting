# tests/features/conftest.py
"""Hermetic fixtures for feature tests — re-exported from the shared tests.fakes module."""

from __future__ import annotations

from tests.fakes import (
    PHYS,
    PINNED,
    FakeNWP,
    FakeObs,
    make_config,
    make_forecast_frame,
    make_obs_frame,
)

__all__ = [
    "PHYS",
    "PINNED",
    "FakeNWP",
    "FakeObs",
    "make_config",
    "make_forecast_frame",
    "make_obs_frame",
]
