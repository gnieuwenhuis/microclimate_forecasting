from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from microclimate.features.snapshot_builder import build_snapshot


def test_signature_takes_issue_time() -> None:
    params = inspect.signature(build_snapshot).parameters
    assert "issue_time" in params  # leakage-proof by signature


def test_builder_is_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        build_snapshot(
            config=None,  # type: ignore[arg-type]
            issue_time=datetime(2026, 5, 30, tzinfo=UTC),
            nwp=None,  # type: ignore[arg-type]
            observations={},
        )
