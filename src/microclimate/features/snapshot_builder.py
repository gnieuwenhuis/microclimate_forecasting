"""The single, only path that produces a FeatureSnapshot (L3).

As-of / no-leakage: this is the only entry point, it takes issue_time, and the only obs
access is bounded to timestamp <= issue_time. There is no parameter for future data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.contracts.snapshot import FeatureSnapshot


def build_snapshot(
    config: DeploymentConfig,
    issue_time: datetime,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
) -> FeatureSnapshot:
    raise NotImplementedError
