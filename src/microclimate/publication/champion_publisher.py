"""Champion model publishing helpers (L5): deterministic naming + local staging.

The deterministic version/tag/asset names let registry.json reference the Release asset URL
*before* the upload happens; the workflow then uploads to exactly that tag/filename. The actual
`gh release upload` + gh-pages push live in the training workflow (this module is offline).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from microclimate.contracts.registry import Task

_TAG_PREFIX = "champion-"


class _Saveable(Protocol):
    """Duck-type contract for any fitted model that can persist itself."""

    def save(self, path: Path) -> None: ...


def champion_version(deployment_id: str, task: Task, run_time: datetime) -> str:
    """Deterministic version string from the run time, e.g. lethbridge-temp-20260603T1405Z.

    The ``Z`` suffix asserts UTC, so normalise first: a naive ``run_time`` is assumed UTC,
    an aware one is converted — otherwise the encoded time wouldn't match the ``Z`` label and
    the version/tag/URL would point at the wrong Release asset.
    """
    utc = run_time.astimezone(UTC) if run_time.tzinfo is not None else run_time.replace(tzinfo=UTC)
    return f"{deployment_id}-{task}-{utc:%Y%m%dT%H%M}Z"


def release_tag(version: str) -> str:
    return f"{_TAG_PREFIX}{version}"


def asset_filename(version: str) -> str:
    return f"{version}.joblib"


def release_asset_url(repo: str, version: str) -> str:
    """Public download URL for the asset the workflow will upload to release_tag(version)."""
    return (
        f"https://github.com/{repo}/releases/download/"
        f"{release_tag(version)}/{asset_filename(version)}"
    )


def save_champion(model: _Saveable, out_dir: Path, version: str) -> Path:
    """Persist the fitted model to out_dir/<version>.joblib (the workflow uploads from here)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / asset_filename(version)
    model.save(path)
    return path
