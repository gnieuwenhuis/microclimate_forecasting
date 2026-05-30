"""Champion-pointer manifest (L0)."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

Task = Literal["temp", "pop"]


def manifest_key(deployment_id: str, task: Task) -> str:
    """Canonical manifest key: '{deployment_id}/{task}'."""
    return f"{deployment_id}/{task}"


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    release_asset_url: str
    promoted_at: AwareDatetime
    holdout_metrics: dict[str, float]


class RegistryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: dict[str, RegistryEntry] = {}
