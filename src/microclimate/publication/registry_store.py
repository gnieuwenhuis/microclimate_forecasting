"""Read/update the champion registry manifest (L5, stub)."""

from __future__ import annotations

from pathlib import Path

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, Task


def read_registry(path: Path) -> RegistryManifest:
    raise NotImplementedError


def promote(
    manifest: RegistryManifest, task: Task, deployment_id: str, entry: RegistryEntry
) -> RegistryManifest:
    raise NotImplementedError
