"""Read/update the champion registry manifest (L5)."""

from __future__ import annotations

from pathlib import Path

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, Task, manifest_key


def read_registry(path: Path) -> RegistryManifest:
    """Parse registry.json, or an empty manifest if the file is absent."""
    if not path.exists():
        return RegistryManifest()
    return RegistryManifest.model_validate_json(path.read_text())


def promote(
    manifest: RegistryManifest, task: Task, deployment_id: str, entry: RegistryEntry
) -> RegistryManifest:
    """Return a new manifest with the (deployment_id, task) entry set (immutable update)."""
    entries = dict(manifest.entries)
    entries[manifest_key(deployment_id, task)] = entry
    return RegistryManifest(entries=entries)


def write_registry(manifest: RegistryManifest, path: Path) -> None:
    """Serialize the manifest to registry.json (pretty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))
