"""Load and schema-validate deployment configs (L1). Imports only contracts/schema."""

from __future__ import annotations

from pathlib import Path

import yaml

from microclimate.config.schema import DeploymentConfig

# repo_root/config/deployments — parents[3] from src/microclimate/config/loader.py
DEPLOYMENTS_DIR = Path(__file__).resolve().parents[3] / "config" / "deployments"


def list_deployments(directory: Path = DEPLOYMENTS_DIR) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.yml"))


def load_deployment(deployment_id: str, directory: Path = DEPLOYMENTS_DIR) -> DeploymentConfig:
    path = directory / f"{deployment_id}.yml"
    if not path.exists():
        raise FileNotFoundError(f"No deployment config: {path}")
    data = yaml.safe_load(path.read_text())
    return DeploymentConfig.model_validate(data)
