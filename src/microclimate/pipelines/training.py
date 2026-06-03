"""Monthly training pipeline (L6): backfill -> train -> gate -> promote -> publish (ADR-0016)."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, validate_config_sources
from microclimate.contracts.registry import RegistryEntry, RegistryManifest, Task
from microclimate.evaluation.metrics import nwp_pop_baseline
from microclimate.evaluation.publish_gate import GateResult, evaluate_challenger
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.pipelines.backfill import backfill_store, hrdps_issue_times
from microclimate.pipelines.training_data import assemble_from_store, temporal_split
from microclimate.publication import champion_publisher as cp
from microclimate.publication.champion_loader import load_champion
from microclimate.publication.registry_store import promote, read_registry, write_registry
from microclimate.training_store.store import TrainingStore

_REPO = os.environ.get("GITHUB_REPOSITORY", "gnieuwenhuis/microclimate_forecasting")


class _Saveable(Protocol):
    """Duck-type contract for any fitted model that can persist itself."""

    def save(self, path: Path) -> None: ...


def _do_promote(
    manifest: RegistryManifest,
    task: Task,
    config: DeploymentConfig,
    model: object,
    result: GateResult,
    output_dir: Path,
    now: datetime,
) -> RegistryManifest:
    version = cp.champion_version(config.deployment_id, task, now)
    cp.save_champion(cast(_Saveable, model), output_dir, version)
    entry = RegistryEntry(
        version=version,
        release_asset_url=cp.release_asset_url(_REPO, version),
        promoted_at=now,
        holdout_metrics=result.metrics,
    )
    return promote(manifest, task, config.deployment_id, entry)


def run_training(
    deployment_id: str,
    *,
    nwp: NWPSource | None = None,
    observations: Mapping[str, ObservationSource] | None = None,
    store: TrainingStore | None = None,
    output_dir: Path | None = None,
    registry_path: Path | None = None,
    now: datetime | None = None,
    start: datetime | None = None,
    holdout_months: int | None = None,
    calib_months: int = 3,
    do_backfill: bool = True,
) -> dict[str, object]:
    """Backfill -> train -> gate -> promote -> write registry + champion binaries.

    Promotion of zero tasks is a normal, successful outcome.
    """
    config = load_deployment(deployment_id)
    validate_config_sources(config)
    now = now or datetime.now(UTC)
    if start is None:
        seed = datetime.fromisoformat(config.training.seed.start)
        # naive seed.start is assumed UTC; an aware value is converted (don't clobber its tz).
        start = seed.astimezone(UTC) if seed.tzinfo is not None else seed.replace(tzinfo=UTC)
    holdout_months = (
        holdout_months if holdout_months is not None else config.training.holdout_months
    )
    output_dir = output_dir or Path(os.environ.get("CHAMPION_OUTPUT_DIR", "champions"))
    registry_path = registry_path or Path(os.environ.get("REGISTRY_PATH", "registry.json"))
    store = store or TrainingStore(Path(os.environ.get("TRAINING_STORE_ROOT", "training-store")))
    nwp = nwp or cast(NWPSource, get_source(config.nwp.historical_connector))
    if observations is None:
        keys = {config.target.connector_key, *(n.connector_key for n in config.neighbors)}
        observations = {k: cast(ObservationSource, get_source(k)) for k in keys}

    if do_backfill:
        n = backfill_store(
            config,
            nwp=nwp,
            observations=observations,
            store=store,
            issue_times=hrdps_issue_times(start, now),
        )
        print(f"backfill: +{n} new runs")

    rows = assemble_from_store(config, store)
    train, calib, test = temporal_split(
        rows, holdout_months=holdout_months, calib_months=calib_months
    )
    print(f"rows={len(rows):,} train={len(train):,} calib={len(calib):,} test={len(test):,}")

    manifest = read_registry(registry_path)
    promoted: list[str] = []
    results: dict[str, GateResult] = {}
    champ_dir = output_dir / "_champion"

    # temp
    temp = TemperatureRegressor()
    temp.fit(pd.concat([train, calib], ignore_index=True))
    res_t = evaluate_challenger(
        "temp",
        temp,
        load_champion(config.deployment_id, registry_path, "temp", champ_dir),
        test["nwp_temp_c"],
        test,
    )
    results["temp"] = res_t
    print(f"temp gate: {res_t.reason}")
    if res_t.promote:
        manifest = _do_promote(manifest, "temp", config, temp, res_t, output_dir, now)
        promoted.append("temp")

    # pop
    pop = PrecipOccurrenceClassifier()
    pop.fit(train)
    pop.calibrate(calib)
    pop_baseline = nwp_pop_baseline(test, config.label.precip_occurrence_threshold_mm)
    res_p = evaluate_challenger(
        "pop",
        pop,
        load_champion(config.deployment_id, registry_path, "pop", champ_dir),
        pop_baseline,
        test,
    )
    results["pop"] = res_p
    print(f"pop gate: {res_p.reason}")
    if res_p.promote:
        manifest = _do_promote(manifest, "pop", config, pop, res_p, output_dir, now)
        promoted.append("pop")

    if promoted:
        write_registry(manifest, registry_path)
    print(f"promoted: {promoted or 'none'}")
    return {
        "rows": len(rows),
        "promoted": promoted,
        "results": results,
        "registry_path": registry_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training for a deployment.")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--no-backfill", action="store_true")
    args = parser.parse_args()
    run_training(args.deployment, do_backfill=not args.no_backfill)


if __name__ == "__main__":
    main()
