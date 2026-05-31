# notebooks/model_dev.py
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Local model development — train & explore the temp and PoP models
#
# Thin notebook: all logic lives in `microclimate.*` (tested) and is exercised by the CI
# smoke test. This file only orchestrates and plots. Open it as a notebook with jupytext or
# VS Code. Requires the `notebook` dependency group: `uv sync --group notebook`.
# CaSPAr historical access must be configured for the chosen deployment.

# %%
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from microclimate.config.loader import load_deployment
from microclimate.connectors.registry import get_source
from microclimate.evaluation.metrics import (
    pop_skill_by_lead,
    reliability_table,
    temp_skill_by_lead,
)
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.pipelines.training_data import assemble_or_load, chronological_split

DEPLOYMENT_ID = "lethbridge"
START = datetime(2024, 1, 1, 0, tzinfo=UTC)
N_ISSUE_TIMES = 24 * 60  # ~60 days of hourly issue times
ARTIFACTS = Path("notebooks/_artifacts")

# %%
config = load_deployment(DEPLOYMENT_ID)
nwp = get_source(config.nwp.historical_connector)
station_keys = {config.target.connector_key, *[n.connector_key for n in config.neighbors]}
observations = {k: get_source(k) for k in station_keys}  # type: ignore[misc]
issue_times = [START + timedelta(hours=i) for i in range(N_ISSUE_TIMES)]

# %%
rows = assemble_or_load(
    config,
    nwp,
    observations,
    issue_times,  # type: ignore[arg-type]
    cache_path=ARTIFACTS / f"{DEPLOYMENT_ID}_rows.parquet",
)
print(f"{len(rows):,} rows  |  {rows['issue_time'].nunique()} issue times")
rows.head()

# %%
train, calib, test = chronological_split(rows, train_frac=0.6, calib_frac=0.2)
print(f"train={len(train):,}  calib={len(calib):,}  test={len(test):,}")

# %%
temp = TemperatureRegressor()
temp.fit(pd.concat([train, calib], ignore_index=True))

pop = PrecipOccurrenceClassifier()
pop.fit(train)
pop.calibrate(calib)

test = test.copy()
test["pred_temp_c"] = temp.predict(test).to_numpy()
test["pred_pop"] = pop.predict(test).to_numpy()
test["baseline_pop"] = (
    test["nwp_precip_mm"] >= config.label.precip_occurrence_threshold_mm
).astype(float)

# %% [markdown]
# ## Temperature: skill vs raw-HRDPS baseline, by lead hour

# %%
ts = temp_skill_by_lead(test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(ts["lead_hour"], ts["rmse"], label="model RMSE")
ax1.plot(ts["lead_hour"], ts["baseline_rmse"], label="HRDPS RMSE")
ax1.set_xlabel("lead hour")
ax1.set_ylabel("°C")
ax1.legend()
ax1.set_title("RMSE")
ax2.axhline(0, color="grey", lw=0.8)
ax2.plot(ts["lead_hour"], ts["skill"])
ax2.set_xlabel("lead hour")
ax2.set_title("RMSE skill (>0 beats HRDPS)")
plt.tight_layout()

# %% [markdown]
# ## Temperature: predicted vs actual & residuals

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(test["label_temp_c"], test["pred_temp_c"], s=4, alpha=0.3)
lims = [test["label_temp_c"].min(), test["label_temp_c"].max()]
ax1.plot(lims, lims, color="red", lw=1)
ax1.set_xlabel("observed °C")
ax1.set_ylabel("predicted °C")
ax1.set_title("pred vs actual")
ax2.scatter(test["pred_temp_c"], test["pred_temp_c"] - test["label_temp_c"], s=4, alpha=0.3)
ax2.axhline(0, color="red", lw=1)
ax2.set_xlabel("predicted °C")
ax2.set_ylabel("residual °C")
ax2.set_title("residuals")
plt.tight_layout()

# %% [markdown]
# ## PoP: Brier Skill Score by lead, and the reliability diagram

# %%
ps = pop_skill_by_lead(test)
rel = reliability_table(test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.axhline(0, color="grey", lw=0.8)
ax1.plot(ps["lead_hour"], ps["bss"])
ax1.set_xlabel("lead hour")
ax1.set_title("Brier Skill Score (>0 beats HRDPS)")
ax2.plot([0, 1], [0, 1], color="grey", lw=1, label="perfect")
ax2.plot(rel["mean_pred"], rel["observed_freq"], marker="o", label="model")
ax2.set_xlabel("predicted PoP")
ax2.set_ylabel("observed frequency")
ax2.set_title("reliability")
ax2.legend()
plt.tight_layout()

# %% [markdown]
# ## Feature importances

# %%
imp = (
    pd.Series(
        temp._model.feature_importances_,
        index=temp._features,  # noqa: SLF001
    )
    .sort_values(ascending=False)
    .head(20)
)
imp.iloc[::-1].plot.barh(figsize=(8, 6), title="Temp model — top feature importances")
plt.tight_layout()

# %% [markdown]
# ## Save the locally-trained models (gitignored)

# %%
ARTIFACTS.mkdir(parents=True, exist_ok=True)
temp.save(ARTIFACTS / f"{DEPLOYMENT_ID}_temp.joblib")
pop.save(ARTIFACTS / f"{DEPLOYMENT_ID}_pop.joblib")
print("saved to", ARTIFACTS)
