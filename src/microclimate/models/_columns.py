"""Model-input column selection shared by both wrappers (L4)."""

from __future__ import annotations

import pandas as pd

# Everything that is metadata/labels, not a model input. lead_hour IS a feature (ADR-0004).
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "feature_schema_version",
        "deployment_id",
        "issue_time",
        "valid_time",
        "label_temp_c",
        "label_precip_occurrence",
    }
)


def feature_columns(rows: pd.DataFrame) -> list[str]:
    """Ordered model-input columns: every column except metadata and labels."""
    return [c for c in rows.columns if c not in NON_FEATURE_COLUMNS]


def single_feature_schema_version(rows: pd.DataFrame) -> str:
    """The rows' one feature_schema_version, or raise if they mix versions.

    Mixing versions in a single fit would train a model whose recorded version is arbitrary
    (the first row's) yet whose inputs are inconsistent — refuse it loudly.
    """
    versions = rows["feature_schema_version"].unique()
    if len(versions) != 1:
        raise ValueError(
            f"rows mix feature_schema_versions {sorted(map(str, versions))}; "
            "refusing to fit on inconsistent feature sets."
        )
    return str(versions[0])
