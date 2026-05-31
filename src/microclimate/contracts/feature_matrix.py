"""Schema of the label-free feature matrix (L0). strict=False — feature columns vary.

Produced by features.build_features from a FeatureSnapshot. Distinct from TRAINING_ROW
(training_store.py), which is this plus labels; the two version independently.
"""

from __future__ import annotations

import pandera.pandas as pa

# Version of the DERIVED-feature set, distinct from SNAPSHOT_SCHEMA_VERSION (the raw-snapshot
# contract). Bump when the set/meaning of derived feature columns changes, so a model trained
# on a stale feature set is refused rather than silently misread (champion/challenger, ADR-0006).
FEATURE_SCHEMA_VERSION = "1.0.0"

FEATURE_ROW = pa.DataFrameSchema(
    {
        "feature_schema_version": pa.Column(str),
        "deployment_id": pa.Column(str),
        "issue_time": pa.Column("datetime64[ns, UTC]"),
        "lead_hour": pa.Column(int, pa.Check.in_range(1, 48)),  # type: ignore[reportUnknownMemberType]
        "valid_time": pa.Column("datetime64[ns, UTC]"),
    },
    strict=False,
    coerce=True,
)
