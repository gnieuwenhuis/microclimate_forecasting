"""Schema of the accumulating training store (L0). strict=False — feature columns vary."""

from __future__ import annotations

import pandera.pandas as pa

TRAINING_ROW = pa.DataFrameSchema(
    {
        "schema_version": pa.Column(str),
        "deployment_id": pa.Column(str),
        "issue_time": pa.Column("datetime64[ns, UTC]"),
        "lead_hour": pa.Column(int, pa.Check.in_range(1, 48)),  # type: ignore[reportUnknownMemberType]
        "valid_time": pa.Column("datetime64[ns, UTC]"),
        "label_temp_c": pa.Column(float, nullable=True),
        "label_precip_occurrence": pa.Column(int, pa.Check.isin([0, 1]), nullable=True),  # type: ignore[reportUnknownMemberType]
    },
    strict=False,
    coerce=True,
)
