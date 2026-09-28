"""Shared translation of Garmin Health API epoch rows."""

import numpy as np
import pandas as pd

from .. import schema


def prepare_epoch_activity_samples(
    frame: pd.DataFrame,
    *,
    category_columns=("intensity", "activity_type"),
    source="garmin_health_api_epoch",
) -> pd.DataFrame:
    """Collapse Garmin category-split epochs into incremental step intervals.

    Garmin can emit several rows for one epoch, usually one per intensity or
    activity category. ``duration_in_ms`` describes the epoch and is therefore
    counted once, while ``activity_steps`` is summed across the category rows.
    Exact duplicate category rows are coalesced. If rows for one category
    disagree, the collapsed epoch retains a missing step count.
    """
    if frame.empty:
        return schema.empty_activity_frame(extra_columns=("source",))
    frame = frame.copy()
    required = {
        schema.LOCAL_TIME_COL,
        schema.UTC_TIME_COL,
        schema.DURATION_IN_MS_COL,
        schema.ACTIVITY_STEPS_COL,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Garmin epoch activity is missing canonical columns: " f"{sorted(missing)}"
        )
    frame[schema.ACTIVITY_STEPS_COL] = pd.to_numeric(
        frame[schema.ACTIVITY_STEPS_COL], errors="coerce"
    )
    frame.loc[frame[schema.ACTIVITY_STEPS_COL].lt(0), schema.ACTIVITY_STEPS_COL] = (
        np.nan
    )
    frame[schema.DURATION_IN_MS_COL] = pd.to_numeric(
        frame[schema.DURATION_IN_MS_COL], errors="coerce"
    )
    epoch_col = "__garmin_epoch_time"
    frame[epoch_col] = frame[schema.UTC_TIME_COL].where(
        frame[schema.UTC_TIME_COL].notna(), frame[schema.LOCAL_TIME_COL]
    )
    categories = [column for column in category_columns if column in frame]
    identity = [epoch_col, *categories]
    comparison = [
        *identity,
        schema.ACTIVITY_STEPS_COL,
        schema.DURATION_IN_MS_COL,
    ]
    frame = frame.drop_duplicates(comparison, keep="last")
    category_versions = frame.groupby(identity, dropna=False)[
        [schema.ACTIVITY_STEPS_COL, schema.DURATION_IN_MS_COL]
    ].nunique(dropna=True)
    conflicting_categories = category_versions.gt(1).any(axis=1)
    conflicting_times = set(
        category_versions.index[conflicting_categories].get_level_values(epoch_col)
    )

    aggregations = {
        schema.LOCAL_TIME_COL: "last",
        schema.UTC_TIME_COL: "last",
        schema.DURATION_IN_MS_COL: "max",
    }
    if schema.TIMEZONE_COL in frame:
        aggregations[schema.TIMEZONE_COL] = "last"
    grouped = frame.groupby(epoch_col, dropna=False)
    collapsed = grouped.agg(aggregations)
    collapsed[schema.ACTIVITY_STEPS_COL] = grouped[
        schema.ACTIVITY_STEPS_COL
    ].sum(min_count=1)
    collapsed = (
        collapsed.reset_index()
        .sort_values(epoch_col, kind="stable")
    )
    collapsed.loc[
        collapsed[epoch_col].isin(conflicting_times),
        schema.ACTIVITY_STEPS_COL,
    ] = np.nan
    collapsed = collapsed.drop(columns=epoch_col)
    collapsed["source"] = source
    return schema.prepare_activity_samples(collapsed)


__all__ = ["prepare_epoch_activity_samples"]
