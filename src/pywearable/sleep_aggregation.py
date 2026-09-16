"""Aggregation of natural-resolution nightly sleep trajectories."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from . import sleep_schema
from .utils import time as time_utils


TEMPORAL_FEATURES = {
    "bedtime",
    "sleep_onset",
    "wakeup",
    "getup",
    "midpoint",
}
CALENDAR_RESOLUTIONS = {"daily", "weekly", "monthly", "yearly", "day", "week", "month", "year"}


def _normalize_input(
    sleep_statistics: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(sleep_statistics, pd.Series):
        if sleep_statistics.name is None:
            raise ValueError("a sleep statistic Series must have a name")
        frame = sleep_statistics.to_frame()
    elif isinstance(sleep_statistics, pd.DataFrame):
        frame = sleep_statistics.copy()
    else:
        raise ValueError("sleep_statistics must be a pandas Series or DataFrame")
    required_index = [sleep_schema.USER_COL, sleep_schema.CALENDAR_DATE_COL]
    if list(frame.index.names) != required_index:
        raise ValueError(
            "sleep_statistics must use a (user_id, calendar_date) MultiIndex"
        )
    frame = frame.reset_index()
    frame[sleep_schema.CALENDAR_DATE_COL] = pd.to_datetime(
        frame[sleep_schema.CALENDAR_DATE_COL], errors="coerce"
    ).dt.normalize()
    if frame[sleep_schema.CALENDAR_DATE_COL].isna().any():
        raise ValueError("calendar_date values must be valid datetimes")
    if frame.duplicated(required_index).any():
        raise ValueError("sleep_statistics must contain at most one row per night")
    return frame


def _resolve_operations(columns, aggregation):
    if isinstance(aggregation, dict):
        unknown = set(aggregation).difference(columns)
        if unknown:
            raise ValueError(
                "aggregation contains unknown features: " + ", ".join(sorted(unknown))
            )
        return {column: aggregation.get(column, "mean") for column in columns}
    return {column: aggregation for column in columns}


def _aggregate_numeric(values: pd.Series, operation):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    if callable(operation):
        return operation(values)
    allowed = {"mean", "median", "min", "max", "std", "sum", "count"}
    if operation not in allowed:
        raise ValueError(
            f"unsupported numeric aggregation {operation!r}; choose one of {sorted(allowed)}"
        )
    return getattr(values, operation)()


def _aggregate_temporal(
    values: pd.Series,
    calendar_dates: pd.Series,
    period_start: pd.Timestamp,
    operation: str | Callable,
):
    values = pd.to_datetime(values, errors="coerce")
    valid = values.notna() & calendar_dates.notna()
    if not valid.any():
        return pd.NaT
    offsets = values[valid] - calendar_dates[valid]
    offset_seconds = offsets.dt.total_seconds()
    if callable(operation):
        result = operation(offsets)
        if isinstance(result, pd.Timedelta):
            return period_start + result
        return result
    if operation == "mean":
        seconds = offset_seconds.mean()
    elif operation == "median":
        seconds = offset_seconds.median()
    elif operation == "min":
        seconds = offset_seconds.min()
    elif operation == "max":
        seconds = offset_seconds.max()
    elif operation == "std":
        return offset_seconds.std() / 60
    elif operation == "count":
        return int(valid.sum())
    else:
        raise ValueError(
            "temporal sleep features support mean, median, min, max, std, or count"
        )
    return period_start + pd.to_timedelta(seconds, unit="s")


def aggregate_sleep_statistics(
    sleep_statistics: pd.Series | pd.DataFrame,
    resolution: str = "weekly",
    aggregation: str | Callable | dict = "mean",
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Aggregate nightly features into exact calendar periods.

    Edge periods are intersected with the requested ``[start_date, end_date)``
    interval. Timestamp features are aggregated as offsets from each night's
    ``calendar_date`` so times around midnight remain adjacent. Their ``std``
    operation is returned in minutes; other supported operations return a
    timestamp anchored to the output period start.
    """
    if resolution not in CALENDAR_RESOLUTIONS:
        raise ValueError("sleep aggregation resolution must be calendar based")
    frame = _normalize_input(sleep_statistics)
    features = [
        column
        for column in frame.columns
        if column not in (sleep_schema.USER_COL, sleep_schema.CALENDAR_DATE_COL)
    ]
    if not features:
        raise ValueError("sleep_statistics must contain at least one feature")
    operations = _resolve_operations(features, aggregation)
    spec = time_utils.get_resampling_spec(resolution, time_reference="local")

    available_start = frame[sleep_schema.CALENDAR_DATE_COL].min()
    available_end = frame[sleep_schema.CALENDAR_DATE_COL].max() + pd.Timedelta(days=1)
    requested_start = (
        available_start
        if start_date is None
        else pd.Timestamp(start_date).normalize()
    )
    requested_end = (
        available_end
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    if requested_end <= requested_start:
        raise ValueError("end_date must be after start_date")
    frame = frame[
        frame[sleep_schema.CALENDAR_DATE_COL].ge(requested_start.normalize())
        & frame[sleep_schema.CALENDAR_DATE_COL].lt(requested_end)
    ]
    window_starts = time_utils.get_calendar_window_starts(
        requested_start, requested_end, spec.rule
    )
    window_ends = list(window_starts[1:]) + [requested_end]

    records = []
    users = sleep_statistics.index.get_level_values(sleep_schema.USER_COL).unique()
    for user in users:
        user_frame = frame[frame[sleep_schema.USER_COL] == user]
        for period_start, period_end in zip(window_starts, window_ends):
            nights = user_frame[
                user_frame[sleep_schema.CALENDAR_DATE_COL].ge(period_start)
                & user_frame[sleep_schema.CALENDAR_DATE_COL].lt(period_end)
            ]
            expected_nights = len(
                pd.date_range(
                    period_start.normalize(), period_end, freq="D", inclusive="left"
                )
            )
            record = {
                sleep_schema.USER_COL: user,
                sleep_schema.CALENDAR_DATE_COL: period_start,
                "period_end": period_end,
                "expected_nights": expected_nights,
            }
            for feature in features:
                values = nights[feature]
                valid_count = int(values.notna().sum())
                if feature in TEMPORAL_FEATURES:
                    value = _aggregate_temporal(
                        values,
                        nights[sleep_schema.CALENDAR_DATE_COL],
                        period_start,
                        operations[feature],
                    )
                else:
                    value = _aggregate_numeric(values, operations[feature])
                record[feature] = value
                record[f"{feature}_count"] = valid_count
                record[f"{feature}_coverage"] = (
                    valid_count / expected_nights * 100 if expected_nights else np.nan
                )
            records.append(record)
    if not records:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=[sleep_schema.USER_COL, sleep_schema.CALENDAR_DATE_COL]
        )
        return pd.DataFrame(index=index)
    return (
        pd.DataFrame(records)
        .set_index([sleep_schema.USER_COL, sleep_schema.CALENDAR_DATE_COL])
        .sort_index()
    )
