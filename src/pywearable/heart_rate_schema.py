"""Canonical heart-rate columns shared by loaders and feature extraction."""

import pandas as pd


UTC_TIME_COL = "utc_time"
LOCAL_TIME_COL = "local_time"
TIMEZONE_COL = "timezone"
HEART_RATE_COL = "heart_rate"
TIMESTAMP_CONFLICT_COL = "timestamp_conflict"

TIME_COLUMNS = (UTC_TIME_COL, LOCAL_TIME_COL)
REQUIRED_COLUMNS = (UTC_TIME_COL, LOCAL_TIME_COL, TIMEZONE_COL, HEART_RATE_COL)


def validate_time_col(time_col: str) -> str:
    """Validate the timestamp used to filter provider samples."""
    if time_col not in TIME_COLUMNS:
        raise ValueError(f"time_col must be one of {TIME_COLUMNS}, not {time_col}")
    return time_col


def normalize_time_bound(value, time_col: str):
    """Return a timezone-naive bound in the selected clock."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        if time_col == UTC_TIME_COL:
            timestamp = timestamp.tz_convert("UTC")
        timestamp = timestamp.tz_localize(None)
    return timestamp


def filter_by_time(
    frame: pd.DataFrame, start_date=None, end_date=None, time_col: str = UTC_TIME_COL
) -> pd.DataFrame:
    """Filter a canonical heart-rate table using an inclusive/exclusive interval."""
    validate_time_col(time_col)
    start = normalize_time_bound(start_date, time_col)
    end = normalize_time_bound(end_date, time_col)
    if start is not None:
        frame = frame[frame[time_col] >= start]
    if end is not None:
        frame = frame[frame[time_col] < end]
    return frame


def empty_frame(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical contract columns."""
    return pd.DataFrame(columns=[*REQUIRED_COLUMNS, *extra_columns])
