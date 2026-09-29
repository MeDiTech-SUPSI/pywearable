import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd


def clock_seconds(value) -> float:
    """Convert a timestamp or clock time to seconds since midnight.

    Calendar dates are intentionally discarded so clock-valued features can
    be compared or aggregated by time of day.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, datetime.time):
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
            + value.microsecond / 1_000_000
        )
    timestamp = pd.Timestamp(value)
    return (
        timestamp.hour * 3600
        + timestamp.minute * 60
        + timestamp.second
        + timestamp.microsecond / 1_000_000
    )


def clock_time(seconds):
    """Convert seconds since midnight to a wrapped :class:`datetime.time`."""
    if pd.isna(seconds):
        return None
    microseconds = int(round(float(seconds) * 1_000_000)) % (
        24 * 60 * 60 * 1_000_000
    )
    hour, remainder = divmod(microseconds, 3_600 * 1_000_000)
    minute, remainder = divmod(remainder, 60 * 1_000_000)
    second, microsecond = divmod(remainder, 1_000_000)
    return datetime.time(hour, minute, second, microsecond)


@dataclass
class ResamplingSpec:
    rule: str
    time_col: str
    calendar_based: bool


def infer_cadence(
    timestamps,
    *,
    min_gaps: int,
    min_cluster_fraction: float,
    round_long_gaps: bool = False,
    median_ratio_bounds: tuple[float, float] | None = None,
) -> pd.Timedelta | None:
    """Infer nominal spacing from a prevalent cluster of positive time gaps."""
    values = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna()
    gaps = values.drop_duplicates().sort_values().diff().dt.total_seconds().dropna()
    gaps = gaps[gaps > 0]
    if len(gaps) < min_gaps:
        return None
    seconds = gaps.to_numpy(dtype=float)
    rounded = (
        np.where(seconds < 10, np.round(seconds, 2), np.round(seconds))
        if round_long_gaps
        else np.round(seconds, 2)
    )
    mode = float(pd.Series(rounded).value_counts().index[0])
    cluster = seconds[np.abs(seconds - mode) <= max(0.05, mode * 0.1)]
    if len(cluster) / len(seconds) < min_cluster_fraction:
        return None
    cadence = float(np.median(cluster))
    if median_ratio_bounds is not None:
        ratio = cadence / float(np.median(seconds))
        if not median_ratio_bounds[0] <= ratio <= median_ratio_bounds[1]:
            return None
    return pd.Timedelta(seconds=cadence)


def sample_coverage(observed_count, start, end, cadence) -> float:
    """Estimate observed/expected samples for one regularly sampled stream."""
    if cadence is None or end <= start:
        return np.nan
    expected = (end - start) / cadence
    return min(1.0, observed_count / expected) if expected > 0 else np.nan


def get_resampling_spec(
    resolution: str,
    time_reference: str = "auto",
) -> ResamplingSpec:

    if time_reference not in {"auto", "utc", "local"}:
        raise ValueError(
            "time_reference must be one of ('auto', 'utc', 'local'), "
            f"not {time_reference}"
        )

    if resolution == "day" or resolution == "daily":
        rule = "d"
        calendar_based = True
    elif resolution == "week" or resolution == "weekly":
        rule = "W-MON"
        calendar_based = True
    elif resolution == "month" or resolution == "monthly":
        rule = "MS"
        calendar_based = True
    elif resolution == "year" or resolution == "yearly":
        rule = "YS"
        calendar_based = True
    else:
        try:
            pd.Timedelta(resolution)
        except ValueError:
            raise ValueError(f"Resolution {resolution} is not a valid choice.")
        rule = resolution
        calendar_based = False

    if time_reference == "auto":
        time_reference = "local" if calendar_based else "utc"
    return ResamplingSpec(
        rule=rule,
        time_col=f"{time_reference}_time",
        calendar_based=calendar_based,
    )


def get_calendar_period_start(value, rule: str) -> pd.Timestamp:
    """Return the canonical start containing ``value`` for a calendar rule."""
    timestamp = pd.Timestamp(value)
    if rule == "d":
        return timestamp.normalize()
    if rule == "W-MON":
        return timestamp.normalize() - pd.Timedelta(days=timestamp.weekday())
    if rule == "MS":
        return timestamp.to_period("M").start_time
    if rule == "YS":
        return timestamp.to_period("Y").start_time
    raise ValueError(f"Rule {rule} is not a supported calendar resolution.")


def get_calendar_window_starts(
    start, end, rule: str, include_end: bool = False
) -> pd.DatetimeIndex:
    """Return starts for calendar windows intersected with ``[start, end)``."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end < start or (end == start and not include_end):
        return pd.DatetimeIndex([])

    canonical_start = get_calendar_period_start(start, rule)
    canonical_boundaries = pd.date_range(
        start=canonical_start,
        end=end,
        freq=rule,
        inclusive="both",
    )
    starts = [start]
    starts.extend(
        boundary
        for boundary in canonical_boundaries
        if boundary > start
        and (boundary < end or (include_end and boundary == end))
    )
    return pd.DatetimeIndex(starts)
