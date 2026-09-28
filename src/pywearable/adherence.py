"""Measure data availability for wearable physiological streams.

The module exposes adherence for heart rate, beat-to-beat intervals (BBI),
and sleep summaries.  Each stream has a different native availability model:

* heart rate measures the fraction of physical time represented by occupied
  one-minute bins;
* BBI measures the positive interval duration represented in a window; and
* sleep measures the presence of a usable main-sleep summary for each local
  calendar date.

Scalar resolutions report that native coverage directly.  Resolution chains
instead turn coverage at each non-final level into a valid/invalid child and
report the fraction of expected children that remain valid at the final level.
Unsupported streams and inapplicable metric/resolution combinations are
represented by missing values, while supported streams with no observations
have zero adherence.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from . import cycle, hierarchy, schema
from .loader.base import BaseLoader
from .utils.legacy import get_user_ids


ADHERENCE_HEART_RATE = "heart_rate"
ADHERENCE_SLEEP = "sleep"
ADHERENCE_BBI = "bbi"
ADHERENCE_METRICS = (
    ADHERENCE_HEART_RATE,
    ADHERENCE_SLEEP,
    ADHERENCE_BBI,
)
_ADHERENCE_COLUMNS = {
    ADHERENCE_HEART_RATE: "heart_rate_adherence",
    ADHERENCE_SLEEP: "sleep_adherence",
    ADHERENCE_BBI: "bbi_adherence",
}


def _metric_list(metric) -> list[str]:
    """Normalize and validate a metric selection while preserving order."""
    if metric is None:
        return list(ADHERENCE_METRICS)
    metrics = [metric] if isinstance(metric, str) else list(metric)
    unknown = set(metrics).difference(ADHERENCE_METRICS)
    if unknown:
        raise ValueError(
            f"metric must be one among {list(ADHERENCE_METRICS)}, "
            f"not {sorted(unknown)}"
        )
    return list(dict.fromkeys(metrics))


def _metric_loader_kwargs(loader_kwargs, metrics) -> dict[str, dict]:
    """Return independent loader keyword dictionaries for selected metrics."""
    if loader_kwargs is None:
        return {metric: {} for metric in metrics}
    if not isinstance(loader_kwargs, dict):
        raise ValueError("loader_kwargs must be a dictionary or None")
    unknown = set(loader_kwargs).difference(ADHERENCE_METRICS)
    if unknown:
        raise ValueError(
            "loader_kwargs must be keyed by adherence metric; unknown keys: "
            + ", ".join(sorted(unknown))
        )
    result = {metric: {} for metric in metrics}
    for metric, kwargs in loader_kwargs.items():
        if not isinstance(kwargs, dict):
            raise ValueError(f"loader_kwargs[{metric!r}] must be a dictionary")
        if metric in result:
            result[metric] = dict(kwargs)
    return result


def _resolve_time_col(loader, metrics, chain, time_reference: str) -> str:
    """Resolve one timestamp column supported by all physical streams.

    Sleep calendar adherence always uses local dates.  Physical streams use
    UTC for fixed windows and local time for calendar or cycle windows when
    ``time_reference='auto'``.
    """
    physiological_chain = _physiological_chain(chain)
    calendar_based = (
        chain[-1] in hierarchy.CALENDAR_RESOLUTIONS or chain[-1] == "nightly"
    )
    if (
        ADHERENCE_SLEEP in metrics
        and calendar_based
        and time_reference == "utc"
    ):
        raise ValueError("sleep adherence uses local calendar dates")

    available = []
    if ADHERENCE_HEART_RATE in metrics:
        available.append(
            set(
                getattr(
                    loader,
                    "heart_rate_time_columns",
                    schema.HEART_RATE_TIME_COLUMNS,
                )
            )
        )
    if (
        ADHERENCE_BBI in metrics
        and getattr(loader, "hrv_source", "bbi") == "bbi"
    ):
        available.append(
            set(getattr(loader, "hrv_time_columns", schema.BBI_TIME_COLUMNS))
        )
    common = (
        set.intersection(*available)
        if available
        else set(schema.HEART_RATE_TIME_COLUMNS)
    )
    return hierarchy.resolve_time_col(
        common,
        time_reference,
        physiological_chain,
        "adherence",
    )


def _physiological_chain(chain):
    """Translate the public ``nightly`` alias into a physical night cycle."""
    return [
        cycle.CYCLE_NIGHT if value == "nightly" else value for value in chain
    ]


def _windows_for_resolution(
    loader, users, start, end, resolution, cycle_kwargs
):
    """Construct cycle, calendar, or fixed windows for one resolution."""
    if resolution in cycle.CYCLE_TYPES:
        return cycle.get_intervals(
            loader,
            cycle_type=resolution,
            user_id=users,
            start_date=start,
            end_date=end,
            **cycle_kwargs,
        )
    return hierarchy.windows_for_resolution(
        users, resolution, {user: (start, end) for user in users}
    )


def _fixed_cycle_windows(users, intervals, resolution):
    """Create fixed-grid windows that intersect valid cycle intervals.

    Edge windows retain the full fixed duration.  Samples are separately
    restricted to the cycle, so a partial edge window can have less than full
    adherence even when every in-cycle instant is represented.
    """
    duration = pd.Timedelta(resolution)
    records = []
    for user in users:
        if user not in intervals.index.get_level_values(0):
            continue
        valid = intervals.xs(user)
        valid = valid[valid["interval_valid"]]
        for interval in valid.itertuples():
            for label in pd.date_range(
                interval.start_time.floor(duration),
                interval.end_time,
                freq=duration,
                inclusive="left",
            ):
                records.append(
                    {
                        schema.USER_COL: user,
                        "window_start": label,
                        "start_time": label,
                        "end_time": label + duration,
                        "interval_valid": True,
                    }
                )
    if not records:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=[schema.USER_COL, "window_start"]
        )
        return pd.DataFrame(
            columns=["start_time", "end_time", "interval_valid"], index=index
        )
    return (
        pd.DataFrame(records)
        .drop_duplicates([schema.USER_COL, "window_start"])
        .set_index([schema.USER_COL, "window_start"])
        .sort_index()
    )


def _primary_windows(
    loader, users, start, end, chain, cycle_kwargs
):
    """Build primary windows and identify cycle intervals restricting data.

    When a later hierarchy level is a cycle, fixed primary windows are created
    only around valid instances of the first such cycle.  Each output window
    stores the physical segment used by the raw adherence calculator.
    """
    primary = chain[0]
    cycle_target = next(
        (value for value in chain[1:] if value in cycle.CYCLE_TYPES), None
    )
    restricting_intervals = None
    if cycle_target is not None:
        restricting_intervals = _windows_for_resolution(
            loader, users, start, end, cycle_target, cycle_kwargs
        )
    if cycle_target is not None and primary not in (
        *cycle.CYCLE_TYPES,
        *hierarchy.CALENDAR_RESOLUTIONS,
    ):
        windows = _fixed_cycle_windows(
            users, restricting_intervals, primary
        )
    else:
        windows = _windows_for_resolution(
            loader, users, start, end, primary, cycle_kwargs
        ).copy()
        if "interval_valid" not in windows:
            windows["interval_valid"] = True
    windows = windows.rename(
        columns={"start_time": "period_start", "end_time": "period_end"}
    )
    windows["segments"] = [
        ((row.period_start, row.period_end),)
        if row.interval_valid
        else tuple()
        for row in windows.itertuples()
    ]
    return windows, restricting_intervals


def _merged_segments(segments):
    """Return the union of valid half-open physical segments."""
    ordered = sorted(
        (
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in segments
            if pd.notna(start) and pd.notna(end) and end > start
        ),
        key=lambda value: value[0],
    )
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def _time_slice_bounds(timestamps, start, end):
    """Locate one half-open interval in sorted datetime64 timestamps."""
    start = pd.Timestamp(start).to_datetime64()
    end = pd.Timestamp(end).to_datetime64()
    left = timestamps.searchsorted(start, side="left")
    right = timestamps.searchsorted(end, side="left")
    return int(left), int(right)


def _physical_bounds(windows, start, end):
    """Return the smallest load interval covering all window segments."""
    segments = [
        segment
        for row_segments in windows["segments"]
        for segment in row_segments
    ]
    if not segments:
        return start, end
    return min(value[0] for value in segments), max(value[1] for value in segments)


def _filter_to_intervals(frame, user, intervals, time_col):
    """Restrict one user's samples to the union of valid cycle intervals."""
    if frame is None or frame.empty or intervals is None:
        return frame
    if user not in intervals.index.get_level_values(0):
        return frame.iloc[0:0]
    valid = intervals.xs(user)
    valid = valid[valid["interval_valid"]]
    if valid.empty:
        return frame.iloc[0:0]
    segments = _merged_segments(
        (interval.start_time, interval.end_time)
        for interval in valid.itertuples()
    )
    timestamps = frame[time_col].to_numpy(copy=False)
    pieces = []
    for start, end in segments:
        left, right = _time_slice_bounds(timestamps, start, end)
        if right > left:
            pieces.append(frame.iloc[left:right])
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces).copy()


def _thresholded(value, threshold):
    """Convert measured coverage into a valid child indicator.

    With no explicit threshold, any positive coverage is valid.  A supplied
    threshold qualifies values greater than or equal to it.  Failed, absent,
    or unsupported values become missing so they count as invalid children at
    the next hierarchy level.
    """
    if pd.isna(value):
        return np.nan
    if threshold is None:
        return 1.0 if value > 0 else np.nan
    if threshold is not None and value < threshold:
        return np.nan
    return 1.0


def _aggregate_adherence(
    trajectory,
    feature,
    loader,
    users,
    start,
    end,
    chain,
    coverage_thresholds,
    cycle_kwargs,
    cycle_intervals=None,
):
    """Aggregate qualified children and return final valid-child coverage.

    The primary trajectory contains both measured coverage and a binary
    feature indicating whether each child qualified.  Every non-final level
    may apply its threshold before feeding the next level.  The final level is
    intentionally not thresholded because its measured ``valid / expected``
    child fraction is the requested result.  With a one-level chain, raw
    coverage is returned directly.
    """
    if len(chain) == 1:
        return trajectory[f"{feature}__coverage"]

    current = trajectory.drop(columns=["segments"], errors="ignore")
    current_feature = feature
    for level_index, target in enumerate(chain[1:], start=1):
        parent, membership = hierarchy.parent_windows_from_trajectory(
            loader,
            users,
            current,
            target,
            chain[level_index - 1],
            start,
            end,
            cycle_kwargs,
            cycle_intervals if target in cycle.CYCLE_TYPES else None,
            prefer_explicit_bounds=True,
            cycle_label_bounds=False,
        )
        if parent is None:
            return pd.Series(dtype=float, name=f"{feature}__coverage")
        threshold = (
            coverage_thresholds[level_index]
            if level_index < len(chain) - 1
            else None
        )
        current, _ = hierarchy.aggregate_level(
            current,
            parent,
            target_resolution=target,
            aggregation="mean",
            membership=membership,
            min_coverage=threshold,
        )
        current_feature = f"{current_feature}__{target}_mean"
    return current[f"{current_feature}__coverage"]


def _load_heart_rate(loader, user, start, end, time_col, loader_kwargs):
    """Load, validate, and clean positive heart-rate samples for one user.

    ``None`` means the loader does not support heart rate; an empty canonical
    frame means the stream is supported but has no samples in the interval.
    Two days of padding are requested to support local-time and cycle bounds.
    """
    kwargs = dict(loader_kwargs)
    if "time_col" in kwargs:
        raise ValueError(
            "set time_reference instead of loader_kwargs['heart_rate']['time_col']"
        )
    try:
        frame = loader.load_heart_rate(
            user_id=user,
            start_date=start - pd.Timedelta(days=2),
            end_date=end + pd.Timedelta(days=2),
            time_col=time_col,
            **kwargs,
        )
    except NotImplementedError:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("heart-rate data must be a pandas DataFrame")
    if frame.empty:
        return pd.DataFrame(columns=[time_col, schema.HEART_RATE_COL])
    missing = {time_col, schema.HEART_RATE_COL}.difference(frame.columns)
    if missing:
        raise ValueError(
            "heart-rate data is missing required columns: "
            + ", ".join(sorted(missing))
        )
    frame = frame.copy()
    if time_col == schema.UTC_TIME_COL:
        frame[time_col] = pd.to_datetime(
            frame[time_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        frame[time_col] = pd.to_datetime(
            frame[time_col], errors="coerce", format="mixed"
        ).map(
            lambda value: value.tz_localize(None)
            if pd.notna(value) and value.tzinfo is not None
            else value
        )
    frame[schema.HEART_RATE_COL] = pd.to_numeric(
        frame[schema.HEART_RATE_COL], errors="coerce"
    )
    frame = frame[
        frame[time_col].notna()
        & np.isfinite(frame[schema.HEART_RATE_COL])
        & frame[schema.HEART_RATE_COL].gt(0)
    ]
    return frame.drop_duplicates(time_col).sort_values(time_col, kind="stable")


def _load_bbi(loader, user, start, end, time_col, loader_kwargs):
    """Load and normalize raw BBI data, or return ``None`` if unsupported.

    Provider-computed HRV summaries are not treated as evidence that raw BBI
    exists.  Duplicate timestamps retain the last normalized interval.
    """
    if getattr(loader, "hrv_source", "bbi") != "bbi":
        return None
    try:
        frame = loader.load_bbi(
            user,
            start - pd.Timedelta(days=2),
            end + pd.Timedelta(days=2),
            **loader_kwargs,
        )
    except NotImplementedError:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("BBI data must be a pandas DataFrame")
    supported = not frame.empty or {
        time_col,
        schema.BBI_COL,
    }.issubset(frame.columns)
    if not supported:
        return None
    frame = schema.prepare_bbi(frame)
    if time_col not in frame.columns:
        raise ValueError(f"BBI data does not provide {time_col}")
    return (
        frame[frame[time_col].notna()]
        .drop_duplicates(time_col, keep="last")
        .sort_values(time_col, kind="stable")
    )


def _heart_rate_adherence(frame, segments, time_col):
    """Return occupied one-minute time divided by segment duration.

    A minute contributes once when it contains at least one valid sample.
    Partial minutes at segment boundaries contribute only their overlap with
    the segment.  ``None`` denotes an unsupported stream; an empty supported
    frame produces zero.
    """
    if frame is None:
        return np.nan
    segments = _merged_segments(segments)
    expected_seconds = sum((end - start).total_seconds() for start, end in segments)
    if expected_seconds <= 0:
        return np.nan
    if frame.empty:
        return 0.0
    timestamps = frame[time_col].to_numpy(copy=False)
    observed_seconds = 0.0
    for start, end in segments:
        left, right = _time_slice_bounds(timestamps, start, end)
        if right <= left:
            continue
        minutes = timestamps[left:right].astype("datetime64[m]")
        unique_minutes = np.empty(len(minutes), dtype=bool)
        unique_minutes[0] = True
        unique_minutes[1:] = minutes[1:] != minutes[:-1]
        minute_starts = minutes[unique_minutes].astype("datetime64[ns]")
        overlap_starts = np.maximum(minute_starts, start.to_datetime64())
        overlap_ends = np.minimum(
            minute_starts + np.timedelta64(1, "m"), end.to_datetime64()
        )
        observed_seconds += (
            (overlap_ends - overlap_starts).sum() / np.timedelta64(1, "s")
        )
    return min(1.0, observed_seconds / expected_seconds)


def _bbi_adherence(frame, segments, time_col):
    """Return represented positive BBI duration divided by segment duration.

    Each interval is assigned by its timestamp.  Coverage is clipped to one,
    which protects against overlap, duplicate representation, and intervals
    extending beyond a window boundary.
    """
    if frame is None:
        return np.nan
    segments = _merged_segments(segments)
    expected_ms = sum(
        (end - start).total_seconds() * 1000 for start, end in segments
    )
    if expected_ms <= 0:
        return np.nan
    if frame.empty:
        return 0.0
    timestamps = frame[time_col].to_numpy(copy=False)
    observed_ms = 0.0
    for start, end in segments:
        left, right = _time_slice_bounds(timestamps, start, end)
        observed_ms += frame[schema.BBI_COL].iloc[left:right].sum()
    return min(1.0, observed_ms / expected_ms)


def _load_sleep_dates(
    loader,
    user,
    start,
    end,
    loader_kwargs,
    check_sleep_validation,
):
    """Return local dates with usable main-sleep summaries for one user.

    All rows are requested so adherence can decide whether a date is present.
    When validation checking is enabled, ``MANUAL`` and ``AUTO_MANUAL`` rows
    are excluded.  ``None`` denotes an unsupported sleep-summary stream,
    whereas an empty set denotes a supported stream with no usable nights.
    """
    if "drop_duplicates" in loader_kwargs:
        raise ValueError(
            "sleep adherence always requests one main-sleep summary per date"
        )
    try:
        frame = loader.load_sleep_summary(
            user_id=user,
            start_date=start,
            end_date=end,
            drop_duplicates=False,
            **loader_kwargs,
        )
    except NotImplementedError:
        return None
    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sleep summary data must be a pandas DataFrame")
    frame = schema.normalize_sleep_columns(frame.copy())
    if frame.empty:
        return set()
    if schema.CALENDAR_DATE_COL not in frame:
        raise ValueError("sleep summary data is missing calendar_date")
    if check_sleep_validation and "validation" in frame:
        validation = frame["validation"].astype("string").str.upper()
        frame = frame[~validation.isin({"MANUAL", "AUTO_MANUAL"})]
    dates = pd.to_datetime(
        frame[schema.CALENDAR_DATE_COL], errors="coerce"
    ).dt.normalize()
    return set(dates.dropna())


def _stream_adherence(
    loader,
    metric,
    users,
    start,
    end,
    time_col,
    chain,
    coverage_thresholds,
    loader_kwargs,
    cycle_kwargs,
):
    """Calculate heart-rate or BBI adherence through a resolution chain.

    The function constructs primary windows, loads one stream per user,
    calculates raw coverage for every valid window, qualifies primary children
    when the chain has upper levels, and delegates all later aggregation to
    :func:`_aggregate_adherence`.
    """
    windows, restricting_intervals = _primary_windows(
        loader, users, start, end, chain, cycle_kwargs
    )
    feature = _ADHERENCE_COLUMNS[metric]
    trajectory = windows[
        ["period_start", "period_end", "segments", "interval_valid"]
    ].copy()
    trajectory[feature] = np.nan
    trajectory[f"{feature}__coverage"] = np.nan
    unsupported_users = set()
    primary_threshold = (
        coverage_thresholds[0] if len(chain) > 1 else None
    )

    for user in users:
        if user in windows.index.get_level_values(0):
            user_windows = windows.xs(user)
        else:
            user_windows = windows.iloc[0:0]
        load_start, load_end = _physical_bounds(
            user_windows, start, end
        )
        if metric == ADHERENCE_HEART_RATE:
            frame = _load_heart_rate(
                loader,
                user,
                load_start,
                load_end,
                time_col,
                loader_kwargs,
            )
            calculator = _heart_rate_adherence
        else:
            frame = _load_bbi(
                loader,
                user,
                load_start,
                load_end,
                time_col,
                loader_kwargs,
            )
            calculator = _bbi_adherence
        if frame is None:
            unsupported_users.add(user)
        frame = _filter_to_intervals(
            frame, user, restricting_intervals, time_col
        )
        for label, window in user_windows.iterrows():
            raw = (
                calculator(frame, window["segments"], time_col)
                if window["interval_valid"]
                else np.nan
            )
            index = (user, label)
            trajectory.at[index, f"{feature}__coverage"] = raw
            trajectory.at[index, feature] = _thresholded(
                raw, primary_threshold
            )

    result = _aggregate_adherence(
        trajectory,
        feature,
        loader,
        users,
        start,
        end,
        chain,
        coverage_thresholds,
        cycle_kwargs,
        restricting_intervals,
    ).rename(feature)
    for user in unsupported_users:
        result.loc[result.index.get_level_values(0) == user] = np.nan
    return result


def _sleep_daily_trajectory(
    loader,
    users,
    start,
    end,
    threshold,
    loader_kwargs,
    check_sleep_validation,
):
    """Build the native daily sleep-availability trajectory.

    Each expected local date has raw coverage one when at least one usable
    main-sleep summary exists and zero otherwise.  Only observed dates become
    valid children for an upper level.
    """
    feature = _ADHERENCE_COLUMNS[ADHERENCE_SLEEP]
    dates = pd.date_range(
        start.normalize(), end, freq="D", inclusive="left"
    )
    records = []
    unsupported_users = set()
    for user in users:
        observed = _load_sleep_dates(
            loader,
            user,
            start,
            end,
            loader_kwargs,
            check_sleep_validation,
        )
        if observed is None:
            unsupported_users.add(user)
        for date in dates:
            raw = (
                np.nan
                if observed is None
                else float(date in observed)
            )
            records.append(
                {
                    schema.USER_COL: user,
                    "window_start": date,
                    "period_start": date,
                    "period_end": date + pd.Timedelta(days=1),
                    feature: (
                        _thresholded(raw, threshold)
                        if raw == 1
                        else np.nan
                    ),
                    f"{feature}__coverage": raw,
                }
            )
    if not records:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=[schema.USER_COL, "window_start"]
        )
        frame = pd.DataFrame(
            columns=[
                "period_start",
                "period_end",
                feature,
                f"{feature}__coverage",
            ],
            index=index,
        )
    else:
        frame = (
            pd.DataFrame(records)
            .set_index([schema.USER_COL, "window_start"])
            .sort_index()
        )
    return frame, unsupported_users


def _sleep_levels(chain, coverage_thresholds):
    """Reduce a requested chain to sleep's daily-and-above native levels.

    Fixed lower levels and cycle labels do not create artificial subdaily sleep
    observations.  Calendar levels above daily are retained with their
    corresponding thresholds.
    """
    native = {
        "daily",
        "nightly",
        cycle.CYCLE_NIGHT,
        cycle.CYCLE_PHYSIOLOGICAL,
    }
    native_index = next(
        (index for index, value in enumerate(chain) if value in native),
        None,
    )
    levels = ["daily"]
    thresholds = [
        coverage_thresholds[native_index]
        if native_index is not None
        else None
    ]
    for index, value in enumerate(chain):
        if value in hierarchy.CALENDAR_RESOLUTIONS and value != "daily":
            levels.append(value)
            thresholds.append(coverage_thresholds[index])
    return levels, thresholds


def _constant_final_series(
    loader,
    users,
    start,
    end,
    chain,
    cycle_kwargs,
    value,
):
    """Return one constant value on the final requested window index."""
    target = chain[-1]
    if target == "nightly":
        target = cycle.CYCLE_NIGHT
    windows = _windows_for_resolution(
        loader, users, start, end, target, cycle_kwargs
    )
    return pd.Series(value, index=windows.index, dtype=float)


def _sleep_adherence_series(
    loader,
    users,
    start,
    end,
    chain,
    coverage_thresholds,
    loader_kwargs,
    cycle_kwargs,
    check_sleep_validation,
):
    """Calculate sleep-summary adherence at applicable final resolutions.

    Awake-cycle adherence is structurally zero.  Fixed and other unsupported
    final resolutions are inapplicable and therefore missing.  Night, daily,
    and larger calendar outputs start from native daily summary presence.
    """
    feature = _ADHERENCE_COLUMNS[ADHERENCE_SLEEP]
    if cycle.CYCLE_AWAKE in chain:
        return _constant_final_series(
            loader,
            users,
            start,
            end,
            chain,
            cycle_kwargs,
            0.0,
        ).rename(feature)

    final = chain[-1]
    sleep_final = (
        final in hierarchy.CALENDAR_RESOLUTIONS
        or final == "nightly"
        or final in {cycle.CYCLE_NIGHT, cycle.CYCLE_PHYSIOLOGICAL}
    )
    if not sleep_final:
        return _constant_final_series(
            loader,
            users,
            start,
            end,
            chain,
            cycle_kwargs,
            np.nan,
        ).rename(feature)

    sleep_chain, sleep_thresholds = _sleep_levels(
        chain, coverage_thresholds
    )
    trajectory, unsupported_users = _sleep_daily_trajectory(
        loader,
        users,
        start,
        end,
        sleep_thresholds[0] if len(sleep_chain) > 1 else None,
        loader_kwargs,
        check_sleep_validation,
    )
    result = _aggregate_adherence(
        trajectory,
        feature,
        loader,
        users,
        start,
        end,
        sleep_chain,
        sleep_thresholds,
        cycle_kwargs,
    ).rename(feature)
    for user in unsupported_users:
        result.loc[result.index.get_level_values(0) == user] = np.nan
    return result


def get_adherence(
    loader: BaseLoader,
    metric: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    resolution: Union[str, pd.Timedelta, list, tuple] = "daily",
    time_reference: str = "auto",
    min_coverage=None,
    loader_kwargs: dict | None = None,
    cycle_kwargs: dict | None = None,
    check_sleep_validation: bool = True,
) -> pd.DataFrame:
    """Return final data-availability adherence over ``[start_date, end_date)``.

    Heart-rate adherence is the fraction of physical time represented by
    one-minute bins containing at least one positive sample.  BBI adherence is
    positive BBI duration divided by expected physical duration.  Sleep
    adherence is usable main-sleep-summary presence over expected local dates.

    A scalar resolution reports raw coverage directly.  A resolution sequence
    qualifies children at every non-final level and reports final
    ``valid children / expected children`` coverage.  Children have equal
    weight, irrespective of their physical duration.  Sleep enters a hierarchy
    at its native daily grain, so lower fixed resolutions are ignored rather
    than populated with artificial observations.

    Parameters
    ----------
    loader
        Loader providing the requested streams and user identifiers.
    metric
        One metric name, a sequence of names, or ``None`` for all of
        ``'heart_rate'``, ``'sleep'``, and ``'bbi'``.  Output columns follow the
        requested order and are named ``<metric>_adherence``.
    user_id
        One user, a sequence of users, ``'all'``, or ``None``.  ``None`` is an
        alias for ``'all'``.
    start_date, end_date
        Required half-open bounds, normalized to the clock selected by
        ``time_reference``.  Sleep calendar bounds are always local.
    resolution
        A fixed resolution, calendar resolution, cycle type, or an ordered
        resolution chain accepted by
        :func:`pywearable.hierarchy.normalize_resolution_chain`.  ``'nightly'``
        is an alias for ``cycle_night`` for physical streams and for
        date-labelled nightly sleep-summary availability for sleep.
    time_reference
        ``'auto'``, ``'local'``, or ``'utc'``.  ``'auto'`` selects UTC for
        fixed physical windows and local time for calendar or cycle windows.
        All requested physical metrics must support the selected clock.  UTC
        cannot represent sleep calendar adherence.
    min_coverage
        Optional number or sequence with one entry per requested resolution.
        On non-final levels, a child qualifies when measured coverage is at
        least this threshold.  ``None`` qualifies any child with positive
        coverage.  The final entry is not applied because final coverage is
        the returned value.  Consequently, this argument has no filtering
        effect for a scalar resolution.
    loader_kwargs
        Optional mapping keyed by metric name.  Each value is forwarded to
        that metric's loader method.  Heart-rate ``time_col`` is controlled by
        ``time_reference`` and sleep ``drop_duplicates`` is controlled here,
        so callers may not override them.
    cycle_kwargs
        Optional keyword arguments forwarded to :func:`cycle.get_intervals`.
    check_sleep_validation
        If true, sleep-summary rows labelled ``MANUAL`` or ``AUTO_MANUAL`` in
        the canonical ``validation`` column do not count as observed nights.

    Returns
    -------
    pandas.DataFrame
        Final adherence values indexed by ``(user_id, time)``.  The time index
        is named ``utc_time`` or ``local_time`` for fixed windows and
        ``calendar_date`` for calendar or cycle outputs.  Values lie in
        ``[0, 1]``; missing values indicate an unsupported stream or an
        inapplicable metric/resolution combination.

    Raises
    ------
    ValueError
        If bounds, metric names, resolutions, thresholds, clocks, or forwarded
        keyword structures are invalid.

    Notes
    -----
    ``resolution='daily'`` measures raw coverage across each day.  In contrast,
    ``resolution=['5min', 'daily']`` measures the fraction of five-minute
    children that satisfy the primary threshold.  These quantities coincide
    only under particular sampling patterns.

    Examples
    --------
    Direct five-minute coverage for heart rate and BBI::

        get_adherence(
            loader,
            metric=['heart_rate', 'bbi'],
            user_id='participant-01',
            start_date='2024-01-01',
            end_date='2024-01-02',
            resolution='5min',
        )

    Fraction of qualifying five-minute heart-rate windows in each night::

        get_adherence(
            loader,
            metric='heart_rate',
            start_date='2024-01-01',
            end_date='2024-02-01',
            resolution=['5min', cycle.CYCLE_NIGHT],
            min_coverage=[0.8, None],
        )
    """
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required for adherence")
    metrics = _metric_list(metric)
    users = get_user_ids(loader, user_id)
    chain, _ = hierarchy.normalize_resolution_chain(resolution)
    physiological_chain = _physiological_chain(chain)
    time_col = _resolve_time_col(loader, metrics, chain, time_reference)
    start = schema.normalize_time_bound(start_date, time_col)
    end = schema.normalize_time_bound(end_date, time_col)
    if end <= start:
        raise ValueError("end_date must be after start_date")
    metric_kwargs = _metric_loader_kwargs(loader_kwargs, metrics)
    coverage_thresholds = hierarchy.normalize_coverage_thresholds(
        min_coverage, len(chain)
    )
    if cycle_kwargs is not None and not isinstance(cycle_kwargs, dict):
        raise ValueError("cycle_kwargs must be a dictionary or None")
    cycle_kwargs = {} if cycle_kwargs is None else dict(cycle_kwargs)
    if not isinstance(check_sleep_validation, bool):
        raise ValueError("check_sleep_validation must be a boolean")
    columns = [_ADHERENCE_COLUMNS[value] for value in metrics]
    if not users:
        index_name = (
            schema.CALENDAR_DATE_COL
            if any(value in cycle.CYCLE_TYPES for value in chain)
            or chain[-1] in hierarchy.CALENDAR_RESOLUTIONS
            or chain[-1] == "nightly"
            else time_col
        )
        index = pd.MultiIndex.from_arrays(
            [[], []], names=[schema.USER_COL, index_name]
        )
        return pd.DataFrame(columns=columns, index=index, dtype=float)
    series = []
    for value in metrics:
        if value == ADHERENCE_SLEEP:
            metric_result = _sleep_adherence_series(
                loader,
                users,
                start,
                end,
                chain,
                coverage_thresholds,
                metric_kwargs[value],
                cycle_kwargs,
                check_sleep_validation,
            )
        else:
            metric_result = _stream_adherence(
                loader,
                value,
                users,
                start,
                end,
                time_col,
                physiological_chain,
                coverage_thresholds,
                metric_kwargs[value],
                cycle_kwargs,
            )
        series.append(metric_result)
    result = pd.concat(series, axis=1).reindex(columns=columns)
    has_cycle = any(
        value in cycle.CYCLE_TYPES or value == "nightly"
        for value in chain
    )
    index_name = (
        schema.CALENDAR_DATE_COL
        if has_cycle or chain[-1] in hierarchy.CALENDAR_RESOLUTIONS
        else time_col
    )
    return result.sort_index().rename_axis(index={"window_start": index_name})


def get_daily_wearing_time(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ceil_to_24: bool = False,
) -> pd.Series:
    """Return daily hours represented by one-minute heart-rate occupancy.

    Each daily adherence fraction is multiplied by the exact requested
    local-day duration, so partial first and last days remain partial.

    Parameters
    ----------
    loader
        Loader providing timestamped heart-rate samples.
    user_id
        One user identifier.
    start_date, end_date
        Required half-open local-time bounds.
    ceil_to_24
        If true, clip reported daily hours at 24.  Adherence is already capped
        at one, so this only preserves the historical option.

    Returns
    -------
    pandas.Series
        Local-date-indexed hours named ``'daily-hr'``.
    """
    result = get_adherence(
        loader,
        metric=ADHERENCE_HEART_RATE,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="daily",
        time_reference="local",
    )
    adherence = result.xs(user_id, level=schema.USER_COL)[
        _ADHERENCE_COLUMNS[ADHERENCE_HEART_RATE]
    ]
    start = schema.normalize_time_bound(start_date, schema.LOCAL_TIME_COL)
    end = schema.normalize_time_bound(end_date, schema.LOCAL_TIME_COL)
    windows = hierarchy.calendar_windows(
        [user_id], "daily", {user_id: (start, end)}
    ).xs(user_id)
    durations = (
        windows["end_time"] - windows["start_time"]
    ).dt.total_seconds() / 3600
    hours = adherence * durations.reindex(adherence.index)
    if ceil_to_24:
        hours = hours.clip(upper=24)
    return hours.rename("daily-hr")


__all__ = [
    "ADHERENCE_BBI",
    "ADHERENCE_HEART_RATE",
    "ADHERENCE_METRICS",
    "ADHERENCE_SLEEP",
    "get_adherence",
    "get_daily_wearing_time",
]
