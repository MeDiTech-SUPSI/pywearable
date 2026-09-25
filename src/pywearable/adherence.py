"""Data-availability adherence for wearable physiological streams."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from . import cycle, hierarchy, schema, utils
from .loader.base import BaseLoader


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
    return [
        cycle.CYCLE_NIGHT if value == "nightly" else value for value in chain
    ]


def _windows_for_resolution(
    loader, users, start, end, resolution, cycle_kwargs
):
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


def _physical_bounds(windows, start, end):
    segments = [
        segment
        for row_segments in windows["segments"]
        for segment in row_segments
    ]
    if not segments:
        return start, end
    return min(value[0] for value in segments), max(value[1] for value in segments)


def _filter_to_intervals(frame, user, intervals, time_col):
    if frame is None or frame.empty or intervals is None:
        return frame
    if user not in intervals.index.get_level_values(0):
        return frame.iloc[0:0]
    valid = intervals.xs(user)
    valid = valid[valid["interval_valid"]]
    if valid.empty:
        return frame.iloc[0:0]
    mask = np.zeros(len(frame), dtype=bool)
    for interval in valid.itertuples():
        mask |= (
            frame[time_col].ge(interval.start_time)
            & frame[time_col].lt(interval.end_time)
        ).to_numpy()
    return frame[mask].copy()


def _thresholded(value, threshold):
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
    if frame is None:
        return np.nan
    segments = _merged_segments(segments)
    expected_seconds = sum((end - start).total_seconds() for start, end in segments)
    if expected_seconds <= 0:
        return np.nan
    if frame.empty:
        return 0.0
    observed_seconds = 0.0
    for start, end in segments:
        timestamps = frame.loc[
            frame[time_col].ge(start) & frame[time_col].lt(end), time_col
        ]
        for minute in timestamps.dt.floor("1min").drop_duplicates():
            overlap_start = max(start, minute)
            overlap_end = min(end, minute + pd.Timedelta(minutes=1))
            if overlap_end > overlap_start:
                observed_seconds += (overlap_end - overlap_start).total_seconds()
    return min(1.0, observed_seconds / expected_seconds)


def _bbi_adherence(frame, segments, time_col):
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
    observed_ms = 0.0
    for start, end in segments:
        observed_ms += frame.loc[
            frame[time_col].ge(start) & frame[time_col].lt(end), schema.BBI_COL
        ].sum()
    return min(1.0, observed_ms / expected_ms)


def _load_sleep_dates(
    loader,
    user,
    start,
    end,
    loader_kwargs,
    check_sleep_validation,
):
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
    """Return final data-availability adherence in the interval ``[start, end)``.

    Heart-rate adherence is the fraction of physical time represented by
    one-minute bins containing at least one valid sample. BBI adherence is
    represented positive BBI duration divided by expected physical duration.
    Sleep adherence is main-sleep-summary availability over expected local
    calendar dates. Unsupported or inapplicable combinations are missing;
    supported streams with no observations have zero adherence.

    A resolution sequence applies each non-final coverage threshold before
    calculating valid/expected child adherence at the next level. Only final
    adherence is returned. Sleep enters a hierarchy at its native daily grain;
    lower fixed resolutions are ignored for sleep. Garmin ``MANUAL`` and
    ``AUTO_MANUAL`` summaries are excluded by default when validation exists.
    """
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required for adherence")
    metrics = _metric_list(metric)
    users = utils.get_user_ids(loader, "all" if user_id is None else user_id)
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
    loading_days: int = 7,
    ceil_to_24: bool = False,
) -> pd.Series:
    """Return daily hours represented by one-minute heart-rate occupancy."""
    _ = loading_days
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
