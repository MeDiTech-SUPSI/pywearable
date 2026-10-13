"""Step-count volume and peak-hour timing across temporal resolutions."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from . import cycle, hierarchy, schema
from .loader.base import BaseLoader

ACTIVITY_STATISTIC_STEPS = schema.ACTIVITY_STEPS_COL
ACTIVITY_STATISTIC_PEAK_60MIN_STEPS = "activity_peak_60min_steps"
ACTIVITY_STATISTIC_PEAK_60MIN_START = "activity_peak_60min_start"

ACTIVITY_STATISTICS = (
    ACTIVITY_STATISTIC_STEPS,
    ACTIVITY_STATISTIC_PEAK_60MIN_STEPS,
    ACTIVITY_STATISTIC_PEAK_60MIN_START,
)
_PEAK_STATISTICS = {
    ACTIVITY_STATISTIC_PEAK_60MIN_STEPS,
    ACTIVITY_STATISTIC_PEAK_60MIN_START,
}


def _statistics_list(statistic, source: str) -> list[str]:
    if statistic is None:
        return (
            [ACTIVITY_STATISTIC_STEPS]
            if source == "daily_summary"
            else list(ACTIVITY_STATISTICS)
        )
    statistics = [statistic] if isinstance(statistic, str) else list(statistic)
    unknown = set(statistics).difference(ACTIVITY_STATISTICS)
    if unknown:
        raise ValueError(
            f"statistic must be one among {list(ACTIVITY_STATISTICS)}, "
            f"not {sorted(unknown)}"
        )
    if source == "daily_summary" and set(statistics).intersection(_PEAK_STATISTICS):
        raise ValueError(
            "peak 60-minute activity requires timestamped step intervals; "
            "daily summaries provide only activity_steps"
        )
    return list(dict.fromkeys(statistics))


def _prepare_chain(resolution, source: str) -> tuple[list[str], bool]:
    chain, _ = hierarchy.normalize_resolution_chain(resolution)
    chain = [cycle.CYCLE_NIGHT if value == "nightly" else value for value in chain]
    hierarchy.normalize_resolution_chain(chain)
    if chain[0] in {"weekly", "monthly", "yearly"}:
        chain = ["daily", *chain]
    if source == "daily_summary" and chain[0] != "daily":
        raise ValueError(
            "provider daily activity summaries can only remain daily or be "
            "aggregated to a larger calendar resolution"
        )
    return chain, len(chain) > 1


def _validate_primary_resolution(loader, resolution: str) -> None:
    requested = hierarchy.fixed_duration(resolution)
    if requested is None:
        return
    native_name = getattr(loader, "activity_primary_resolution", None)
    if native_name is None:
        raise ValueError("the loader does not declare an activity sample resolution")
    native = pd.Timedelta(native_name)
    if requested < native or requested.value % native.value:
        raise ValueError(
            f"resolution must be an integer multiple of the native {native_name} "
            "activity interval"
        )


def _interval_union_seconds(starts, ends) -> float:
    intervals = sorted(
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in zip(starts, ends)
        if pd.notna(start) and pd.notna(end) and end > start
    )
    if not intervals:
        return 0.0
    total = pd.Timedelta(0)
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total.total_seconds()


def _heart_rate_observation_minutes(
    loader,
    user,
    start_date,
    end_date,
    time_col,
):
    """Return one-minute intervals containing at least one valid HR sample."""
    heart_rate = loader.load_heart_rate(
        user,
        start_date=start_date,
        end_date=end_date,
        time_col=time_col,
    )
    columns = ["__activity_observation_start", "__activity_observation_end"]
    if heart_rate.empty:
        return pd.DataFrame(columns=columns)
    missing = {time_col, schema.HEART_RATE_COL}.difference(heart_rate.columns)
    if missing:
        raise ValueError(
            "heart-rate coverage data is missing canonical columns: "
            f"{sorted(missing)}"
        )
    times = pd.to_datetime(heart_rate[time_col], errors="coerce")
    values = pd.to_numeric(heart_rate[schema.HEART_RATE_COL], errors="coerce")
    minutes = (
        times[times.notna() & values.gt(0)]
        .dt.floor("min")
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    return pd.DataFrame(
        {
            columns[0]: minutes,
            columns[1]: minutes + pd.Timedelta(minutes=1),
        }
    )


def _observation_intervals(rows, heart_rate_rows, time_col, end_col):
    """Combine HR presence and positive-step intervals without double counting."""
    positive_steps = rows[rows[schema.ACTIVITY_STEPS_COL].gt(0)]
    frames = []
    if not positive_steps.empty:
        frames.append(
            positive_steps[[time_col, end_col]].rename(
                columns={
                    time_col: "__activity_observation_start",
                    end_col: "__activity_observation_end",
                }
            )
        )
    if not heart_rate_rows.empty:
        frames.append(
            heart_rate_rows[
                ["__activity_observation_start", "__activity_observation_end"]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "__activity_observation_start",
                "__activity_observation_end",
            ]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values("__activity_observation_start", kind="stable")
    )


def _peak_sixty_minutes(rows, observations, time_col, end_col, window_end):
    """Return the largest fully observed contiguous 60-minute step total."""
    unknown = rows[rows[schema.ACTIVITY_STEPS_COL].isna()]
    rows = rows[rows[schema.ACTIVITY_STEPS_COL].notna()].sort_values(time_col)
    if observations.empty:
        return np.nan, pd.NaT
    hour = pd.Timedelta(hours=1)
    best_steps = np.nan
    best_start = pd.NaT
    candidates = (
        pd.DatetimeIndex(rows[time_col].drop_duplicates())
        if not rows.empty
        else pd.DatetimeIndex(
            observations["__activity_observation_start"].head(1)
        )
    )
    for candidate in candidates:
        candidate_end = candidate + hour
        if candidate_end > window_end:
            break
        if (unknown[time_col].lt(candidate_end) & unknown[end_col].gt(candidate)).any():
            continue
        selected = rows[rows[time_col].ge(candidate) & rows[end_col].le(candidate_end)]
        observed_rows = observations[
            observations["__activity_observation_start"].ge(candidate)
            & observations["__activity_observation_end"].le(candidate_end)
        ]
        observed = _interval_union_seconds(
            observed_rows["__activity_observation_start"],
            observed_rows["__activity_observation_end"],
        )
        if observed < hour.total_seconds():
            continue
        steps = selected[schema.ACTIVITY_STEPS_COL].sum(min_count=1)
        if pd.isna(steps):
            steps = 0.0
        if pd.isna(best_steps) or steps > best_steps:
            best_steps = float(steps)
            best_start = candidate
    return best_steps, best_start


def _cycle_bounds(intervals, user):
    if intervals is None or intervals.empty:
        return None
    available = set(intervals.index.get_level_values(0))
    if user not in available:
        return None
    rows = intervals.xs(user)
    valid = rows[rows["interval_valid"]]
    if valid.empty:
        return None
    return valid["start_time"].min(), valid["end_time"].max()


def _primary_windows(
    user,
    frame,
    time_col,
    resolution,
    start_date,
    end_date,
    cycle_intervals,
):
    if resolution in cycle.CYCLE_TYPES:
        if cycle_intervals is None or cycle_intervals.empty:
            return pd.DataFrame()
        available = set(cycle_intervals.index.get_level_values(0))
        if user not in available:
            return pd.DataFrame()
        windows = cycle_intervals.xs(user).copy()
        return windows.rename_axis("window_start")
    bounds = hierarchy.observed_bounds(
        frame, time_col, start_date, end_date, resolution
    )
    if bounds is None:
        return pd.DataFrame()
    return hierarchy.windows_for_resolution([user], resolution, {user: bounds}).xs(user)


def _sample_primary_trajectory(
    loader,
    users,
    statistics,
    resolution,
    chain,
    start_date,
    end_date,
    time_reference,
    loader_kwargs,
    min_coverage,
    return_coverage,
    cycle_intervals,
):
    """Build primary activity windows from canonical incremental intervals."""
    _validate_primary_resolution(loader, resolution)
    time_col = hierarchy.resolve_time_col(
        getattr(loader, "activity_time_columns", schema.ACTIVITY_TIME_COLUMNS),
        time_reference,
        chain,
        "activity",
    )
    frames = []
    padding = pd.Timedelta(days=2)
    for user in users:
        cycle_bounds = _cycle_bounds(cycle_intervals, user)
        user_start, user_end = start_date, end_date
        if cycle_intervals is not None:
            if cycle_bounds is None:
                continue
            user_start, user_end = cycle_bounds
        load_start = None if user_start is None else pd.Timestamp(user_start) - padding
        load_end = None if user_end is None else pd.Timestamp(user_end) + padding
        samples = schema.prepare_activity_samples(
            loader.load_activity_samples(
                user,
                start_date=load_start,
                end_date=load_end,
                **loader_kwargs,
            )
        )
        samples = samples[samples[time_col].notna()].sort_values(
            time_col, kind="stable"
        )
        needs_observations = return_coverage or bool(
            set(statistics).intersection(_PEAK_STATISTICS)
        )
        heart_rate_minutes = (
            _heart_rate_observation_minutes(
                loader,
                user,
                load_start,
                load_end,
                time_col,
            )
            if needs_observations
            else pd.DataFrame(
                columns=[
                    "__activity_observation_start",
                    "__activity_observation_end",
                ]
            )
        )
        windows = _primary_windows(
            user,
            samples,
            time_col,
            resolution,
            user_start,
            user_end,
            cycle_intervals,
        )
        if windows.empty:
            continue
        end_col = "__activity_interval_end"
        samples[end_col] = samples[time_col] + pd.to_timedelta(
            samples[schema.DURATION_IN_MS_COL], unit="ms"
        )
        sample_times = pd.DatetimeIndex(samples[time_col])
        heart_rate_times = pd.DatetimeIndex(
            heart_rate_minutes["__activity_observation_start"]
        )
        records = []
        for label, window in windows.iterrows():
            window_start = window["start_time"]
            window_end = window["end_time"]
            interval_valid = bool(window.get("interval_valid", True))
            if (
                not interval_valid
                or pd.isna(window_start)
                or pd.isna(window_end)
                or window_end <= window_start
            ):
                rows = samples.iloc[0:0]
                observations = pd.DataFrame(
                    columns=[
                        "__activity_observation_start",
                        "__activity_observation_end",
                    ]
                )
                coverage = np.nan
            else:
                left = sample_times.searchsorted(window_start, side="left")
                right = sample_times.searchsorted(window_end, side="left")
                rows = samples.iloc[left:right]
                rows = rows[rows[end_col].le(window_end)]
                heart_rate_left = heart_rate_times.searchsorted(
                    window_start, side="left"
                )
                heart_rate_right = heart_rate_times.searchsorted(
                    window_end, side="left"
                )
                heart_rate_rows = heart_rate_minutes.iloc[
                    heart_rate_left:heart_rate_right
                ]
                heart_rate_rows = heart_rate_rows[
                    heart_rate_rows["__activity_observation_end"].le(window_end)
                ]
                observations = _observation_intervals(
                    rows, heart_rate_rows, time_col, end_col
                )
                observed_seconds = _interval_union_seconds(
                    observations["__activity_observation_start"],
                    observations["__activity_observation_end"],
                )
                expected_seconds = (window_end - window_start).total_seconds()
                coverage = (
                    min(1.0, observed_seconds / expected_seconds)
                    if expected_seconds > 0
                    else np.nan
                )
            accepted = min_coverage is None or (
                pd.notna(coverage) and coverage >= min_coverage
            )
            values = {name: np.nan for name in statistics}
            if accepted and (not rows.empty or coverage > 0):
                if ACTIVITY_STATISTIC_STEPS in statistics:
                    steps = rows[
                        schema.ACTIVITY_STEPS_COL
                    ].sum(min_count=1, skipna=False)
                    values[ACTIVITY_STATISTIC_STEPS] = (
                        0.0 if rows.empty and coverage > 0 else steps
                    )
                if set(statistics).intersection(_PEAK_STATISTICS):
                    peak_steps, peak_start = _peak_sixty_minutes(
                        rows, observations, time_col, end_col, window_end
                    )
                    if ACTIVITY_STATISTIC_PEAK_60MIN_STEPS in statistics:
                        values[ACTIVITY_STATISTIC_PEAK_60MIN_STEPS] = peak_steps
                    if ACTIVITY_STATISTIC_PEAK_60MIN_START in statistics:
                        values[ACTIVITY_STATISTIC_PEAK_60MIN_START] = peak_start
            records.append(
                {
                    schema.USER_COL: user,
                    "window_start": label,
                    "period_start": window_start,
                    "period_end": window_end,
                    **(
                        hierarchy.fixed_window_time_metadata(rows, label, time_col)
                        if hierarchy.fixed_duration(resolution) is not None
                        else {}
                    ),
                    **values,
                    **(
                        {f"{name}__coverage": coverage for name in statistics}
                        if return_coverage
                        else {}
                    ),
                }
            )
        frames.append(
            pd.DataFrame(records).set_index([schema.USER_COL, "window_start"])
        )
    return (
        pd.concat(frames).sort_index()
        if frames
        else hierarchy.empty_trajectory(statistics, return_coverage)
    )


def _summary_primary_trajectory(
    loader,
    users,
    statistics,
    start_date,
    end_date,
    loader_kwargs,
    min_coverage,
    return_coverage,
):
    """Build complete daily rows from step summaries."""
    frames = []
    for user in users:
        summary = schema.prepare_activity_summary(
            loader.load_activity_summary(
                user,
                start_date=start_date,
                end_date=end_date,
                **loader_kwargs,
            )
        )
        start = (
            pd.Timestamp(start_date).normalize()
            if start_date is not None
            else summary[schema.CALENDAR_DATE_COL].min()
        )
        end = (
            pd.Timestamp(end_date).normalize()
            if end_date is not None
            else (
                summary[schema.CALENDAR_DATE_COL].max() + pd.Timedelta(days=1)
                if not summary.empty
                else pd.NaT
            )
        )
        if pd.isna(start) or pd.isna(end):
            continue
        if end <= start:
            raise ValueError("end_date must be after start_date")
        values = summary.set_index(schema.CALENDAR_DATE_COL)[schema.ACTIVITY_STEPS_COL]
        windows = hierarchy.windows_for_resolution(
            [user], "daily", {user: (start, end)}
        ).xs(user)
        records = []
        for label, window in windows.iterrows():
            value = values.get(pd.Timestamp(label).normalize(), np.nan)
            coverage = float(pd.notna(value))
            if min_coverage is not None and coverage < min_coverage:
                value = np.nan
            records.append(
                {
                    schema.USER_COL: user,
                    "window_start": label,
                    "period_start": window["start_time"],
                    "period_end": window["end_time"],
                    ACTIVITY_STATISTIC_STEPS: value,
                    **(
                        {f"{ACTIVITY_STATISTIC_STEPS}__coverage": coverage}
                        if return_coverage
                        else {}
                    ),
                }
            )
        frames.append(
            pd.DataFrame(records).set_index([schema.USER_COL, "window_start"])
        )
    return (
        pd.concat(frames).sort_index()
        if frames
        else hierarchy.empty_trajectory(statistics, return_coverage)
    )


def _aggregate_default_level(
    trajectory,
    parent,
    target,
    temporal,
    membership,
    min_coverage,
    min_valid_children,
    time_col,
):
    """Aggregate step totals by sum and peak descriptors by mean."""
    metadata = [
        column
        for column in (
            "period_start",
            "period_end",
            "interval_valid",
            "end_source",
            *hierarchy.TIME_METADATA_COLUMNS,
        )
        if column in trajectory
    ]
    features = [
        column
        for column in trajectory
        if column not in metadata and not column.endswith("__coverage")
    ]
    groups = {
        "sum": [
            feature
            for feature in features
            if feature.startswith(ACTIVITY_STATISTIC_STEPS)
        ],
        "mean": [
            feature
            for feature in features
            if not feature.startswith(ACTIVITY_STATISTIC_STEPS)
        ],
    }
    frames = []
    next_temporal = set()
    output_features = []
    for operation, group in groups.items():
        if not group:
            continue
        frame, group_temporal = hierarchy.aggregate_level(
            trajectory[[*metadata, *group]],
            parent,
            target_resolution=target,
            aggregation=operation,
            temporal_features=temporal.intersection(group),
            membership=membership,
            min_coverage=min_coverage,
            min_valid_children=min_valid_children,
            allow_sum=True,
            time_col=time_col,
        )
        frames.append(frame)
        next_temporal.update(group_temporal)
        output_features.extend(
            f"{feature}__{target}_{operation}" for feature in group
        )
    return (
        hierarchy.combine_trajectories(
            frames, output_features, include_coverage=True
        ),
        next_temporal,
    )


def get_activity_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, tuple, None] = "all",
    start_date=None,
    end_date=None,
    resolution: Union[str, pd.Timedelta, list, tuple] = "daily",
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    aggregation=None,
    min_coverage=None,
    cycle_kwargs: dict | None = None,
    min_valid_children=None,
):
    """Return step-volume and timing features at one or more resolutions.

    Parameters
    ----------
    loader
        Loader exposing timestamped incremental step intervals or calendar-day
        step totals through the shared activity schema.
    statistic
        One feature name or a sequence. Timestamped sources support
        ``activity_steps``, ``activity_peak_60min_steps``, and
        ``activity_peak_60min_start``. Daily-summary sources support only
        ``activity_steps``. Omitting the argument selects every supported
        feature for that source.
    user_id
        One participant, several participants, or ``"all"``/``None``.
    start_date, end_date
        Inclusive/exclusive bounds. Calendar resolutions use participant-local
        dates; fixed resolutions use the clock selected by ``time_reference``.
    resolution
        A fixed duration, calendar or sleep-cycle name, or an increasing
        hierarchy. ``"1h"`` returns hourly step totals. Calendar-day sources
        start at daily resolution. Weekly, monthly, and yearly scalar requests
        are shorthand for daily-to-parent hierarchies.
    loader_kwargs
        Extra keyword arguments passed only to the loader method.
    time_reference
        ``"auto"``, ``"local"``, or ``"utc"``. Calendar and cycle windows
        default to local time; fixed windows prefer UTC when available.
    aggregation
        Operation at each upper hierarchy level. Activity defaults are
        feature-specific: steps are summed, while peak values and peak clock
        times are averaged. Thus ``resolution="weekly"`` returns total weekly
        steps, whereas ``resolution=["daily", "weekly"], aggregation="mean"``
        returns mean valid daily steps.
    min_coverage
        Optional threshold per level. For timestamped primary windows,
        coverage is the union of one-minute intervals containing valid heart
        rate and native intervals containing positive steps, divided by the
        requested window duration. For native daily summaries it is value
        availability. At upper levels it is the fraction of valid child
        features. Values below threshold become missing while measured
        coverage remains in ``<feature>__coverage`` columns.
    cycle_kwargs
        Extra arguments passed to :func:`pywearable.cycle.get_intervals`.
    min_valid_children
        Optional minimum number of valid children at each upper level.

    Returns
    -------
    pandas.DataFrame
        A ``(user_id, time)`` trajectory containing every requested feature
        and its ``__coverage`` column. An interval contributes when it is
        fully contained in the requested window. Peak 60-minute values require
        one completely observed contiguous hour inside the requested window.
        Fixed-duration output also contains the alternate clock and the
        ``timezone`` UTC offset for each window.
    """
    source = getattr(loader, "activity_source", None)
    if source not in {"samples", "daily_summary"}:
        raise ValueError(
            f"{type(loader).__name__} does not expose a supported activity source"
        )
    if source == "daily_summary" and time_reference == "utc":
        raise ValueError("native daily activity summaries use local calendar labels")
    statistics = _statistics_list(statistic, source)
    chain, hierarchical = _prepare_chain(resolution, source)
    if not hierarchical and aggregation is not None:
        raise ValueError("aggregation requires a resolution hierarchy")
    if not hierarchical and min_valid_children is not None:
        raise ValueError("min_valid_children requires a resolution hierarchy")
    users = loader.resolve_user_ids(user_id)
    loader_kwargs = {} if loader_kwargs is None else dict(loader_kwargs)
    cycle_kwargs = {} if cycle_kwargs is None else dict(cycle_kwargs)
    aggregation_levels = hierarchy.normalize_aggregation_levels(
        aggregation, len(chain) - 1
    )
    coverage_thresholds = hierarchy.normalize_coverage_thresholds(
        min_coverage, len(chain)
    )
    child_thresholds = hierarchy.normalize_valid_child_thresholds(
        min_valid_children, len(chain)
    )
    fixed_time_col = None
    if source == "samples":
        fixed_time_col = hierarchy.resolve_time_col(
            getattr(loader, "activity_time_columns", schema.ACTIVITY_TIME_COLUMNS),
            time_reference,
            chain,
            "activity",
        )
    bound_time_col = fixed_time_col or schema.LOCAL_TIME_COL
    start_date = schema.normalize_time_bound(start_date, bound_time_col)
    end_date = schema.normalize_time_bound(end_date, bound_time_col)
    cycle_target = next((value for value in chain if value in cycle.CYCLE_TYPES), None)
    cycle_intervals = (
        cycle.get_intervals(
            loader,
            cycle_type=cycle_target,
            user_id=users,
            start_date=start_date,
            end_date=end_date,
            **cycle_kwargs,
        )
        if cycle_target is not None
        else None
    )
    include_coverage = True
    if source == "samples":
        trajectory = _sample_primary_trajectory(
            loader,
            users,
            statistics,
            chain[0],
            chain,
            start_date,
            end_date,
            time_reference,
            loader_kwargs,
            coverage_thresholds[0],
            include_coverage,
            cycle_intervals,
        )
    else:
        trajectory = _summary_primary_trajectory(
            loader,
            users,
            statistics,
            start_date,
            end_date,
            loader_kwargs,
            coverage_thresholds[0],
            include_coverage,
        )

    if hierarchical:
        renamed = {name: f"{name}__{chain[0]}" for name in statistics}
        trajectory = trajectory.rename(columns=renamed)
        temporal = (
            {renamed[ACTIVITY_STATISTIC_PEAK_60MIN_START]}
            if ACTIVITY_STATISTIC_PEAK_60MIN_START in renamed
            else set()
        )
        for level, target in enumerate(chain[1:]):
            parent, membership = hierarchy.parent_windows_from_trajectory(
                loader,
                users,
                trajectory,
                target,
                chain[level],
                start_date,
                end_date,
                cycle_kwargs,
                cycle_intervals if target == cycle_target else None,
                prefer_explicit_bounds=True,
            )
            if parent is None:
                return trajectory.iloc[0:0].rename_axis(
                    index={"window_start": schema.CALENDAR_DATE_COL}
                )
            if (
                chain[level] not in hierarchy.CALENDAR_RESOLUTIONS
                and chain[level] not in cycle.CYCLE_TYPES
            ):
                membership = "physical"
            specification = aggregation_levels[level]
            if specification is None:
                trajectory, temporal = _aggregate_default_level(
                    trajectory,
                    parent,
                    target,
                    temporal,
                    membership,
                    coverage_thresholds[level + 1],
                    child_thresholds[level + 1],
                    fixed_time_col,
                )
            else:
                trajectory, temporal = hierarchy.aggregate_level(
                    trajectory,
                    parent,
                    target_resolution=target,
                    aggregation=specification,
                    temporal_features=temporal,
                    membership=membership,
                    min_coverage=coverage_thresholds[level + 1],
                    min_valid_children=child_thresholds[level + 1],
                    allow_sum=True,
                    time_col=fixed_time_col,
                )
        index_name = (
            schema.CALENDAR_DATE_COL
            if chain[-1] in hierarchy.CALENDAR_RESOLUTIONS
            or chain[-1] in cycle.CYCLE_TYPES
            else hierarchy.resolve_time_col(
                getattr(loader, "activity_time_columns", schema.ACTIVITY_TIME_COLUMNS),
                time_reference,
                chain,
                "activity",
            )
        )
        trajectory = trajectory.rename_axis(index={"window_start": index_name})
        if hierarchy.fixed_duration(chain[-1]) is not None:
            return hierarchy.format_fixed_time_output(trajectory, index_name)
        return trajectory

    columns = [
        column
        for name in statistics
        for column in (name, f"{name}__coverage")
    ]
    result = trajectory[
        columns
        + [
            column
            for column in hierarchy.TIME_METADATA_COLUMNS
            if column in trajectory
        ]
    ].copy()
    index_name = (
        schema.CALENDAR_DATE_COL
        if chain[0] in hierarchy.CALENDAR_RESOLUTIONS or chain[0] in cycle.CYCLE_TYPES
        else hierarchy.resolve_time_col(
            getattr(loader, "activity_time_columns", schema.ACTIVITY_TIME_COLUMNS),
            time_reference,
            chain,
            "activity",
        )
    )
    result = result.rename_axis(index={"window_start": index_name})
    if hierarchy.fixed_duration(chain[0]) is not None:
        return hierarchy.format_fixed_time_output(result, index_name)
    return result


def get_activity_statistic(loader: BaseLoader, statistic: str, **kwargs):
    """Return one named activity feature; see :func:`get_activity_statistics`."""
    return get_activity_statistics(loader, statistic=statistic, **kwargs)


def get_daily_activity_statistics(loader: BaseLoader, **kwargs):
    """Return participant-local daily activity features.

    Timestamped step intervals are aggregated into local calendar days, while
    calendar-day step totals retain their date labels.
    """
    return get_activity_statistics(loader, resolution="daily", **kwargs)


__all__ = [
    "ACTIVITY_STATISTICS",
    "ACTIVITY_STATISTIC_PEAK_60MIN_START",
    "ACTIVITY_STATISTIC_PEAK_60MIN_STEPS",
    "ACTIVITY_STATISTIC_STEPS",
    "get_activity_statistic",
    "get_activity_statistics",
    "get_daily_activity_statistics",
]
