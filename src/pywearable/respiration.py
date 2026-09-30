"""Respiratory-rate and oxygen-saturation feature extraction."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd

from . import cycle, hierarchy, schema
from .loader.base import BaseLoader, resolve_user_ids
from .utils import time as time_utils


RESPIRATORY_RATE_STATISTIC_MEAN = "respiratory_rate_mean"
RESPIRATORY_RATE_STATISTIC_MEDIAN = "respiratory_rate_median"
RESPIRATORY_RATE_STATISTIC_STD = "respiratory_rate_std"
RESPIRATORY_RATE_STATISTIC_P10 = "respiratory_rate_p10"
RESPIRATORY_RATE_STATISTIC_P90 = "respiratory_rate_p90"
RESPIRATORY_RATE_STATISTIC_COUNT = "respiratory_rate_count"

SPO2_STATISTIC_MEAN = "spo2_mean"
SPO2_STATISTIC_MEDIAN = "spo2_median"
SPO2_STATISTIC_STD = "spo2_std"
SPO2_STATISTIC_P10 = "spo2_p10"
SPO2_STATISTIC_MIN = "spo2_min"
SPO2_STATISTIC_COUNT = "spo2_count"
SPO2_STATISTIC_BELOW_THRESHOLD_FRACTION = "spo2_below_threshold_fraction"
SPO2_STATISTIC_BELOW_THRESHOLD_DURATION = "spo2_below_threshold_duration_seconds"

RESPIRATORY_RATE_STATISTICS = (
    RESPIRATORY_RATE_STATISTIC_MEAN,
    RESPIRATORY_RATE_STATISTIC_MEDIAN,
    RESPIRATORY_RATE_STATISTIC_STD,
    RESPIRATORY_RATE_STATISTIC_P10,
    RESPIRATORY_RATE_STATISTIC_P90,
    RESPIRATORY_RATE_STATISTIC_COUNT,
)
SPO2_STATISTICS = (
    SPO2_STATISTIC_MEAN,
    SPO2_STATISTIC_MEDIAN,
    SPO2_STATISTIC_STD,
    SPO2_STATISTIC_P10,
    SPO2_STATISTIC_MIN,
    SPO2_STATISTIC_COUNT,
    SPO2_STATISTIC_BELOW_THRESHOLD_FRACTION,
    SPO2_STATISTIC_BELOW_THRESHOLD_DURATION,
)
RESPIRATION_STATISTICS = RESPIRATORY_RATE_STATISTICS + SPO2_STATISTICS

_FAMILY_CONFIG = {
    "respiratory_rate": {
        "value_col": schema.RESPIRATORY_RATE_COL,
        "source_attr": "respiratory_rate_source",
        "time_columns_attr": "respiratory_rate_time_columns",
        "sample_loader": "load_respiration",
        "sleep_sample_loader": "load_sleep_respiration",
        "summary_loader": "load_respiration_summary",
    },
    "spo2": {
        "value_col": schema.SPO2_COL,
        "source_attr": "spo2_source",
        "time_columns_attr": "spo2_time_columns",
        "sample_loader": "load_pulse_ox",
        "sleep_sample_loader": "load_sleep_pulse_ox",
        "summary_loader": "load_spo2_summary",
    },
}


def _statistics_list(statistic) -> list[str]:
    """Return unique requested feature names, validating core and ``p0``–``p100`` names.

    ``None`` requests all core respiratory-rate and SpO2 statistics.
    """
    if statistic is None:
        return list(RESPIRATION_STATISTICS)
    statistics = [statistic] if isinstance(statistic, str) else list(statistic)
    unknown = [
        name
        for name in statistics
        if name not in RESPIRATION_STATISTICS
        and hierarchy.feature_percentile(name, "respiratory_rate") is None
        and hierarchy.feature_percentile(name, "spo2") is None
    ]
    if unknown:
        raise ValueError(
            f"statistic must be one among {list(RESPIRATION_STATISTICS)} or a "
            f"percentile such as 'spo2_p20', not {sorted(set(unknown))}"
        )
    return list(dict.fromkeys(statistics))


def _below_threshold_duration(rows, time_col, value_col, threshold, cadence, end):
    """Estimate seconds below ``threshold`` from timestamped SpO2 ``rows``.

    Each reading represents at most one ``cadence`` up to ``end``; return
    missing when cadence cannot be inferred.
    """
    if cadence is None or rows.empty:
        return np.nan if cadence is None else 0.0
    ordered = rows.sort_values(time_col, kind="stable").drop_duplicates(time_col)
    timestamps = pd.DatetimeIndex(ordered[time_col])
    next_times = pd.Series(timestamps[1:].append(pd.DatetimeIndex([end])))
    durations = pd.to_timedelta(next_times.to_numpy() - timestamps.to_numpy())
    durations = durations.where(durations <= cadence, cadence)
    durations = durations.where(durations > pd.Timedelta(0), pd.Timedelta(0))
    below = ordered[value_col].lt(threshold).to_numpy()
    return float(durations[below].sum().total_seconds())


def _summarize_samples(
    rows, family, statistics, time_col, end, cadence, spo2_threshold
):
    """Return requested feature values for one window of canonical sample rows.

    ``family`` selects respiratory rate or SpO2. ``end`` and ``cadence``
    bound the SpO2 below-threshold duration; unavailable features stay missing.
    """
    config = _FAMILY_CONFIG[family]
    values = rows[config["value_col"]].dropna()
    result = {name: np.nan for name in statistics}
    for name in statistics:
        percentile = hierarchy.feature_percentile(name, family)
        if percentile is not None:
            result[name] = values.quantile(percentile) if not values.empty else np.nan
        elif name.endswith("_mean"):
            result[name] = values.mean()
        elif name.endswith("_median"):
            result[name] = values.median()
        elif name.endswith("_std"):
            result[name] = values.std(ddof=1)
        elif name.endswith("_min"):
            result[name] = values.min()
        elif name.endswith("_count"):
            result[name] = int(values.size)
        elif name == SPO2_STATISTIC_BELOW_THRESHOLD_FRACTION:
            result[name] = (
                float(values.lt(spo2_threshold).mean())
                if not values.empty
                else np.nan
            )
        elif name == SPO2_STATISTIC_BELOW_THRESHOLD_DURATION:
            result[name] = _below_threshold_duration(
                rows,
                time_col,
                config["value_col"],
                spo2_threshold,
                cadence,
                end,
            )
    return result


def _resolve_time_col(loader, family, time_reference, chain):
    """Return a loader-supported timestamp column for this family and hierarchy."""
    available = getattr(
        loader,
        _FAMILY_CONFIG[family]["time_columns_attr"],
        schema.RESPIRATION_TIME_COLUMNS,
    )
    return hierarchy.resolve_time_col(available, time_reference, chain, family)


def _prepare_samples(frame, family):
    """Return validated canonical sample rows for respiratory rate or SpO2."""
    if family == "respiratory_rate":
        return schema.prepare_respiration_samples(frame)
    return schema.prepare_spo2_samples(frame)


def _load_sample_family(
    loader, family, user, start_date, end_date, loader_kwargs, sleep_only=False
):
    """Load and normalize one user's dated samples for ``family``.

    ``sleep_only`` selects the loader's sleep-filtered stream; the result is a
    canonical timestamped DataFrame.
    """
    config = _FAMILY_CONFIG[family]
    method_name = (
        config["sleep_sample_loader"] if sleep_only else config["sample_loader"]
    )
    frame = getattr(loader, method_name)(
        user_id=user,
        start_date=start_date,
        end_date=end_date,
        **loader_kwargs,
    )
    return _prepare_samples(frame, family)


def _sample_family_trajectory(
    loader,
    users,
    family,
    statistics,
    chain,
    start_date,
    end_date,
    time_reference,
    loader_kwargs,
    min_coverage,
    return_coverage,
    cycle_intervals,
    spo2_threshold,
    unresolved_coverage,
):
    """Build one family's feature trajectory from raw timestamped samples.

    ``chain[0]`` sets the primary fixed or cycle windows. Each row of the
    returned ``(user_id, window_start)`` DataFrame contains period bounds,
    requested values, and optional sample coverage. Unknown coverage streams
    are recorded in ``unresolved_coverage`` when a threshold is requested.
    """
    frames = []
    resolution = chain[0]
    is_cycle = resolution in cycle.CYCLE_TYPES
    time_col = _resolve_time_col(loader, family, time_reference, chain)
    intervals = cycle_intervals
    interval_users = (
        set(intervals.index.get_level_values(0)) if intervals is not None else set()
    )

    for user in users:
        user_intervals = None
        load_start, load_end = start_date, end_date
        if intervals is not None:
            if user not in interval_users:
                continue
            user_intervals = intervals.xs(user)
            valid = user_intervals[user_intervals["interval_valid"]]
            if valid.empty:
                continue
            load_start = valid["start_time"].min()
            load_end = valid["end_time"].max()
        elif start_date is not None or end_date is not None:
            padding = pd.Timedelta(days=2)
            load_start = (
                None if start_date is None else pd.Timestamp(start_date) - padding
            )
            load_end = None if end_date is None else pd.Timestamp(end_date) + padding
        frame = _load_sample_family(
            loader,
            family,
            user,
            load_start,
            load_end,
            loader_kwargs,
            sleep_only=cycle.CYCLE_NIGHT in chain,
        )
        frame = frame[frame[time_col].notna()].sort_values(time_col, kind="stable")
        cadence = time_utils.infer_cadence(
            frame[time_col], min_gaps=3, min_cluster_fraction=0.5
        )

        if is_cycle:
            windows = user_intervals
        else:
            bounds = (
                (load_start, load_end)
                if user_intervals is not None
                else hierarchy.observed_bounds(
                    frame, time_col, start_date, end_date, resolution
                )
            )
            if bounds is None:
                continue
            start, end = bounds
            frame = frame[frame[time_col].ge(start) & frame[time_col].lt(end)]
            windows = hierarchy.windows_for_resolution(
                [user], resolution, {user: bounds}
            ).xs(user)

        timestamps = pd.DatetimeIndex(frame[time_col])
        records = []
        for label, window in windows.iterrows():
            valid_interval = bool(window.get("interval_valid", True))
            values = {name: np.nan for name in statistics}
            coverage = np.nan
            if valid_interval:
                left = timestamps.searchsorted(window["start_time"], side="left")
                right = timestamps.searchsorted(window["end_time"], side="left")
                rows = frame.iloc[left:right]
                coverage = time_utils.sample_coverage(
                    rows[time_col].nunique(),
                    window["start_time"],
                    window["end_time"],
                    cadence,
                )
                accepted = min_coverage is None or (
                    pd.notna(coverage) and coverage >= min_coverage
                )
                if accepted:
                    values = _summarize_samples(
                        rows,
                        family,
                        statistics,
                        time_col,
                        window["end_time"],
                        cadence,
                        spo2_threshold,
                    )
                elif pd.isna(coverage):
                    unresolved_coverage.add((user, family))
            records.append(
                {
                    schema.USER_COL: user,
                    "window_start": label,
                    "period_start": window["start_time"],
                    "period_end": window["end_time"],
                    **values,
                    **(
                        {f"{name}__coverage": coverage for name in statistics}
                        if return_coverage
                        else {}
                    ),
                }
            )
        if records:
            frames.append(
                pd.DataFrame(records).set_index([schema.USER_COL, "window_start"])
            )
    return (
        pd.concat(frames).sort_index()
        if frames
        else hierarchy.empty_trajectory(statistics, return_coverage)
    )


def _summary_values(row, family, statistics):
    """Read available features from one canonical provider-summary row.

    Return a name-to-value mapping; statistics absent at this native grain
    remain missing rather than being reconstructed from summary means.
    """
    result = {name: np.nan for name in statistics}
    if family == "respiratory_rate":
        mapping = {
            RESPIRATORY_RATE_STATISTIC_MEAN: schema.RESPIRATORY_RATE_MEAN_COL,
            RESPIRATORY_RATE_STATISTIC_STD: schema.RESPIRATORY_RATE_STD_COL,
        }
    else:
        mapping = {SPO2_STATISTIC_MEAN: schema.SPO2_MEAN_COL}
    for statistic, column in mapping.items():
        if statistic in result:
            result[statistic] = row.get(column, np.nan)
    return result


def _summary_family_trajectory(
    loader,
    users,
    family,
    statistics,
    start_date,
    end_date,
    loader_kwargs,
    min_coverage,
    return_coverage,
    unresolved_coverage,
):
    """Build one family's native-night trajectory from provider summaries.

    ``start_date``/``end_date`` filter night labels, not artificial sample
    windows. Return a ``(user_id, window_start)`` DataFrame with period bounds,
    requested values, and optional provider-reported coverage.
    """
    frames = []
    method = getattr(loader, _FAMILY_CONFIG[family]["summary_loader"])
    for user in users:
        summary = method(
            user_id=user,
            start_date=start_date,
            end_date=end_date,
            **loader_kwargs,
        )
        summary = (
            schema.prepare_respiration_summary(summary)
            if family == "respiratory_rate"
            else schema.prepare_spo2_summary(summary)
        )
        if start_date is not None:
            summary = summary[
                summary[schema.CALENDAR_DATE_COL]
                >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            summary = summary[
                summary[schema.CALENDAR_DATE_COL]
                < pd.Timestamp(end_date).normalize()
            ]
        records = []
        for row in summary.to_dict(orient="records"):
            label = pd.Timestamp(row[schema.CALENDAR_DATE_COL]).normalize()
            period_start = row.get(schema.LOCAL_TIME_COL)
            period_end = row.get(schema.LOCAL_END_TIME_COL)
            period_start = (
                label if pd.isna(period_start) else pd.Timestamp(period_start)
            )
            period_end = (
                label + pd.Timedelta(days=1)
                if pd.isna(period_end)
                else pd.Timestamp(period_end)
            )
            coverage = row.get(f"{family}_coverage", np.nan)
            accepted = min_coverage is None or (
                pd.notna(coverage) and coverage >= min_coverage
            )
            values = (
                _summary_values(row, family, statistics)
                if accepted
                else {name: np.nan for name in statistics}
            )
            if min_coverage is not None and pd.isna(coverage):
                unresolved_coverage.add((user, family))
            records.append(
                {
                    schema.USER_COL: user,
                    "window_start": label,
                    "period_start": period_start,
                    "period_end": period_end,
                    **values,
                    **(
                        {f"{name}__coverage": coverage for name in statistics}
                        if return_coverage
                        else {}
                    ),
                }
            )
        if records:
            frames.append(
                pd.DataFrame(records).set_index([schema.USER_COL, "window_start"])
            )
    return (
        pd.concat(frames).sort_index()
        if frames
        else hierarchy.empty_trajectory(statistics, return_coverage)
    )


def get_respiration_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    resolution: Union[str, pd.Timedelta, list, tuple] = cycle.CYCLE_NIGHT,
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    aggregation=None,
    min_coverage=None,
    cycle_kwargs: dict | None = None,
    min_valid_children=None,
    spo2_threshold: float = 90.0,
) -> pd.DataFrame:
    """Extract respiratory-rate and SpO2 features at the requested resolution.

    Parameters
    ----------
    loader : BaseLoader
        Supplies canonical samples or provider-native nightly summaries.
    statistic : str, sequence of str, or None
        Feature name(s); ``None`` selects all core metrics. Percentiles such as
        ``spo2_p20`` and ``respiratory_rate_p75`` are also accepted.
    user_id : str, sequence of str, or None
        Participant(s); ``"all"`` or ``None`` selects every loader user.
    start_date, end_date : date-like or None
        Half-open time or night-label range, ``[start_date, end_date)``.
    resolution : str, Timedelta, or sequence
        Primary window or ascending hierarchy, for example ``"5min"`` or
        ``["5min", "cycle_night"]``. Provider summaries support only their
        native ``"cycle_night"`` grain and larger calendar periods.
    loader_kwargs, cycle_kwargs : dict or None
        Extra keyword arguments passed to the corresponding loader or cycle
        interval function.
    time_reference : {"auto", "local", "utc"}
        Clock for raw samples; ``"auto"`` uses local time for calendar/cycle
        windows and UTC for fixed windows when available. Provider summaries
        are date-labelled and do not support UTC selection.
    aggregation : str, sequence, or None
        Operation(s) at each parent level; requires a hierarchy. ``None``
        uses the shared standard aggregations.
    min_coverage : float, sequence, or None
        Minimum valid fraction at each level. Raw primary coverage estimates
        observed versus expected samples; provider coverage is used only when
        supplied. Unknown primary coverage fails an explicit threshold.
    min_valid_children : int, sequence, or None
        Minimum valid child windows at each parent level.
    spo2_threshold : float
        SpO2 percentage threshold for below-threshold fraction and duration;
        the comparison is strictly below this value.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(user_id, utc_time/local_time/calendar_date)`` with one
        ``__coverage`` column per requested feature. Hierarchical column names
        include each resolution and aggregation. Unsupported provider-summary
        features are missing, never inferred from summary means.
        Below-threshold duration is in seconds.
    """
    statistics = _statistics_list(statistic)
    users = resolve_user_ids(loader, user_id)
    loader_kwargs = {} if loader_kwargs is None else dict(loader_kwargs)
    cycle_kwargs = {} if cycle_kwargs is None else dict(cycle_kwargs)
    if not 0 <= spo2_threshold <= 100:
        raise ValueError("spo2_threshold must be between 0 and 100")

    chain, hierarchical = hierarchy.normalize_resolution_chain(resolution)
    chain = [cycle.CYCLE_NIGHT if value == "nightly" else value for value in chain]
    families = [
        family for family in _FAMILY_CONFIG if hierarchy.family_statistics(statistics, family)
    ]
    sources = {
        getattr(loader, _FAMILY_CONFIG[family]["source_attr"], "samples")
        for family in families
    }
    if not sources.issubset({"samples", "provider_summary"}):
        raise ValueError(
            "respiratory_rate_source and spo2_source must be 'samples' or "
            "'provider_summary'"
        )
    if len(sources) > 1:
        raise ValueError(
            "sample and provider-summary families must be requested separately"
        )
    source = next(iter(sources), "samples")

    direct_summary = False
    if source == "provider_summary":
        if time_reference == "utc":
            raise ValueError("provider respiration summaries use local calendar labels")
        primary = chain[0]
        if primary == cycle.CYCLE_NIGHT:
            direct_summary = not hierarchical
        elif not hierarchical and primary in {"weekly", "monthly", "yearly"}:
            chain = [cycle.CYCLE_NIGHT, primary]
            hierarchical = True
        else:
            raise ValueError(
                "provider respiration summaries are available only at their "
                "native nightly grain or larger calendar resolutions"
            )

    if not hierarchical and aggregation is not None:
        raise ValueError("aggregation requires a resolution hierarchy")
    if not hierarchical and min_valid_children is not None:
        raise ValueError("min_valid_children requires a resolution hierarchy")
    hierarchy.normalize_resolution_chain(chain)
    aggregation_levels = hierarchy.normalize_aggregation_levels(
        aggregation, len(chain) - 1
    )
    coverage_thresholds = hierarchy.normalize_coverage_thresholds(
        min_coverage, len(chain)
    )
    child_thresholds = hierarchy.normalize_valid_child_thresholds(
        min_valid_children, len(chain)
    )
    hierarchy.warn_if_wide(statistics, aggregation_levels)

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
        if source == "samples" and cycle_target is not None
        else None
    )

    unresolved_coverage = set()
    family_frames = []
    for family in families:
        requested = hierarchy.family_statistics(statistics, family)
        if source == "provider_summary":
            frame = _summary_family_trajectory(
                loader,
                users,
                family,
                requested,
                start_date,
                end_date,
                loader_kwargs,
                coverage_thresholds[0],
                True,
                unresolved_coverage,
            )
        else:
            frame = _sample_family_trajectory(
                loader,
                users,
                family,
                requested,
                chain,
                start_date,
                end_date,
                time_reference,
                loader_kwargs,
                coverage_thresholds[0],
                True,
                cycle_intervals,
                spo2_threshold,
                unresolved_coverage,
            )
        family_frames.append(frame)
    trajectory = hierarchy.combine_trajectories(
        family_frames, statistics, include_coverage=True
    )

    if coverage_thresholds[0] is not None and unresolved_coverage:
        warnings.warn(
            "Primary respiration coverage could not be determined for "
            f"{len(unresolved_coverage)} user/signal stream(s); features requiring "
            "a primary coverage threshold were excluded.",
            UserWarning,
            stacklevel=2,
        )

    if hierarchical:
        trajectory = trajectory.rename(
            columns={name: f"{name}__{chain[0]}" for name in statistics}
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
            )
            if parent is None:
                return trajectory.iloc[0:0].rename_axis(
                    index={"window_start": schema.CALENDAR_DATE_COL}
                )
            trajectory, _ = hierarchy.aggregate_level(
                trajectory,
                parent,
                target_resolution=target,
                aggregation=aggregation_levels[level],
                membership=membership,
                min_coverage=coverage_thresholds[level + 1],
                min_valid_children=child_thresholds[level + 1],
            )
        index_name = (
            schema.CALENDAR_DATE_COL
            if chain[-1] in hierarchy.CALENDAR_RESOLUTIONS
            or chain[-1] in cycle.CYCLE_TYPES
            else _resolve_time_col(loader, families[0], time_reference, chain)
        )
        return trajectory.rename_axis(index={"window_start": index_name})

    columns = [
        column
        for name in statistics
        for column in (name, f"{name}__coverage")
    ]
    result = trajectory[columns].copy()
    index_name = (
        schema.CALENDAR_DATE_COL
        if direct_summary
        or chain[0] in hierarchy.CALENDAR_RESOLUTIONS
        or chain[0] in cycle.CYCLE_TYPES
        else _resolve_time_col(loader, families[0], time_reference, chain)
    )
    result = result.rename_axis(index={"window_start": index_name})
    return result


def get_respiration_statistic(loader: BaseLoader, statistic: str, **kwargs):
    """Return one named respiratory-rate or SpO2 feature.

    ``loader`` and ``kwargs`` have the same meaning as in
    :func:`get_respiration_statistics`. A direct single-feature request
    always includes the feature's coverage column.
    """
    return get_respiration_statistics(loader, statistic=statistic, **kwargs)


def get_five_minute_respiration_statistics(loader: BaseLoader, **kwargs):
    """Return raw-sample features in non-overlapping five-minute windows.

    Pass feature, user, date, and coverage options via ``kwargs`` as in
    :func:`get_respiration_statistics`. Provider-only nightly summaries cannot
    produce this resolution. Output is a DataFrame by user and time.
    """
    return get_respiration_statistics(loader, resolution="5min", **kwargs)


def get_night_respiration_statistics(loader: BaseLoader, **kwargs):
    """Return one feature row per main-sleep night.

    Raw samples are grouped by sleep-cycle intervals; provider summaries are
    returned at their native night labels. ``kwargs`` are forwarded to
    :func:`get_respiration_statistics`; output is indexed by user and night.
    """
    return get_respiration_statistics(loader, resolution=cycle.CYCLE_NIGHT, **kwargs)


def get_awake_respiration_statistics(loader: BaseLoader, **kwargs):
    """Return one raw-sample feature row per awake cycle.

    Pass feature, user, date, and coverage options via ``kwargs`` as in
    :func:`get_respiration_statistics`. Provider-only nightly summaries cannot
    produce awake cycles. Output is indexed by user and cycle date.
    """
    return get_respiration_statistics(loader, resolution=cycle.CYCLE_AWAKE, **kwargs)


__all__ = [
    "RESPIRATION_STATISTICS",
    "RESPIRATORY_RATE_STATISTIC_COUNT",
    "RESPIRATORY_RATE_STATISTIC_MEAN",
    "RESPIRATORY_RATE_STATISTIC_MEDIAN",
    "RESPIRATORY_RATE_STATISTIC_P10",
    "RESPIRATORY_RATE_STATISTIC_P90",
    "RESPIRATORY_RATE_STATISTIC_STD",
    "RESPIRATORY_RATE_STATISTICS",
    "SPO2_STATISTIC_BELOW_THRESHOLD_DURATION",
    "SPO2_STATISTIC_BELOW_THRESHOLD_FRACTION",
    "SPO2_STATISTIC_COUNT",
    "SPO2_STATISTIC_MEAN",
    "SPO2_STATISTIC_MEDIAN",
    "SPO2_STATISTIC_MIN",
    "SPO2_STATISTIC_P10",
    "SPO2_STATISTIC_STD",
    "SPO2_STATISTICS",
    "get_awake_respiration_statistics",
    "get_five_minute_respiration_statistics",
    "get_respiration_statistic",
    "get_respiration_statistics",
    "get_night_respiration_statistics",
]
