"""Extraction of explicitly provider-native wearable features.

Feature names retain their provider prefix because similarly named vendor
scores are not assumed to measure the same construct. Timestamped streams can
be windowed or assigned to cycles; native daily and nightly features can only
remain at their native grain or be aggregated upward.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from . import cycle, hierarchy, schema
from .loader.base import BaseLoader
from .utils import time as time_utils


GARMIN_STRESS_LEVEL = "garmin_stress_level"
GARMIN_BODY_BATTERY = "garmin_body_battery"
GARMIN_DAILY_AVERAGE_STRESS = "garmin_daily_average_stress"
GARMIN_DAILY_MAX_STRESS = "garmin_daily_max_stress"
GARMIN_DAILY_STRESS_DURATION_SECONDS = "garmin_daily_stress_duration_seconds"
GARMIN_DAILY_REST_STRESS_DURATION_SECONDS = (
    "garmin_daily_rest_stress_duration_seconds"
)
GARMIN_DAILY_ACTIVITY_STRESS_DURATION_SECONDS = (
    "garmin_daily_activity_stress_duration_seconds"
)
GARMIN_DAILY_LOW_STRESS_DURATION_SECONDS = (
    "garmin_daily_low_stress_duration_seconds"
)
GARMIN_DAILY_MEDIUM_STRESS_DURATION_SECONDS = (
    "garmin_daily_medium_stress_duration_seconds"
)
GARMIN_DAILY_HIGH_STRESS_DURATION_SECONDS = (
    "garmin_daily_high_stress_duration_seconds"
)
GARMIN_DAILY_STRESS_QUALIFIER = "garmin_daily_stress_qualifier"

GARMIN_BODY_BATTERY_BEDTIME = "garmin_body_battery_bedtime"
GARMIN_BODY_BATTERY_GETUP = "garmin_body_battery_getup"
GARMIN_BODY_BATTERY_SLEEP_GAIN = "garmin_body_battery_sleep_gain"
GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION = (
    "garmin_body_battery_sleep_recovery_fraction"
)

FITBIT_STRESS_MANAGEMENT_SCORE = "fitbit_stress_management_score"
FITBIT_STRESS_SLEEP_POINTS = "fitbit_stress_sleep_points"
FITBIT_STRESS_MAX_SLEEP_POINTS = "fitbit_stress_max_sleep_points"
FITBIT_STRESS_RESPONSIVENESS_POINTS = "fitbit_stress_responsiveness_points"
FITBIT_STRESS_MAX_RESPONSIVENESS_POINTS = (
    "fitbit_stress_max_responsiveness_points"
)
FITBIT_STRESS_EXERTION_POINTS = "fitbit_stress_exertion_points"
FITBIT_STRESS_MAX_EXERTION_POINTS = "fitbit_stress_max_exertion_points"

OURA_READINESS_SCORE = "oura_readiness_score"
OURA_READINESS_TEMPERATURE_DEVIATION = (
    "oura_readiness_temperature_deviation"
)
OURA_READINESS_TEMPERATURE_TREND_DEVIATION = (
    "oura_readiness_temperature_trend_deviation"
)
OURA_READINESS_ACTIVITY_BALANCE = "oura_readiness_activity_balance"
OURA_READINESS_BODY_TEMPERATURE = "oura_readiness_body_temperature"
OURA_READINESS_HRV_BALANCE = "oura_readiness_hrv_balance"
OURA_READINESS_PREVIOUS_DAY_ACTIVITY = (
    "oura_readiness_previous_day_activity"
)
OURA_READINESS_PREVIOUS_NIGHT = "oura_readiness_previous_night"
OURA_READINESS_RECOVERY_INDEX = "oura_readiness_recovery_index"
OURA_READINESS_RESTING_HEART_RATE = (
    "oura_readiness_resting_heart_rate"
)
OURA_READINESS_SLEEP_BALANCE = "oura_readiness_sleep_balance"


@dataclass(frozen=True)
class _FeatureDefinition:
    """Describe a provider feature's native rows and aggregation capability.

    ``grain`` is ``"samples"`` for timestamped observations, ``"daily"`` for
    provider-native date summaries, or a cycle name for a derived cycle row.
    ``numeric`` controls whether parent-level numeric aggregation is valid.
    """

    grain: str
    numeric: bool = True


_FEATURE_DEFINITIONS = {
    GARMIN_STRESS_LEVEL: _FeatureDefinition("samples"),
    GARMIN_BODY_BATTERY: _FeatureDefinition("samples"),
    GARMIN_DAILY_AVERAGE_STRESS: _FeatureDefinition("daily"),
    GARMIN_DAILY_MAX_STRESS: _FeatureDefinition("daily"),
    GARMIN_DAILY_STRESS_DURATION_SECONDS: _FeatureDefinition("daily"),
    GARMIN_DAILY_REST_STRESS_DURATION_SECONDS: _FeatureDefinition("daily"),
    GARMIN_DAILY_ACTIVITY_STRESS_DURATION_SECONDS: _FeatureDefinition("daily"),
    GARMIN_DAILY_LOW_STRESS_DURATION_SECONDS: _FeatureDefinition("daily"),
    GARMIN_DAILY_MEDIUM_STRESS_DURATION_SECONDS: _FeatureDefinition("daily"),
    GARMIN_DAILY_HIGH_STRESS_DURATION_SECONDS: _FeatureDefinition("daily"),
    GARMIN_DAILY_STRESS_QUALIFIER: _FeatureDefinition("daily", numeric=False),
    GARMIN_BODY_BATTERY_BEDTIME: _FeatureDefinition(cycle.CYCLE_NIGHT),
    GARMIN_BODY_BATTERY_GETUP: _FeatureDefinition(cycle.CYCLE_NIGHT),
    GARMIN_BODY_BATTERY_SLEEP_GAIN: _FeatureDefinition(cycle.CYCLE_NIGHT),
    GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION: _FeatureDefinition(
        cycle.CYCLE_NIGHT
    ),
    FITBIT_STRESS_MANAGEMENT_SCORE: _FeatureDefinition("daily"),
    FITBIT_STRESS_SLEEP_POINTS: _FeatureDefinition("daily"),
    FITBIT_STRESS_MAX_SLEEP_POINTS: _FeatureDefinition("daily"),
    FITBIT_STRESS_RESPONSIVENESS_POINTS: _FeatureDefinition("daily"),
    FITBIT_STRESS_MAX_RESPONSIVENESS_POINTS: _FeatureDefinition("daily"),
    FITBIT_STRESS_EXERTION_POINTS: _FeatureDefinition("daily"),
    FITBIT_STRESS_MAX_EXERTION_POINTS: _FeatureDefinition("daily"),
    OURA_READINESS_SCORE: _FeatureDefinition("daily"),
    OURA_READINESS_TEMPERATURE_DEVIATION: _FeatureDefinition("daily"),
    OURA_READINESS_TEMPERATURE_TREND_DEVIATION: _FeatureDefinition("daily"),
    OURA_READINESS_ACTIVITY_BALANCE: _FeatureDefinition("daily"),
    OURA_READINESS_BODY_TEMPERATURE: _FeatureDefinition("daily"),
    OURA_READINESS_HRV_BALANCE: _FeatureDefinition("daily"),
    OURA_READINESS_PREVIOUS_DAY_ACTIVITY: _FeatureDefinition("daily"),
    OURA_READINESS_PREVIOUS_NIGHT: _FeatureDefinition("daily"),
    OURA_READINESS_RECOVERY_INDEX: _FeatureDefinition("daily"),
    OURA_READINESS_RESTING_HEART_RATE: _FeatureDefinition("daily"),
    OURA_READINESS_SLEEP_BALANCE: _FeatureDefinition("daily"),
}

PROVIDER_FEATURES = tuple(_FEATURE_DEFINITIONS)
_BODY_BATTERY_SLEEP_FEATURES = {
    GARMIN_BODY_BATTERY_BEDTIME,
    GARMIN_BODY_BATTERY_GETUP,
    GARMIN_BODY_BATTERY_SLEEP_GAIN,
    GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION,
}
_SAMPLE_STATISTICS = {"mean", "median", "std", "min", "max", "count"}


def get_available_provider_features(loader: BaseLoader) -> list[str]:
    """Return provider-feature names implemented by ``loader``.

    Parameters
    ----------
    loader
        Loader whose provider-native feature support should be inspected.

    Returns
    -------
    list of str
        Sorted feature names accepted by :func:`get_provider_features`.
    """
    available = set(getattr(loader, "provider_features", ()))
    if GARMIN_BODY_BATTERY in available:
        available.update(_BODY_BATTERY_SLEEP_FEATURES)
    return sorted(available.intersection(_FEATURE_DEFINITIONS))


def _requested_features(loader, feature) -> list[str]:
    """Validate feature names, loader support, and one shared native grain."""
    features = [feature] if isinstance(feature, str) else list(feature)
    if not features:
        raise ValueError("feature must contain at least one provider feature")
    features = list(dict.fromkeys(features))
    available = set(get_available_provider_features(loader))
    unknown = set(features).difference(_FEATURE_DEFINITIONS)
    if unknown:
        raise ValueError(f"unknown provider feature(s): {sorted(unknown)}")
    unsupported = set(features).difference(available)
    if unsupported:
        raise ValueError(
            f"{type(loader).__name__} does not provide {sorted(unsupported)}; "
            f"available features are {sorted(available)}"
        )
    grains = {_FEATURE_DEFINITIONS[name].grain for name in features}
    if len(grains) != 1:
        raise ValueError(
            "all requested provider features must have the same native grain"
        )
    return features


def _sample_statistics(statistic) -> list[str]:
    """Return validated primary reductions for timestamped features."""
    if statistic is None:
        return ["mean"]
    statistics = [statistic] if isinstance(statistic, str) else list(statistic)
    invalid = [
        value
        for value in statistics
        if value not in _SAMPLE_STATISTICS and not hierarchy.is_percentile(value)
    ]
    if not statistics or invalid:
        raise ValueError(
            "sample statistics must be mean, median, std, min, max, count, "
            f"or percentiles such as p10; invalid values: {invalid}"
        )
    return list(dict.fromkeys(statistics))


def _prepare_chain(resolution, grain):
    """Apply provider-feature native-grain rules to a normalized chain.

    Shared hierarchy validation checks that resolutions increase. This helper
    additionally prevents native daily or nightly provider values from being
    expanded into smaller artificial windows and expands scalar parent
    requests such as ``"monthly"`` to ``[native_grain, "monthly"]``.
    """
    chain, hierarchical = hierarchy.normalize_resolution_chain(resolution)
    chain = [cycle.CYCLE_NIGHT if value == "nightly" else value for value in chain]
    if grain == "samples":
        hierarchy.normalize_resolution_chain(chain)
        return chain, hierarchical

    native = grain
    if not hierarchical and chain[0] == native:
        return chain, False
    allowed_parents = {"weekly", "monthly", "yearly"}
    if not hierarchical and chain[0] in allowed_parents:
        chain = [native, chain[0]]
        hierarchy.normalize_resolution_chain(chain)
        return chain, True
    if hierarchical and chain[0] == native:
        hierarchy.normalize_resolution_chain(chain)
        return chain, True
    label = "native daily" if native == "daily" else "native cycle_night"
    raise ValueError(
        f"{label} provider features can only remain at that grain or be "
        "aggregated to weekly, monthly, or yearly resolutions"
    )


def _summarize(values: pd.Series, statistic: str):
    """Reduce valid numeric sample values with one supported statistic."""
    if statistic == "count":
        return int(pd.to_numeric(values, errors="coerce").notna().sum())
    return hierarchy.aggregate_values(values, statistic)


def _sample_feature_trajectory(
    loader,
    users,
    feature,
    statistics,
    chain,
    start_date,
    end_date,
    time_reference,
    loader_kwargs,
    min_coverage,
    return_coverage,
    cycle_intervals,
    unresolved_coverage,
):
    """Build primary windows for one timestamped provider feature."""
    outputs = [f"{feature}_{statistic}" for statistic in statistics]
    frames = []
    resolution = chain[0]
    is_cycle = resolution in cycle.CYCLE_TYPES
    available_times = getattr(
        loader,
        "provider_feature_time_columns",
        (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL),
    )
    time_col = hierarchy.resolve_time_col(
        available_times, time_reference, chain, feature
    )
    interval_users = (
        set(cycle_intervals.index.get_level_values(0))
        if cycle_intervals is not None
        else set()
    )
    for user in users:
        user_intervals = None
        load_start, load_end = start_date, end_date
        if cycle_intervals is not None:
            if user not in interval_users:
                continue
            user_intervals = cycle_intervals.xs(user)
            valid = user_intervals[user_intervals["interval_valid"]]
            if valid.empty:
                continue
            load_start = valid["start_time"].min() - pd.Timedelta(days=1)
            load_end = valid["end_time"].max() + pd.Timedelta(days=1)
        elif start_date is not None or end_date is not None:
            padding = pd.Timedelta(days=2)
            load_start = (
                None if start_date is None else pd.Timestamp(start_date) - padding
            )
            load_end = None if end_date is None else pd.Timestamp(end_date) + padding
        samples = schema.prepare_provider_feature_samples(
            loader.load_provider_feature(
                feature,
                user,
                load_start,
                load_end,
                **loader_kwargs,
            ),
            feature,
        )
        samples = samples[samples[time_col].notna()].sort_values(
            time_col, kind="stable"
        )
        cadence = time_utils.infer_cadence(
            samples[time_col], min_gaps=3, min_cluster_fraction=0.5
        )
        if is_cycle:
            windows = user_intervals
        else:
            bounds = hierarchy.observed_bounds(
                samples, time_col, start_date, end_date, resolution
            )
            if bounds is None:
                continue
            start, end = bounds
            samples = samples[
                samples[time_col].ge(start) & samples[time_col].lt(end)
            ]
            windows = hierarchy.windows_for_resolution(
                [user], resolution, {user: bounds}
            ).xs(user)
        timestamps = pd.DatetimeIndex(samples[time_col])
        records = []
        for label, window in windows.iterrows():
            values = {name: np.nan for name in outputs}
            coverage = np.nan
            if bool(window.get("interval_valid", True)):
                left = timestamps.searchsorted(window["start_time"], side="left")
                right = timestamps.searchsorted(window["end_time"], side="left")
                rows = samples.iloc[left:right]
                valid_count = rows.loc[rows[feature].notna(), time_col].nunique()
                coverage = time_utils.sample_coverage(
                    valid_count,
                    window["start_time"],
                    window["end_time"],
                    cadence,
                )
                accepted = min_coverage is None or (
                    pd.notna(coverage) and coverage >= min_coverage
                )
                if accepted:
                    values = {
                        f"{feature}_{statistic}": _summarize(
                            rows[feature], statistic
                        )
                        for statistic in statistics
                    }
                elif pd.isna(coverage):
                    unresolved_coverage.add((user, feature))
            records.append(
                {
                    schema.USER_COL: user,
                    "window_start": label,
                    "period_start": window["start_time"],
                    "period_end": window["end_time"],
                    **values,
                    **(
                        {f"{name}__coverage": coverage for name in outputs}
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
        else hierarchy.empty_trajectory(outputs, return_coverage)
    )


def _daily_feature_trajectory(
    loader,
    users,
    feature,
    start_date,
    end_date,
    loader_kwargs,
    min_coverage,
    return_coverage,
):
    """Build a complete native-day trajectory, retaining missing dates."""
    frames = []
    numeric = _FEATURE_DEFINITIONS[feature].numeric
    for user in users:
        summary = schema.prepare_provider_feature_summary(
            loader.load_provider_feature(
                feature,
                user,
                start_date,
                end_date,
                **loader_kwargs,
            ),
            feature,
            numeric=numeric,
        )
        start = (
            pd.Timestamp(start_date).normalize()
            if start_date is not None
            else summary[schema.CALENDAR_DATE_COL].min()
        )
        end = (
            pd.Timestamp(end_date).normalize()
            if end_date is not None
            else summary[schema.CALENDAR_DATE_COL].max() + pd.Timedelta(days=1)
            if not summary.empty
            else pd.NaT
        )
        if pd.isna(start) or pd.isna(end):
            continue
        if end <= start:
            raise ValueError("end_date must be after start_date")
        values = summary.set_index(schema.CALENDAR_DATE_COL)[feature]
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
                    feature: value,
                    **(
                        {f"{feature}__coverage": coverage}
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
        else hierarchy.empty_trajectory([feature], return_coverage)
    )


def _nearest_value(times, values, target, direction, tolerance):
    """Return the first directional sample near a sleep-cycle boundary."""
    side = "right" if direction == "backward" else "left"
    position = times.searchsorted(target, side=side)
    position = position - 1 if direction == "backward" else position
    if position < 0 or position >= len(times):
        return np.nan
    distance = (
        target - times[position]
        if direction == "backward"
        else times[position] - target
    )
    return values[position] if pd.Timedelta(0) <= distance <= tolerance else np.nan


def _body_battery_sleep_trajectory(
    loader,
    users,
    features,
    loader_kwargs,
    min_coverage,
    return_coverage,
    intervals,
):
    """Derive Body Battery features from already selected night intervals.

    ``get_provider_features`` applies the requested date bounds when creating
    ``intervals``. This helper loads only the Body Battery range needed by
    those physical intervals, including a five-minute boundary tolerance.
    """
    frames = []
    interval_users = (
        set(intervals.index.get_level_values(0)) if not intervals.empty else set()
    )
    tolerance = pd.Timedelta(minutes=5)
    for user in users:
        if user not in interval_users:
            continue
        user_intervals = intervals.xs(user)
        valid = user_intervals[user_intervals["interval_valid"]]
        if valid.empty:
            continue
        samples = schema.prepare_provider_feature_samples(
            loader.load_provider_feature(
                GARMIN_BODY_BATTERY,
                user,
                valid["start_time"].min() - tolerance,
                valid["end_time"].max() + tolerance,
                **loader_kwargs,
            ),
            GARMIN_BODY_BATTERY,
        )
        samples = samples.dropna(
            subset=[schema.LOCAL_TIME_COL, GARMIN_BODY_BATTERY]
        ).sort_values(schema.LOCAL_TIME_COL, kind="stable")
        times = pd.DatetimeIndex(samples[schema.LOCAL_TIME_COL])
        values = samples[GARMIN_BODY_BATTERY].to_numpy(dtype=float)
        records = []
        for label, window in user_intervals.iterrows():
            calculated = {name: np.nan for name in features}
            coverage = {name: 0.0 for name in features}
            if bool(window.get("interval_valid", True)):
                start_value = _nearest_value(
                    times, values, window["start_time"], "backward", tolerance
                )
                getup_value = _nearest_value(
                    times, values, window["end_time"], "forward", tolerance
                )
                all_values = {
                    GARMIN_BODY_BATTERY_BEDTIME: start_value,
                    GARMIN_BODY_BATTERY_GETUP: getup_value,
                    GARMIN_BODY_BATTERY_SLEEP_GAIN: (
                        getup_value - start_value
                        if pd.notna(start_value) and pd.notna(getup_value)
                        else np.nan
                    ),
                    GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION: (
                        max((getup_value - start_value) / (100 - start_value), 0)
                        if pd.notna(start_value)
                        and pd.notna(getup_value)
                        and start_value < 100
                        else np.nan
                    ),
                }
                for feature in features:
                    coverage[feature] = float(pd.notna(all_values[feature]))
                    if min_coverage is None or coverage[feature] >= min_coverage:
                        calculated[feature] = all_values[feature]
            records.append(
                {
                    schema.USER_COL: user,
                    "window_start": label,
                    "period_start": window["start_time"],
                    "period_end": window["end_time"],
                    **calculated,
                    **(
                        {
                            f"{feature}__coverage": coverage[feature]
                            for feature in features
                        }
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
        else hierarchy.empty_trajectory(features, return_coverage)
    )


def get_provider_features(
    loader: BaseLoader,
    feature: Union[str, list, tuple],
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    resolution: Union[str, pd.Timedelta, list, tuple, None] = None,
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    aggregation=None,
    min_coverage=None,
    cycle_kwargs: dict | None = None,
    min_valid_children=None,
) -> pd.DataFrame:
    """Extract provider-native features without equating vendors' metrics.

    Parameters
    ----------
    loader : BaseLoader
        Loader declaring the requested feature in ``provider_features`` and
        returning canonical rows from ``load_provider_feature``.
    feature : str or sequence of str
        Names reported by :func:`get_available_provider_features`. Features in
        one call must share a native grain. This argument is required because
        a loader can expose unrelated sample, daily, nightly, numeric, and
        categorical provider features that cannot form one implicit result.
    statistic : str, sequence of str, or None
        Primary reduction(s) for timestamped streams. ``None`` means ``mean``.
        Native summaries reject this argument because they are already values.
    user_id : str, sequence of str, or None
        Participant selection; ``"all"`` and ``None`` select every loader user.
    start_date, end_date : date-like or None
        Half-open sample or native-label range, ``[start_date, end_date)``.
    resolution : str, pandas.Timedelta, sequence, or None
        Primary window or increasing hierarchy. ``None`` selects daily windows
        for samples and the native grain for summaries. Native summaries may
        only remain native or aggregate to week, month, or year.
    loader_kwargs, cycle_kwargs : dict or None
        Extra arguments for the loader or cycle interval extraction.
    time_reference : {"auto", "local", "utc"}
        Clock used by timestamped streams. Native summaries use local labels.
    aggregation : str, sequence, or None
        Parent-level operation(s); valid only for a hierarchy.
    min_coverage : float, sequence, or None
        Minimum valid sample/day/child fraction at each requested level.
    min_valid_children : int, sequence, or None
        Minimum valid children at each parent level.

    Returns
    -------
    pandas.DataFrame
        Indexed by participant and time/date, with one ``__coverage`` column
        per output feature. Hierarchical column names encode resolutions and
        operations.
    """
    features = _requested_features(loader, feature)
    users = loader.resolve_user_ids(user_id)
    loader_kwargs = {} if loader_kwargs is None else dict(loader_kwargs)
    cycle_kwargs = {} if cycle_kwargs is None else dict(cycle_kwargs)
    definition = _FEATURE_DEFINITIONS[features[0]]
    grain = definition.grain
    if resolution is None:
        resolution = "daily" if grain == "samples" else grain
    if grain == "samples":
        statistics = _sample_statistics(statistic)
        outputs = [
            f"{name}_{operation}"
            for name in features
            for operation in statistics
        ]
    else:
        if statistic is not None:
            raise ValueError("statistic applies only to timestamped provider features")
        outputs = list(features)
    if grain != "samples" and time_reference == "utc":
        raise ValueError("native provider summaries use local calendar labels")
    if any(not _FEATURE_DEFINITIONS[name].numeric for name in features):
        numeric = [name for name in features if _FEATURE_DEFINITIONS[name].numeric]
        if numeric:
            raise ValueError(
                "numeric and categorical provider features must be requested "
                "separately"
            )

    chain, hierarchical = _prepare_chain(resolution, grain)
    if not hierarchical and aggregation is not None:
        raise ValueError("aggregation requires a resolution hierarchy")
    if not hierarchical and min_valid_children is not None:
        raise ValueError("min_valid_children requires a resolution hierarchy")
    if hierarchical and any(
        not _FEATURE_DEFINITIONS[name].numeric for name in features
    ):
        raise ValueError("categorical provider features cannot be aggregated")
    aggregation_levels = hierarchy.normalize_aggregation_levels(
        aggregation, len(chain) - 1
    )
    coverage_thresholds = hierarchy.normalize_coverage_thresholds(
        min_coverage, len(chain)
    )
    child_thresholds = hierarchy.normalize_valid_child_thresholds(
        min_valid_children, len(chain)
    )
    hierarchy.warn_if_wide(outputs, aggregation_levels)

    cycle_target = next((value for value in chain if value in cycle.CYCLE_TYPES), None)
    intervals = (
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
    unresolved_coverage = set()
    if grain == "samples":
        frames = [
            _sample_feature_trajectory(
                loader,
                users,
                name,
                statistics,
                chain,
                start_date,
                end_date,
                time_reference,
                loader_kwargs,
                coverage_thresholds[0],
                include_coverage,
                intervals,
                unresolved_coverage,
            )
            for name in features
        ]
        trajectory = hierarchy.combine_trajectories(
            frames, outputs, include_coverage
        )
    elif grain == "daily":
        frames = [
            _daily_feature_trajectory(
                loader,
                users,
                name,
                start_date,
                end_date,
                loader_kwargs,
                coverage_thresholds[0],
                include_coverage,
            )
            for name in features
        ]
        trajectory = hierarchy.combine_trajectories(
            frames, outputs, include_coverage
        )
    else:
        trajectory = _body_battery_sleep_trajectory(
            loader,
            users,
            features,
            loader_kwargs,
            coverage_thresholds[0],
            include_coverage,
            intervals,
        )

    if coverage_thresholds[0] is not None and unresolved_coverage:
        warnings.warn(
            "Primary provider-feature coverage could not be determined for "
            f"{len(unresolved_coverage)} user/feature stream(s); values requiring "
            "a primary coverage threshold were excluded.",
            UserWarning,
            stacklevel=2,
        )

    if hierarchical:
        trajectory = trajectory.rename(
            columns={name: f"{name}__{chain[0]}" for name in outputs}
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
                intervals if target == cycle_target else None,
                prefer_explicit_bounds=grain == "daily",
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
            else hierarchy.resolve_time_col(
                getattr(
                    loader,
                    "provider_feature_time_columns",
                    (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL),
                ),
                time_reference,
                chain,
                "provider feature",
            )
        )
        return trajectory.rename_axis(index={"window_start": index_name})

    columns = [
        column
        for name in outputs
        for column in (name, f"{name}__coverage")
    ]
    result = trajectory[columns].copy()
    if (
        grain == "samples"
        and chain[0] not in hierarchy.CALENDAR_RESOLUTIONS
        and chain[0] not in cycle.CYCLE_TYPES
    ):
        index_name = hierarchy.resolve_time_col(
            getattr(
                loader,
                "provider_feature_time_columns",
                (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL),
            ),
            time_reference,
            chain,
            "provider feature",
        )
    else:
        index_name = schema.CALENDAR_DATE_COL
    result = result.rename_axis(index={"window_start": index_name})
    return result


__all__ = [
    "FITBIT_STRESS_EXERTION_POINTS",
    "FITBIT_STRESS_MANAGEMENT_SCORE",
    "FITBIT_STRESS_MAX_EXERTION_POINTS",
    "FITBIT_STRESS_MAX_RESPONSIVENESS_POINTS",
    "FITBIT_STRESS_MAX_SLEEP_POINTS",
    "FITBIT_STRESS_RESPONSIVENESS_POINTS",
    "FITBIT_STRESS_SLEEP_POINTS",
    "GARMIN_BODY_BATTERY",
    "GARMIN_BODY_BATTERY_BEDTIME",
    "GARMIN_BODY_BATTERY_GETUP",
    "GARMIN_BODY_BATTERY_SLEEP_GAIN",
    "GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION",
    "GARMIN_DAILY_ACTIVITY_STRESS_DURATION_SECONDS",
    "GARMIN_DAILY_AVERAGE_STRESS",
    "GARMIN_DAILY_HIGH_STRESS_DURATION_SECONDS",
    "GARMIN_DAILY_LOW_STRESS_DURATION_SECONDS",
    "GARMIN_DAILY_MAX_STRESS",
    "GARMIN_DAILY_MEDIUM_STRESS_DURATION_SECONDS",
    "GARMIN_DAILY_REST_STRESS_DURATION_SECONDS",
    "GARMIN_DAILY_STRESS_DURATION_SECONDS",
    "GARMIN_DAILY_STRESS_QUALIFIER",
    "GARMIN_STRESS_LEVEL",
    "OURA_READINESS_ACTIVITY_BALANCE",
    "OURA_READINESS_BODY_TEMPERATURE",
    "OURA_READINESS_HRV_BALANCE",
    "OURA_READINESS_PREVIOUS_DAY_ACTIVITY",
    "OURA_READINESS_PREVIOUS_NIGHT",
    "OURA_READINESS_RECOVERY_INDEX",
    "OURA_READINESS_RESTING_HEART_RATE",
    "OURA_READINESS_SCORE",
    "OURA_READINESS_SLEEP_BALANCE",
    "OURA_READINESS_TEMPERATURE_DEVIATION",
    "OURA_READINESS_TEMPERATURE_TREND_DEVIATION",
    "PROVIDER_FEATURES",
    "get_available_provider_features",
    "get_provider_features",
]
