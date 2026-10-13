"""Analysis utilities for canonical wearable feature trajectories.

The functions in this module operate on the ``(user_id, time)`` DataFrames
returned by pywearable feature modules.  Imputation is explicit, leaves
coverage unchanged, and records which values it fills.  Trend estimates are
computed independently for each participant and use historical observations
only by default.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from . import hierarchy, schema
from .utils import time as time_utils


_TREND_METHODS = {"ma", "ema", "cumulative_mean"}
_HISTORICAL_IMPUTATION_METHODS = {*_TREND_METHODS, "rolling_median"}
_WHOLE_TRAJECTORY_METHODS = {
    "user_mean",
    "user_median",
    "population_mean",
    "population_median",
}
_METHOD_ALIASES = {
    "moving_average": "ma",
    "rolling_mean": "ma",
    "mean_rolling": "ma",
    "moving_median": "rolling_median",
    "median_rolling": "rolling_median",
    "exponential_moving_average": "ema",
    "expanding_mean": "cumulative_mean",
}
_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def _normalize_method(method: str, allowed) -> str:
    """Normalize one public method name and validate it for its caller.

    Imputation and trend estimation intentionally expose different method
    sets.  Keeping alias handling here gives both APIs one spelling and one
    validation path without making the estimators themselves parse user input.
    """
    if not isinstance(method, str):
        raise ValueError("method must be a string")
    normalized = _METHOD_ALIASES.get(method.lower(), method.lower())
    if normalized not in allowed:
        raise ValueError(f"method must be one among {sorted(allowed)}")
    return normalized


def _trajectory_features(frame: pd.DataFrame, features) -> list[str]:
    """Validate the canonical trajectory shape and select feature columns.

    When ``features`` is omitted, feature columns are inferred from matching
    ``<feature>__coverage`` columns.  This excludes period bounds and derived
    metadata without maintaining a domain-specific list of feature names.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("trajectory must be a pandas DataFrame")
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels < 2:
        raise ValueError("trajectory must use a (user_id, time) index")
    if schema.USER_COL not in frame.index.names:
        raise ValueError("trajectory index must contain a user_id level")
    time_levels = [
        name
        for name in frame.index.names
        if name not in {schema.USER_COL, schema.TIMEZONE_COL}
    ]
    if len(time_levels) != 1:
        raise ValueError(
            "trajectory index must contain one time level and may contain "
            "a timezone level"
        )
    if not frame.index.is_unique:
        raise ValueError("trajectory index must contain unique participant/time rows")

    if features is None:
        selected = [
            column[: -len("__coverage")]
            for column in frame.columns
            if isinstance(column, str)
            and column.endswith("__coverage")
            and column[: -len("__coverage")] in frame.columns
        ]
    elif isinstance(features, str):
        selected = [features]
    else:
        selected = list(features)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("features must contain at least one trajectory feature")

    missing = [feature for feature in selected if feature not in frame]
    if missing:
        raise ValueError(f"trajectory is missing features: {missing}")
    missing_coverage = [
        feature
        for feature in selected
        if f"{feature}__coverage" not in frame.columns
    ]
    if missing_coverage:
        raise ValueError(
            "trajectory is missing coverage columns for: "
            f"{missing_coverage}"
        )
    return selected


def _time_values(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return validated timestamps from a canonical trajectory index."""
    time_level = next(
        name
        for name in frame.index.names
        if name not in {schema.USER_COL, schema.TIMEZONE_COL}
    )
    values = pd.to_datetime(
        frame.index.get_level_values(time_level), errors="coerce"
    )
    if values.isna().any():
        raise ValueError("trajectory time index must contain valid timestamps")
    return pd.DatetimeIndex(values)


def _ordered_participant_positions(frame: pd.DataFrame):
    """Yield each participant's row positions in chronological order.

    Calculations must be ordered and isolated by participant, while returned
    DataFrames must preserve the caller's original row order.  Positional
    indices let the estimators satisfy both requirements.
    """
    users = frame.index.get_level_values(schema.USER_COL)
    times = _time_values(frame)
    order = pd.DataFrame(
        {
            "__position": np.arange(len(frame)),
            schema.USER_COL: users,
            "__time": times,
        }
    )
    for user, rows in order.groupby(schema.USER_COL, sort=False, dropna=False):
        rows = rows.sort_values("__time", kind="stable")
        yield user, rows["__position"].to_numpy(), pd.DatetimeIndex(rows["__time"])


def _known_clock_feature(feature: str) -> bool:
    """Return whether a canonical feature name represents a clock time."""
    # Imports stay local so importing trajectory does not eagerly load domains.
    from .activity import ACTIVITY_STATISTIC_PEAK_60MIN_START
    from .sleep import SLEEP_TEMPORAL_STATISTICS

    names = {*SLEEP_TEMPORAL_STATISTICS, ACTIVITY_STATISTIC_PEAK_60MIN_START}
    return any(feature == name or feature.startswith(f"{name}__") for name in names)


def _is_clock_series(values: pd.Series) -> bool:
    """Infer clock-valued data from datetime-like values."""
    if is_datetime64_any_dtype(values.dtype):
        return True
    observed = values.dropna()
    return not observed.empty and isinstance(
        observed.iloc[0], (datetime.datetime, datetime.time, pd.Timestamp)
    )


def _clock_features(frame, features, temporal_features) -> set[str]:
    """Resolve clock features from overrides, canonical names, and dtypes."""
    explicit = set()
    if temporal_features is not None:
        explicit = (
            {temporal_features}
            if isinstance(temporal_features, str)
            else set(temporal_features)
        )
        unknown = explicit.difference(features)
        if unknown:
            raise ValueError(
                f"temporal_features contains unselected features: {sorted(unknown)}"
            )
    return {
        feature
        for feature in features
        if feature in explicit
        or _known_clock_feature(feature)
        or _is_clock_series(frame[feature])
    }


def _numeric_values(values: pd.Series, feature: str) -> pd.Series:
    """Coerce one selected feature to float and reject mixed value types."""
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"feature {feature!r} must be numeric or temporal")
    return pd.Series(numeric.to_numpy(dtype=float), index=values.index)


def _clock_series(values: pd.Series) -> pd.Series:
    """Represent clock values as floating-point seconds since midnight."""
    return values.map(time_utils.clock_seconds).astype(float)


def _clock_representation(values: pd.Series) -> str:
    """Record whether restored clock values should be times or timestamps."""
    observed = values.dropna()
    if not observed.empty and isinstance(observed.iloc[0], datetime.time):
        return "time"
    return "timestamp"


def _previous_day_sleep_clock_feature(feature: str) -> bool:
    """Return whether a sleep clock value belongs to the preceding evening."""
    from .sleep import SLEEP_STATISTIC_BEDTIME, SLEEP_STATISTIC_SLEEP_ONSET

    return any(
        feature == name or feature.startswith(f"{name}__")
        for name in (SLEEP_STATISTIC_BEDTIME, SLEEP_STATISTIC_SLEEP_ONSET)
    )


def _restore_clock_value(seconds, target_time, feature, representation):
    """Restore seconds-of-day on the relevant trajectory calendar date.

    Sleep trajectories are labelled by wake date, so evening values belong to
    the preceding civil date.  Other timestamps are attached directly to the
    trajectory row's date.
    """
    if pd.isna(seconds):
        return pd.NaT if representation == "timestamp" else None
    seconds = float(seconds) % 86_400
    if representation == "time":
        return time_utils.clock_time(seconds)
    date = pd.Timestamp(target_time).normalize()
    if _previous_day_sleep_clock_feature(feature) and seconds >= 12 * 3_600:
        date -= pd.Timedelta(days=1)
    return date + pd.to_timedelta(seconds, unit="s")


def _is_integer_window(window) -> bool:
    """Return whether a window denotes a number of trajectory rows."""
    return isinstance(window, (int, np.integer)) and not isinstance(window, bool)


def _validate_history_parameters(method, window, min_periods):
    """Validate window and support settings for a historical estimator."""
    if not isinstance(min_periods, (int, np.integer)) or min_periods < 1:
        raise ValueError("min_periods must be a positive integer")
    if method in {"ma", "rolling_median"}:
        if _is_integer_window(window):
            if window < 1:
                raise ValueError("window must be positive")
        else:
            try:
                duration = pd.Timedelta(window)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "window must be a positive integer or timedelta-like value"
                ) from error
            if duration <= pd.Timedelta(0):
                raise ValueError("window must be positive")
    elif method == "ema":
        if not isinstance(window, (int, float, np.integer, np.floating)) or window <= 0:
            raise ValueError("EMA span must be a positive number")


def _rolling_operation(values, times, window, min_periods, operation, include_current):
    """Apply a row-count or elapsed-time rolling operation."""
    if _is_integer_window(window):
        rolling = values.rolling(window=int(window), min_periods=min_periods)
        result = getattr(rolling, operation)()
        return result if include_current else result.shift(1)

    timed = pd.Series(values.to_numpy(), index=times)
    rolling = timed.rolling(
        window=window,
        min_periods=min_periods,
        closed=None if include_current else "left",
    )
    return pd.Series(getattr(rolling, operation)().to_numpy(), index=values.index)


def _historical_operation(
    values,
    times,
    method,
    window,
    min_periods,
    operation,
    include_current=False,
):
    """Compute one historical numeric estimator on an ordered participant."""
    if method in {"ma", "rolling_median"}:
        return _rolling_operation(
            values,
            times,
            window,
            min_periods,
            "median" if method == "rolling_median" else operation,
            include_current,
        )
    if method == "ema":
        exponential = values.ewm(
            span=window,
            min_periods=min_periods,
            adjust=False,
            ignore_na=True,
        )
        result = (
            exponential.std(bias=False)
            if operation == "std"
            else exponential.mean()
        )
        return result if include_current else result.shift(1)
    expanding = values.expanding(min_periods=min_periods)
    result = getattr(expanding, operation)()
    return result if include_current else result.shift(1)


def _historical_support(values, times, method, window, include_current=False):
    """Count observed inputs available to each historical estimate."""
    observed = values.notna().astype(float)
    if method in {"ma", "rolling_median"}:
        if _is_integer_window(window):
            result = observed.rolling(window=int(window), min_periods=1).sum()
            return result if include_current else result.shift(1, fill_value=0)
        timed = pd.Series(observed.to_numpy(), index=times)
        result = timed.rolling(
            window=window,
            min_periods=0,
            closed=None if include_current else "left",
        ).sum()
        return pd.Series(result.to_numpy(), index=values.index)
    result = observed.cumsum()
    return result if include_current else result.shift(1, fill_value=0)


def _circular_historical_center(
    seconds, times, method, window, min_periods, include_current=False
):
    """Compute a historical center for clock-valued observations."""
    if method == "rolling_median":
        # Use the same noon-to-noon ordering as hierarchical clock medians so
        # late-evening and early-morning sleep times remain adjacent.
        ordered = seconds.where(seconds >= 12 * 3_600, seconds + 86_400)
        return _rolling_operation(
            ordered,
            times,
            window,
            min_periods,
            "median",
            include_current,
        ) % 86_400
    angles = seconds / 86_400 * 2 * np.pi
    sine = np.sin(angles).where(seconds.notna())
    cosine = np.cos(angles).where(seconds.notna())
    mean_sine = _historical_operation(
        sine, times, method, window, min_periods, "mean", include_current
    )
    mean_cosine = _historical_operation(
        cosine, times, method, window, min_periods, "mean", include_current
    )
    resultant = np.hypot(mean_sine, mean_cosine)
    angle = np.arctan2(mean_sine, mean_cosine) % (2 * np.pi)
    center = angle / (2 * np.pi) * 86_400
    return center.where(resultant >= 1e-12)


def _circular_historical_std(
    seconds, times, method, window, min_periods, include_current=False
):
    """Compute historical circular spread in minutes."""
    angles = seconds / 86_400 * 2 * np.pi
    sine = np.sin(angles).where(seconds.notna())
    cosine = np.cos(angles).where(seconds.notna())
    mean_sine = _historical_operation(
        sine, times, method, window, min_periods, "mean", include_current
    )
    mean_cosine = _historical_operation(
        cosine, times, method, window, min_periods, "mean", include_current
    )
    resultant = np.hypot(mean_sine, mean_cosine).clip(upper=1)
    spread = np.sqrt(-2 * np.log(resultant)) * 1_440 / (2 * np.pi)
    return spread.where(resultant >= 1e-12)


def _whole_center(values, temporal, method):
    """Compute a full-user or full-population mean or median."""
    observed = values.dropna()
    if observed.empty:
        return np.nan
    statistic = method.rsplit("_", maxsplit=1)[-1]
    if not temporal:
        return getattr(observed, statistic)()
    if statistic == "median":
        ordered = observed.where(observed >= 12 * 3_600, observed + 86_400)
        return float(ordered.median() % 86_400)
    return time_utils.circular_mean_seconds(observed)


def _source_values(frame, feature, temporal, positions=None, include_imputed=False):
    """Return analysis-ready values, optionally masking prior imputations."""
    values = frame[feature] if positions is None else frame[feature].iloc[positions]
    source = _clock_series(values) if temporal else _numeric_values(values, feature)
    flag = f"{feature}__is_imputed"
    if not include_imputed and flag in frame:
        imputed = frame[flag] if positions is None else frame[flag].iloc[positions]
        source = source.mask(imputed.fillna(False).astype(bool).to_numpy())
    return source.reset_index(drop=True)


def _historical_estimates(
    frame,
    feature,
    temporal,
    method,
    window,
    min_periods,
    *,
    operation="mean",
    include_current=False,
    include_imputed=False,
):
    """Calculate participant-isolated historical estimates and support."""
    estimates = pd.Series(np.nan, index=np.arange(len(frame)), dtype=float)
    supports = pd.Series(0, index=np.arange(len(frame)), dtype="int64")
    for _, positions, times in _ordered_participant_positions(frame):
        values = _source_values(
            frame,
            feature,
            temporal,
            positions,
            include_imputed=include_imputed,
        )
        if temporal:
            calculated = (
                _circular_historical_std(
                    values, times, method, window, min_periods, include_current
                )
                if operation == "std"
                else _circular_historical_center(
                    values, times, method, window, min_periods, include_current
                )
            )
        else:
            calculated = _historical_operation(
                values,
                times,
                method,
                window,
                min_periods,
                operation,
                include_current,
            )
        support = _historical_support(
            values, times, method, window, include_current
        )
        estimates.iloc[positions] = calculated.to_numpy()
        supports.iloc[positions] = support.fillna(0).astype(int).to_numpy()
    return estimates, supports


def _whole_trajectory_estimates(frame, feature, temporal, method):
    """Broadcast a full-user or full-population center to trajectory rows."""
    source = _source_values(frame, feature, temporal)
    estimates = pd.Series(np.nan, index=np.arange(len(frame)), dtype=float)
    if method.startswith("population"):
        estimates[:] = _whole_center(source, temporal, method)
        return estimates
    for _, positions, _ in _ordered_participant_positions(frame):
        estimates.iloc[positions] = _whole_center(
            source.iloc[positions], temporal, method
        )
    return estimates


def impute_trajectory(
    trajectory: pd.DataFrame,
    features: str | Sequence[str] | None = None,
    *,
    method: str = "ma",
    window=7,
    min_periods: int = 3,
    fallback: str | None = None,
    temporal_features: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fill missing values in one or more canonical feature trajectories.

    Parameters
    ----------
    trajectory
        DataFrame indexed by ``(user_id, time)``. A timezone index level is
        optional. Every selected feature must have a matching
        ``<feature>__coverage`` column.
    features
        Feature name or names to impute. If omitted, all feature/coverage
        column pairs are selected.
    method
        Estimator used for missing values. Historical choices are ``"ma"``
        (trailing moving mean), ``"rolling_median"``, ``"ema"``, and
        ``"cumulative_mean"``. Retrospective choices are ``"user_mean"``,
        ``"user_median"``, ``"population_mean"``, and
        ``"population_median"``.
    window
        For ``"ma"`` and ``"rolling_median"``, either the number of preceding
        rows or an elapsed-time window such as ``"30D"``. For ``"ema"``, the
        exponential span. Ignored by the other methods.
    min_periods
        Minimum number of observed values required by a historical estimator.
    fallback
        Optional ``"user_mean"``, ``"user_median"``, ``"population_mean"``,
        or ``"population_median"`` used when a historical estimator has
        insufficient support. No fallback is applied by default.
    temporal_features
        Additional selected features to interpret as clock times. Canonical
        sleep times, peak-activity start times, and datetime-like columns are
        recognized automatically.

    Returns
    -------
    pandas.DataFrame
        A copy with fillable values replaced and one boolean
        ``<feature>__is_imputed`` column per selected feature. Input rows and
        coverage columns are unchanged.

    Notes
    -----
    Historical estimates are participant-specific, use observations preceding
    the missing row, and never use values filled by this or an earlier call.
    Full-user methods compute the mean or median from all observed values for
    that participant in the complete supplied trajectory. Population methods
    compute one mean or median from all observed values across every
    participant and date in the supplied trajectory, then use it for every
    participant; no loader or external cohort data is consulted. Clock
    features are aggregated by time of day and restored on the missing row's
    relevant calendar date.
    """
    selected = _trajectory_features(trajectory, features)
    clock = _clock_features(trajectory, selected, temporal_features)
    method = _normalize_method(
        method,
        _HISTORICAL_IMPUTATION_METHODS | _WHOLE_TRAJECTORY_METHODS,
    )
    if method in _HISTORICAL_IMPUTATION_METHODS:
        _validate_history_parameters(method, window, min_periods)
    if fallback is not None:
        fallback = _normalize_method(fallback, _WHOLE_TRAJECTORY_METHODS)
        if method not in _HISTORICAL_IMPUTATION_METHODS:
            raise ValueError("fallback applies only to historical imputation methods")

    result = trajectory.copy()
    times = _time_values(result)
    for feature in selected:
        temporal = feature in clock
        if method in _HISTORICAL_IMPUTATION_METHODS:
            estimates, _ = _historical_estimates(
                result,
                feature,
                temporal,
                method,
                window,
                min_periods,
            )
        else:
            estimates = _whole_trajectory_estimates(
                result, feature, temporal, method
            )
        if fallback is not None:
            fallback_values = _whole_trajectory_estimates(
                result, feature, temporal, fallback
            )
            estimates = estimates.fillna(fallback_values)

        missing = result[feature].isna().to_numpy()
        fillable = missing & estimates.notna().to_numpy()
        filled = result[feature].copy()
        if temporal:
            representation = _clock_representation(result[feature])
            for position in np.flatnonzero(fillable):
                filled.iloc[position] = _restore_clock_value(
                    estimates.iloc[position],
                    times[position],
                    feature,
                    representation,
                )
        else:
            filled.iloc[np.flatnonzero(fillable)] = estimates[fillable].to_numpy()
        result[feature] = filled

        flag = f"{feature}__is_imputed"
        previous = (
            result[flag].fillna(False).astype(bool)
            if flag in result
            else pd.Series(False, index=result.index)
        )
        result[flag] = previous.to_numpy() | fillable
    return result


def _clock_output(seconds, spread, times, feature, representation):
    """Restore clock centers and symmetric circular range bounds."""
    center = []
    lower = []
    upper = []
    for value, width, target in zip(seconds, spread, times):
        restored = _restore_clock_value(value, target, feature, representation)
        center.append(restored)
        if pd.isna(value) or pd.isna(width):
            lower.append(pd.NaT if representation == "timestamp" else None)
            upper.append(pd.NaT if representation == "timestamp" else None)
        elif representation == "timestamp":
            lower.append(restored - pd.to_timedelta(width, unit="m"))
            upper.append(restored + pd.to_timedelta(width, unit="m"))
        else:
            lower.append(time_utils.clock_time(value - width * 60))
            upper.append(time_utils.clock_time(value + width * 60))
    return center, lower, upper


def compute_trajectory_trend(
    trajectory: pd.DataFrame,
    features: str | Sequence[str] | None = None,
    *,
    baseline_method: str = "ma",
    baseline_window=7,
    min_periods_baseline: int = 3,
    normal_range_method: str = "ma",
    normal_range_window=30,
    min_periods_normal_range: int = 20,
    std_multiplier: float = 1.0,
    include_current: bool = False,
    include_imputed: bool = False,
    temporal_features: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add participant-specific historical baselines and normal ranges.

    Parameters
    ----------
    trajectory
        Canonical ``(user_id, time)`` trajectory DataFrame. Every selected
        feature must have a matching ``<feature>__coverage`` column.
    features
        Feature name or names to analyze. If omitted, all feature/coverage
        column pairs are selected.
    baseline_method, normal_range_method
        Independently selected historical estimators: ``"ma"``, ``"ema"``,
        or ``"cumulative_mean"``. The baseline describes the expected value;
        the normal-range method independently determines its center and
        standard deviation.
    baseline_window, normal_range_window
        For ``"ma"``, either a number of preceding rows or an elapsed-time
        window. For ``"ema"``, the exponential span. Ignored by cumulative
        means.
    min_periods_baseline, min_periods_normal_range
        Minimum observed values required for the corresponding estimate.
    std_multiplier
        Non-negative number of standard deviations below and above the normal
        center used for range bounds.
    include_current
        Include the value on the current row in historical estimates. The
        default ``False`` makes every estimate strictly backward-looking.
    include_imputed
        Include values marked by ``<feature>__is_imputed`` as estimator input.
        They are excluded by default.
    temporal_features
        Additional selected features to interpret as clock times. Canonical
        temporal features and datetime-like columns are detected automatically.

    Returns
    -------
    pandas.DataFrame
        A copy of the input with these columns for every selected feature:
        ``__trend_baseline``, ``__trend_normal_center``,
        ``__trend_normal_std``, ``__trend_normal_lower``,
        ``__trend_normal_upper``, ``__trend_deviation``, and support counts for
        the baseline and normal range.

    Notes
    -----
    Participants are always analyzed independently. Numeric deviations and
    standard deviations retain the feature's units. Clock-valued centers and
    bounds remain clock values; their deviation and circular standard
    deviation are expressed in minutes.
    """
    selected = _trajectory_features(trajectory, features)
    clock = _clock_features(trajectory, selected, temporal_features)
    baseline_method = _normalize_method(baseline_method, _TREND_METHODS)
    normal_range_method = _normalize_method(normal_range_method, _TREND_METHODS)
    _validate_history_parameters(
        baseline_method, baseline_window, min_periods_baseline
    )
    _validate_history_parameters(
        normal_range_method, normal_range_window, min_periods_normal_range
    )
    if not np.isfinite(std_multiplier) or std_multiplier < 0:
        raise ValueError("std_multiplier must be a finite non-negative number")

    result = trajectory.copy()
    times = _time_values(result)
    for feature in selected:
        temporal = feature in clock
        baseline, baseline_support = _historical_estimates(
            result,
            feature,
            temporal,
            baseline_method,
            baseline_window,
            min_periods_baseline,
            include_current=include_current,
            include_imputed=include_imputed,
        )
        normal_center, normal_support = _historical_estimates(
            result,
            feature,
            temporal,
            normal_range_method,
            normal_range_window,
            min_periods_normal_range,
            include_current=include_current,
            include_imputed=include_imputed,
        )
        normal_std, _ = _historical_estimates(
            result,
            feature,
            temporal,
            normal_range_method,
            normal_range_window,
            min_periods_normal_range,
            operation="std",
            include_current=include_current,
            include_imputed=include_imputed,
        )
        source = _source_values(
            result,
            feature,
            temporal,
            include_imputed=include_imputed,
        )

        prefix = f"{feature}__trend"
        if temporal:
            representation = _clock_representation(result[feature])
            baseline_values = [
                _restore_clock_value(value, target, feature, representation)
                for value, target in zip(baseline, times)
            ]
            normal_values, lower, upper = _clock_output(
                normal_center,
                normal_std * std_multiplier,
                times,
                feature,
                representation,
            )
            deviation = ((source - baseline + 43_200) % 86_400 - 43_200) / 60
            result[f"{prefix}_baseline"] = baseline_values
            result[f"{prefix}_normal_center"] = normal_values
            result[f"{prefix}_normal_std"] = normal_std.to_numpy()
            result[f"{prefix}_normal_lower"] = lower
            result[f"{prefix}_normal_upper"] = upper
            result[f"{prefix}_deviation"] = deviation.to_numpy()
        else:
            result[f"{prefix}_baseline"] = baseline.to_numpy()
            result[f"{prefix}_normal_center"] = normal_center.to_numpy()
            result[f"{prefix}_normal_std"] = normal_std.to_numpy()
            result[f"{prefix}_normal_lower"] = (
                normal_center - std_multiplier * normal_std
            ).to_numpy()
            result[f"{prefix}_normal_upper"] = (
                normal_center + std_multiplier * normal_std
            ).to_numpy()
            result[f"{prefix}_deviation"] = (source - baseline).to_numpy()
        result[f"{prefix}_baseline_support"] = pd.array(
            baseline_support.to_numpy(), dtype="Int64"
        )
        result[f"{prefix}_normal_support"] = pd.array(
            normal_support.to_numpy(), dtype="Int64"
        )
    return result


def aggregate_trajectory_by_weekday(
    trajectory: pd.DataFrame,
    features: str | Sequence[str] | None = None,
    *,
    aggregation: str = "mean",
    include_imputed: bool = False,
    temporal_features: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate repeated calendar weekdays within each participant.

    This operation is intentionally separate from resolution hierarchies:
    Mondays across different weeks are a descriptive group, not one
    chronological parent window.

    Parameters
    ----------
    trajectory
        Canonical ``(user_id, time)`` trajectory with feature coverage columns.
        Daily, nightly, cycle, and subdaily trajectories are all accepted.
    features
        Feature name or names. By default, every value column paired with a
        ``__coverage`` column is selected.
    aggregation
        ``"mean"``, ``"median"``, ``"min"``, ``"max"``, ``"std"``, or a
        percentile such as ``"p90"``. Clock features use circular operations.
    include_imputed
        Include values marked by ``<feature>__is_imputed``. They are excluded
        by default.
    temporal_features
        Additional selected columns to interpret as clock-valued features.

    Returns
    -------
    pandas.DataFrame
        One row for every participant and Monday-through-Sunday label. Each
        feature is accompanied by ``__support`` and mean input ``__coverage``.
    """
    selected = _trajectory_features(trajectory, features)
    temporal = _clock_features(trajectory, selected, temporal_features)
    if (
        aggregation not in {"mean", "median", "min", "max", "std"}
        and not hierarchy.is_percentile(aggregation)
    ):
        raise ValueError("aggregation must be mean, median, min, max, std, or pXX")

    times = _time_values(trajectory)
    users = trajectory.index.get_level_values(schema.USER_COL)
    records = []
    for user in pd.Index(users).unique():
        user_positions = np.flatnonzero(np.asarray(users == user))
        user_weekdays = times[user_positions].weekday
        for weekday, label in enumerate(_WEEKDAY_NAMES):
            positions = user_positions[user_weekdays == weekday]
            record = {schema.USER_COL: user, "weekday": label}
            for feature in selected:
                values = trajectory[feature].iloc[positions].copy()
                flag = f"{feature}__is_imputed"
                if flag in trajectory and not include_imputed:
                    imputed = (
                        trajectory[flag]
                        .iloc[positions]
                        .fillna(False)
                        .astype(bool)
                    )
                    values = values.mask(imputed.to_numpy())
                record[feature] = hierarchy.aggregate_values(
                    values,
                    aggregation,
                    temporal=feature in temporal,
                )
                record[f"{feature}__support"] = int(values.notna().sum())
                record[f"{feature}__coverage"] = pd.to_numeric(
                    trajectory[f"{feature}__coverage"].iloc[positions],
                    errors="coerce",
                ).mean()
            records.append(record)

    columns = [
        column
        for feature in selected
        for column in (
            feature,
            f"{feature}__support",
            f"{feature}__coverage",
        )
    ]
    result = pd.DataFrame(records)
    result["weekday"] = pd.Categorical(
        result["weekday"], categories=_WEEKDAY_NAMES, ordered=True
    )
    return result.set_index([schema.USER_COL, "weekday"])[columns].sort_index()


__all__ = [
    "aggregate_trajectory_by_weekday",
    "compute_trajectory_trend",
    "impute_trajectory",
]
