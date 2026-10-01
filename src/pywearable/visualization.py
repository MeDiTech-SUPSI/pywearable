"""Matplotlib visualizations for wearable trajectories and monitoring data.

Most functions consume the canonical trajectory DataFrames returned by the
domain modules. Loader-aware composite views are also provided when a useful
figure must combine multiple streams, such as sleep stages, heart rate,
activity, or adherence. Every public function can display and save its figure
and returns its axes for further customization.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from pandas.api.types import is_datetime64_any_dtype

from . import (
    activity,
    adherence,
    heart_rate,
    hierarchy,
    hrv,
    provider_features,
    respiration,
    schema,
    sleep,
    trajectory as trajectory_analysis,
)
from .utils import time as time_utils

# Okabe-Ito colors, with neutral greys for missing and contextual layers.
BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#222222"
GREY = "#8A8A8A"
LIGHT_GREY = "#D9D9D9"
PALETTE = (BLUE, ORANGE, GREEN, PURPLE, LIGHT_BLUE, VERMILLION, BLACK)

SLEEP_STAGE_COLORS = {
    # Sleep-stage colors and opacity shared by sleep visualizations.
    schema.SLEEP_STAGE_AWAKE_STAGE_VALUE: "hotpink",
    schema.SLEEP_STAGE_REM_STAGE_VALUE: "darkmagenta",
    schema.SLEEP_STAGE_LIGHT_STAGE_VALUE: "royalblue",
    schema.SLEEP_STAGE_DEEP_STAGE_VALUE: "darkblue",
    schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: GREY,
    "asleep": PURPLE,
}

SLEEP_STAGE_ALPHA = {
    schema.SLEEP_STAGE_AWAKE_STAGE_VALUE: 0.35,
    schema.SLEEP_STAGE_REM_STAGE_VALUE: 0.9,
    schema.SLEEP_STAGE_LIGHT_STAGE_VALUE: 0.35,
    schema.SLEEP_STAGE_DEEP_STAGE_VALUE: 0.9,
    schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: 0.45,
    "asleep": 0.6,
}
SLEEP_HEART_RATE_STAGE_ALPHA = {
    schema.SLEEP_STAGE_AWAKE_STAGE_VALUE: 0.10,
    schema.SLEEP_STAGE_REM_STAGE_VALUE: 0.34,
    schema.SLEEP_STAGE_LIGHT_STAGE_VALUE: 0.28,
    schema.SLEEP_STAGE_DEEP_STAGE_VALUE: 0.28,
    schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: 0.20,
    "asleep": 0.18,
}
SLEEP_STAGE_LEVELS = {
    schema.SLEEP_STAGE_DEEP_STAGE_VALUE: (0, "deep"),
    schema.SLEEP_STAGE_LIGHT_STAGE_VALUE: (1, "light"),
    "asleep": (1, "light"),
    schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: (None, "unknown"),
    schema.SLEEP_STAGE_REM_STAGE_VALUE: (2, "rem"),
    schema.SLEEP_STAGE_AWAKE_STAGE_VALUE: (3, "wake"),
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

_KNOWN_CLOCK_FEATURES = {
    *sleep.SLEEP_TEMPORAL_STATISTICS,
    "activity_peak_60min_start",
}


def _finish_figure(fig, *, show, save_to):
    if save_to is not None:
        fig.savefig(Path(save_to), bbox_inches="tight")
    if show:
        plt.show()


def _style_axis(ax: Axes, *, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)


def _trajectory_time_level(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("trajectory must be a pandas DataFrame")
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("trajectory must use a (user_id, time) index")
    if schema.USER_COL not in frame.index.names:
        raise ValueError("trajectory index must contain a user_id level")
    levels = [
        name
        for name in frame.index.names
        if name not in {schema.USER_COL, schema.TIMEZONE_COL}
    ]
    if len(levels) != 1:
        raise ValueError(
            "trajectory index must contain one time level and may contain a "
            "timezone level"
        )
    return levels[0]


def _select_user(frame: pd.DataFrame, user_id=None) -> tuple[pd.DataFrame, object, str]:
    time_level = _trajectory_time_level(frame)
    users = pd.Index(frame.index.get_level_values(schema.USER_COL)).unique()
    if not len(users):
        raise ValueError("trajectory must contain at least one row")
    if user_id is None:
        if len(users) != 1:
            raise ValueError("user_id is required when a trajectory has multiple users")
        user_id = users[0]
    if user_id not in set(users):
        raise ValueError(f"user_id {user_id!r} is not present in the trajectory")

    selected = frame.loc[
        frame.index.get_level_values(schema.USER_COL) == user_id
    ].copy()
    times = pd.to_datetime(selected.index.get_level_values(time_level), errors="coerce")
    if times.isna().any():
        raise ValueError("trajectory time index must contain valid timestamps")
    selected.index = pd.DatetimeIndex(times, name=time_level)
    return selected.sort_index(kind="stable"), user_id, time_level


def _feature_list(frame: pd.DataFrame, features) -> list[str]:
    if isinstance(features, str):
        selected = [features]
    elif isinstance(features, Sequence):
        selected = list(features)
    else:
        raise ValueError("features must be a feature name or a sequence of names")
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("features must contain at least one feature")
    missing = [feature for feature in selected if feature not in frame.columns]
    if missing:
        raise ValueError(f"trajectory is missing features: {missing}")
    return selected


def _is_clock_series(values: pd.Series, feature: str, temporal=None) -> bool:
    if temporal is not None:
        return bool(temporal)
    base = feature.split("__", 1)[0]
    if base in _KNOWN_CLOCK_FEATURES:
        return True
    if is_datetime64_any_dtype(values.dtype):
        return True
    observed = values.dropna()
    return not observed.empty and isinstance(
        observed.iloc[0], (datetime.datetime, datetime.time, pd.Timestamp)
    )


def _clock_anchor(*series: pd.Series) -> float:
    available = [value.dropna() for value in series if value is not None]
    values = pd.concat(available) if available else pd.Series(dtype=float)
    if values.empty:
        return 12.0
    hours = values.map(time_utils.clock_seconds).to_numpy(dtype=float) / 3600
    center_seconds = time_utils.circular_mean_seconds(hours * 3600)
    if pd.isna(center_seconds):
        return 0.0
    center = center_seconds / 3600
    return (center - 12) % 24


def _clock_hours(values: pd.Series, anchor: float) -> pd.Series:
    return values.map(
        lambda value: time_utils.unwrap_clock_seconds(
            time_utils.clock_seconds(value), anchor * 3600
        )
        / 3600
    ).astype(float)


def _clock_formatter(value, _):
    if not np.isfinite(value):
        return ""
    minutes = int(round((value % 24) * 60)) % (24 * 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def _format_clock_axis(ax: Axes):
    ax.yaxis.set_major_formatter(FuncFormatter(_clock_formatter))


def _trajectory_axes(ax, show_coverage, figsize):
    if not show_coverage:
        if ax is None:
            fig, main_ax = plt.subplots(figsize=figsize)
        elif isinstance(ax, Axes):
            main_ax = ax
            fig = ax.figure
        else:
            raise ValueError("ax must be a Matplotlib Axes")
        return fig, main_ax, None

    if ax is None:
        fig, (main_ax, coverage_ax) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
            constrained_layout=True,
        )
    else:
        axes = list(np.asarray(ax, dtype=object).flat)
        if len(axes) != 2 or not all(isinstance(item, Axes) for item in axes):
            raise ValueError("show_coverage=True requires two axes")
        main_ax, coverage_ax = axes
        fig = main_ax.figure
        if coverage_ax.figure is not fig:
            raise ValueError("the trajectory and coverage axes must share a figure")
    return fig, main_ax, coverage_ax


def _bar_width(times: pd.DatetimeIndex, fraction=0.75) -> float:
    if len(times) < 2:
        return fraction
    differences = np.diff(np.sort(times.unique().asi8)) / 86_400_000_000_000
    positive = differences[differences > 0]
    return float(np.median(positive) * fraction) if len(positive) else fraction


def _plot_coverage_axis(ax, frame, features, colors):
    found = False
    for feature, color in zip(features, colors):
        column = f"{feature}__coverage"
        if column not in frame:
            continue
        coverage = pd.to_numeric(frame[column], errors="coerce")
        ax.step(
            frame.index,
            coverage,
            where="mid",
            color=color,
            linewidth=1.5,
            label=feature,
        )
        found = True
    if not found:
        raise ValueError("the selected features do not have coverage columns")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_yticks([0, 0.5, 1])
    if len(features) > 1:
        ax.legend(frameon=False, ncol=min(3, len(features)))
    _style_axis(ax)


def plot_trajectory(
    trajectory: pd.DataFrame,
    features: str | Sequence[str],
    *,
    user_id=None,
    kind: str = "line",
    temporal: bool | None = None,
    show_imputed: bool = True,
    show_coverage: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize=(11, 5),
    ax=None,
    show: bool = True,
    save_to=None,
):
    """Plot one participant's feature trajectory.

    Parameters
    ----------
    trajectory
        Canonical ``(user_id, time)`` trajectory DataFrame.
    features
        One feature name or several features that can share an axis.
    user_id
        Participant to draw. It may be omitted for a single-user trajectory.
    kind
        ``"line"`` or ``"bar"``. Bars support one feature at a time.
    temporal
        Override automatic clock-time detection. Clock-valued observations
        are unwrapped around midnight before plotting.
    show_imputed
        Draw values marked by ``<feature>__is_imputed``. They use a distinct
        marker or hatch and are never confused with observed values.
    show_coverage
        Add a lower panel from the corresponding ``__coverage`` columns.
    show, save_to
        Display the figure and/or save it to the supplied path.

    Returns
    -------
    matplotlib.axes.Axes or dict
        The trajectory axis, or ``{"trajectory", "coverage"}`` when the
        coverage panel is requested.

    Notes
    -----
    Missing values remain gaps and zero values remain visible observations.
    This function does not calculate features, imputation, or trends.
    """
    frame, _, time_level = _select_user(trajectory, user_id)
    selected = _feature_list(frame, features)
    if kind not in {"line", "bar"}:
        raise ValueError("kind must be 'line' or 'bar'")
    if kind == "bar" and len(selected) > 1:
        raise ValueError("bar trajectories support one feature at a time")

    clock_flags = [_is_clock_series(frame[name], name, temporal) for name in selected]
    if any(clock_flags) and not all(clock_flags):
        raise ValueError("numeric and clock-valued features cannot share one axis")
    is_clock = all(clock_flags)
    anchor = _clock_anchor(*(frame[name] for name in selected)) if is_clock else None

    fig, main_ax, coverage_ax = _trajectory_axes(ax, show_coverage, figsize)
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(selected))]
    width = _bar_width(frame.index)
    for feature, color in zip(selected, colors):
        values = (
            _clock_hours(frame[feature], anchor)
            if is_clock
            else pd.to_numeric(frame[feature], errors="coerce")
        )
        flag = f"{feature}__is_imputed"
        imputed = (
            frame[flag].fillna(False).astype(bool)
            if flag in frame
            else pd.Series(False, index=frame.index)
        )
        visible = values.mask(imputed) if not show_imputed else values
        if kind == "bar":
            observed = visible.notna() & ~imputed
            main_ax.bar(
                frame.index[observed],
                visible[observed],
                width=width,
                color=color,
                alpha=0.78,
                label=feature,
            )
            if show_imputed:
                filled = visible.notna() & imputed
                main_ax.bar(
                    frame.index[filled],
                    visible[filled],
                    width=width,
                    color=LIGHT_GREY,
                    edgecolor=color,
                    hatch="//",
                    label=f"{feature} (imputed)",
                )
        else:
            main_ax.plot(
                frame.index,
                visible,
                color=color,
                linewidth=1.7,
                label=feature,
            )
            observed = visible.notna() & ~imputed
            main_ax.scatter(
                frame.index[observed],
                visible[observed],
                color=color,
                s=22,
                zorder=3,
            )
            if show_imputed:
                filled = visible.notna() & imputed
                main_ax.scatter(
                    frame.index[filled],
                    visible[filled],
                    facecolor="white",
                    edgecolor=color,
                    marker="s",
                    s=42,
                    linewidth=1.4,
                    label=f"{feature} (imputed)",
                    zorder=4,
                )

    if is_clock:
        _format_clock_axis(main_ax)
    main_ax.set(
        title=title,
        xlabel=None if show_coverage else (xlabel or time_level.replace("_", " ")),
        ylabel=ylabel,
    )
    _style_axis(main_ax)
    legend_columns = min(3, len(main_ax.get_legend_handles_labels()[0]))
    main_ax.legend(frameon=False, ncol=legend_columns)
    if coverage_ax is not None:
        _plot_coverage_axis(coverage_ax, frame, selected, colors)
        coverage_ax.set_xlabel(xlabel or time_level.replace("_", " "))
    fig.autofmt_xdate()
    _finish_figure(fig, show=show, save_to=save_to)
    return (
        {"trajectory": main_ax, "coverage": coverage_ax}
        if coverage_ax is not None
        else main_ax
    )


def plot_trajectory_trend(
    trajectory: pd.DataFrame,
    feature: str,
    *,
    user_id=None,
    kind: str = "line",
    temporal: bool | None = None,
    show_imputed: bool = True,
    show_baseline: bool = True,
    show_normal_range: bool = True,
    show_deviation: bool = False,
    show_coverage: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize=(11, 5),
    ax=None,
    show: bool = True,
    save_to=None,
):
    """Plot observed or imputed values with their historical trend context.

    ``trajectory`` must already contain columns produced by
    :func:`pywearable.trajectory.compute_trajectory_trend`. The plot can show
    the baseline, normal range, deviations from baseline, and source coverage
    independently. Clock-valued features use a continuous time-of-day axis
    across midnight.

    Parameters
    ----------
    trajectory
        Canonical trajectory enriched by
        :func:`pywearable.trajectory.compute_trajectory_trend`.
    feature
        Feature and corresponding ``<feature>__trend_*`` columns to draw.
    user_id
        Participant to draw. It may be omitted for a single-user trajectory.
    kind
        ``"line"`` or ``"bar"`` for the source trajectory.
    temporal
        Override automatic clock-time detection. Clock features are unwrapped
        across midnight.
    show_imputed
        Include values marked by ``<feature>__is_imputed`` and identify them
        separately from observations.
    show_baseline
        Draw ``<feature>__trend_baseline``.
    show_normal_range
        Shade the interval between ``__trend_normal_lower`` and
        ``__trend_normal_upper``.
    show_deviation
        Draw a segment from each value to its historical baseline.
    show_coverage
        Add a lower panel from ``<feature>__coverage``.
    title, xlabel, ylabel, figsize, ax
        Labels and layout for a new or supplied axis/axes. Coverage mode
        requires two axes.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Returns
    -------
    matplotlib.axes.Axes or dict
        The trend axis, or ``{"trajectory", "coverage"}`` when coverage is
        requested.
    """
    frame, _, time_level = _select_user(trajectory, user_id)
    _feature_list(frame, feature)
    if kind not in {"line", "bar"}:
        raise ValueError("kind must be 'line' or 'bar'")
    prefix = f"{feature}__trend"
    required = [
        f"{prefix}_baseline",
        f"{prefix}_normal_lower",
        f"{prefix}_normal_upper",
    ]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"trajectory is missing trend columns: {missing}")

    is_clock = _is_clock_series(frame[feature], feature, temporal)
    source_columns = [frame[feature], *(frame[name] for name in required)]
    anchor = _clock_anchor(*source_columns) if is_clock else None

    def values(name):
        source = frame[name]
        return (
            _clock_hours(source, anchor)
            if is_clock
            else pd.to_numeric(source, errors="coerce")
        )

    observed_values = values(feature)
    baseline = values(f"{prefix}_baseline")
    lower = values(f"{prefix}_normal_lower")
    upper = values(f"{prefix}_normal_upper")
    flag = f"{feature}__is_imputed"
    imputed = (
        frame[flag].fillna(False).astype(bool)
        if flag in frame
        else pd.Series(False, index=frame.index)
    )
    visible = observed_values.mask(imputed) if not show_imputed else observed_values

    fig, main_ax, coverage_ax = _trajectory_axes(ax, show_coverage, figsize)
    if show_normal_range:
        main_ax.fill_between(
            frame.index,
            lower,
            upper,
            color=LIGHT_BLUE,
            alpha=0.22,
            label="Historical normal range",
            zorder=0,
        )
    if show_baseline:
        main_ax.plot(
            frame.index,
            baseline,
            color=GREEN,
            linewidth=2,
            label="Baseline",
            zorder=2,
        )
    present = visible.notna()
    if kind == "bar":
        width = _bar_width(frame.index)
        source_colors = np.where(imputed.to_numpy(), LIGHT_GREY, BLUE)
        main_ax.bar(
            frame.index[present],
            visible[present],
            width=width,
            color=source_colors[present],
            alpha=0.8,
            label="Trajectory",
            zorder=1,
        )
    else:
        main_ax.plot(
            frame.index,
            visible,
            color=BLUE,
            linewidth=1.2,
            alpha=0.7,
            zorder=1,
        )
        measured = present & ~imputed
        main_ax.scatter(
            frame.index[measured],
            visible[measured],
            color=BLUE,
            s=24,
            label="Observed",
            zorder=3,
        )
        if show_imputed:
            filled = present & imputed
            main_ax.scatter(
                frame.index[filled],
                visible[filled],
                facecolor="white",
                edgecolor=VERMILLION,
                marker="s",
                s=46,
                linewidth=1.5,
                label="Imputed",
                zorder=4,
            )
    if show_deviation:
        deviation_rows = present & baseline.notna()
        main_ax.vlines(
            frame.index[deviation_rows],
            baseline[deviation_rows],
            visible[deviation_rows],
            color=VERMILLION,
            linewidth=1.3,
            alpha=0.75,
            label="Deviation from baseline",
            zorder=2,
        )
    if is_clock:
        _format_clock_axis(main_ax)
    main_ax.set(
        title=title,
        xlabel=None if show_coverage else (xlabel or time_level.replace("_", " ")),
        ylabel=ylabel,
    )
    _style_axis(main_ax)
    main_ax.legend(frameon=False, ncol=2)
    if coverage_ax is not None:
        _plot_coverage_axis(coverage_ax, frame, [feature], [GREEN])
        coverage_ax.set_xlabel(xlabel or time_level.replace("_", " "))
    fig.autofmt_xdate()
    _finish_figure(fig, show=show, save_to=save_to)
    return (
        {"trajectory": main_ax, "coverage": coverage_ax}
        if coverage_ax is not None
        else main_ax
    )


def _provider_feature_request(statistic):
    operations = {"mean", "median", "std", "min", "max", "count"}
    for feature in sorted(provider_features.PROVIDER_FEATURES, key=len, reverse=True):
        if statistic == feature:
            return feature, None
        prefix = f"{feature}_"
        if statistic.startswith(prefix):
            operation = statistic[len(prefix) :]
            if operation in operations or hierarchy.is_percentile(operation):
                return feature, operation
    return None


def _statistic_trajectory(loader, statistic, **kwargs):
    """Dispatch one canonical statistic to its domain extraction function."""
    if statistic in activity.ACTIVITY_STATISTICS:
        return activity.get_activity_statistics(loader, statistic=statistic, **kwargs)
    if statistic in heart_rate.HEART_RATE_STATISTICS:
        return heart_rate.get_heart_rate_statistics(
            loader, statistic=statistic, **kwargs
        )
    if statistic in hrv.HRV_STATISTICS:
        return hrv.get_hrv_statistics(loader, statistic=statistic, **kwargs)
    if statistic in respiration.RESPIRATION_STATISTICS or any(
        statistic.startswith(prefix)
        and hierarchy.is_percentile(statistic[len(prefix) :])
        for prefix in ("respiratory_rate_", "spo2_")
    ):
        return respiration.get_respiration_statistics(
            loader, statistic=statistic, **kwargs
        )
    if statistic in sleep.SLEEP_STATISTICS_DICT:
        return sleep.get_sleep_statistics(loader, statistic=statistic, **kwargs)
    provider_request = _provider_feature_request(statistic)
    if provider_request is not None:
        feature, operation = provider_request
        return provider_features.get_provider_features(
            loader,
            feature=feature,
            statistic=operation,
            **kwargs,
        )
    raise ValueError(f"cannot infer a pywearable domain for statistic {statistic!r}")


def _single_value_column(frame, *, suffix=None):
    columns = [
        column
        for column in frame.columns
        if column not in {"period_start", "period_end"}
        and not column.endswith("__coverage")
        and (suffix is None or column.endswith(suffix))
    ]
    if len(columns) != 1:
        raise ValueError(
            "hierarchical plotting expected one matching feature column, "
            f"found {columns}"
        )
    return columns[0]


def plot_hierarchical_trajectory(
    loader,
    statistic: str,
    resolutions,
    *,
    user_id,
    start_date=None,
    end_date=None,
    std_multiplier: float = 1.0,
    min_coverage=None,
    min_valid_children=None,
    statistics_kwargs: Mapping | None = None,
    temporal: bool | None = None,
    child_label: str | None = None,
    center_label: str | None = None,
    range_label: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize=(12, 5),
    ax: Axes | None = None,
    show: bool = True,
    save_to=None,
):
    """Compare one statistic at a child resolution with parent mean ± SD.

    The statistic prefix selects the activity, heart-rate, HRV, respiration,
    sleep, or provider-feature API. The required child and parent trajectories
    are calculated internally, avoiding manual construction of hierarchy
    column names.

    Parameters
    ----------
    loader
        Loader accepted by the statistic's domain function.
    statistic
        One canonical feature name, such as ``"sleep_TST"``,
        ``"heart_rate_mean"``, or ``"activity_steps"``.
    resolutions
        Exactly two increasing resolutions: the observed child grain and the
        parent comparison grain, for example ``("nightly", "weekly")``.
    user_id
        Participant to plot.
    start_date, end_date
        Half-open analysis bounds forwarded to the domain API.
    std_multiplier
        Width of the parent band in standard deviations around its mean.
    min_coverage
        Optional domain coverage threshold. A two-item sequence supplies child
        and parent thresholds; a scalar applies according to the domain API.
    min_valid_children
        Optional minimum child count for each parent estimate.
    statistics_kwargs
        Additional domain arguments such as ``time_reference``,
        ``loader_kwargs``, ``cycle_kwargs``, ``backend``, or ``chronotype``.
    temporal
        Override automatic clock-feature detection.
    child_label, center_label, range_label
        Optional legend labels. Defaults describe the statistic and requested
        resolutions.
    title, xlabel, ylabel, figsize, ax
        Labels and layout for a new or supplied axis.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing child observations, parent means, and mean ± SD bands.
    """
    if not isinstance(statistic, str):
        raise ValueError("statistic must be one canonical feature name")
    chain, hierarchical = hierarchy.normalize_resolution_chain(resolutions)
    if not hierarchical or len(chain) != 2:
        raise ValueError("resolutions must contain exactly two increasing levels")
    if not np.isfinite(std_multiplier) or std_multiplier < 0:
        raise ValueError("std_multiplier must be finite and non-negative")
    extra = dict(statistics_kwargs or {})
    reserved = {
        "loader",
        "statistic",
        "feature",
        "user_id",
        "start_date",
        "end_date",
        "resolution",
        "aggregation",
        "min_coverage",
        "min_valid_children",
    }
    duplicated = reserved.intersection(extra)
    if duplicated:
        raise ValueError(
            "statistics_kwargs duplicates explicit arguments: "
            f"{sorted(duplicated)}"
        )
    child_coverage = (
        min_coverage[0]
        if isinstance(min_coverage, (list, tuple))
        else min_coverage
    )
    common = {
        "user_id": user_id,
        "start_date": start_date,
        "end_date": end_date,
        **extra,
    }
    child_trajectory = _statistic_trajectory(
        loader,
        statistic,
        resolution=chain[0],
        min_coverage=child_coverage,
        **common,
    )
    parent_trajectory = _statistic_trajectory(
        loader,
        statistic,
        resolution=chain,
        aggregation=["mean", "std"],
        min_coverage=min_coverage,
        min_valid_children=min_valid_children,
        **common,
    )
    children, _, time_level = _select_user(child_trajectory, user_id)
    parents, _, _ = _select_user(parent_trajectory, user_id)
    child_feature = _single_value_column(children)
    center_feature = _single_value_column(
        parents, suffix=f"__{chain[1]}_mean"
    )
    spread_feature = _single_value_column(
        parents, suffix=f"__{chain[1]}_std"
    )

    is_clock = _is_clock_series(children[child_feature], child_feature, temporal)
    anchor = (
        _clock_anchor(children[child_feature], parents[center_feature])
        if is_clock
        else None
    )
    if is_clock:
        child_values = _clock_hours(children[child_feature], anchor)
        centers = _clock_hours(parents[center_feature], anchor)
        spread = pd.to_numeric(parents[spread_feature], errors="coerce") / 60
    else:
        child_values = pd.to_numeric(children[child_feature], errors="coerce")
        centers = pd.to_numeric(parents[center_feature], errors="coerce")
        spread = pd.to_numeric(parents[spread_feature], errors="coerce")
    lower = centers - std_multiplier * spread
    upper = centers + std_multiplier * spread

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.scatter(
        children.index,
        child_values,
        color=GREY,
        alpha=0.65,
        s=22,
        label=child_label or f"{chain[0]} observations",
        zorder=3,
    )
    starts = pd.to_datetime(parents.get("period_start", parents.index))
    ends = pd.to_datetime(parents.get("period_end", parents.index))
    range_labeled = False
    center_labeled = False
    for position, (start, end, center) in enumerate(zip(starts, ends, centers)):
        if pd.isna(start) or pd.isna(end) or pd.isna(center):
            continue
        if pd.notna(lower.iloc[position]) and pd.notna(upper.iloc[position]):
            ax.fill_between(
                [start, end],
                [lower.iloc[position]] * 2,
                [upper.iloc[position]] * 2,
                color=LIGHT_BLUE,
                alpha=0.22,
                label=(
                    range_label or f"{chain[1]} mean ± {std_multiplier:g} SD"
                )
                if not range_labeled
                else None,
                zorder=0,
            )
            range_labeled = True
        ax.hlines(
            center,
            start,
            end,
            color=BLUE,
            linewidth=2.4,
            label=(center_label or f"{chain[1]} mean")
            if not center_labeled
            else None,
            zorder=2,
        )
        center_labeled = True
    if is_clock:
        _format_clock_axis(ax)
    ax.set(
        title=title or f"{statistic}: {chain[0]} to {chain[1]}",
        xlabel=xlabel or time_level.replace("_", " "),
        ylabel=ylabel,
    )
    _style_axis(ax)
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    _finish_figure(fig, show=show, save_to=save_to)
    return ax


def _calendar_matrix(dates, values, start_date=None, end_date=None):
    if not len(dates) and (start_date is None or end_date is None):
        raise ValueError(
            "an empty trajectory requires explicit start_date and end_date"
        )
    start = (
        dates.min().normalize()
        if start_date is None
        else pd.Timestamp(start_date).normalize()
    )
    end = (
        dates.max().normalize() + pd.Timedelta(days=1)
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    if end <= start:
        raise ValueError("end_date must be after start_date")
    monday = start - pd.Timedelta(days=start.weekday())
    final_day = end - pd.Timedelta(days=1)
    sunday = final_day + pd.Timedelta(days=6 - final_day.weekday())
    weeks = int((sunday - monday).days // 7 + 1)
    matrix = np.full((7, weeks), np.nan, dtype=float)
    inside = np.zeros((7, weeks), dtype=bool)
    for day in pd.date_range(start, end, inclusive="left"):
        week = int((day.normalize() - monday).days // 7)
        inside[day.weekday(), week] = True
    for day, value in zip(dates.normalize(), values):
        if day < start or day >= end:
            continue
        week = int((day - monday).days // 7)
        matrix[day.weekday(), week] = value
    return matrix, inside, start, end, monday


def _draw_month_boundaries(ax, start, end, monday, *, color, linewidth):
    """Outline calendar cells where a neighboring day belongs to another month."""
    period = pd.date_range(start, end, inclusive="left")
    available = set(period)
    for day in period:
        column = int((day - monday).days // 7)
        row = day.weekday()
        month = day.to_period("M")
        previous_day = day - pd.Timedelta(days=1)
        next_day = day + pd.Timedelta(days=1)
        previous_week = day - pd.Timedelta(days=7)
        next_week = day + pd.Timedelta(days=7)
        different = {
            "top": row == 0
            or previous_day not in available
            or previous_day.to_period("M") != month,
            "bottom": row == 6
            or next_day not in available
            or next_day.to_period("M") != month,
            "left": previous_week not in available
            or previous_week.to_period("M") != month,
            "right": next_week not in available or next_week.to_period("M") != month,
        }
        if different["top"]:
            ax.plot(
                [column, column + 1],
                [row, row],
                color=color,
                linewidth=linewidth,
                clip_on=False,
            )
        if different["bottom"]:
            ax.plot(
                [column, column + 1],
                [row + 1, row + 1],
                color=color,
                linewidth=linewidth,
                clip_on=False,
            )
        if different["left"]:
            ax.plot(
                [column, column],
                [row, row + 1],
                color=color,
                linewidth=linewidth,
                clip_on=False,
            )
        if different["right"]:
            ax.plot(
                [column + 1, column + 1],
                [row, row + 1],
                color=color,
                linewidth=linewidth,
                clip_on=False,
            )


def plot_calendar_heatmap(
    trajectory: pd.DataFrame,
    feature: str,
    *,
    user_id=None,
    start_date=None,
    end_date=None,
    temporal: bool | None = None,
    cmap: str | Colormap = "viridis",
    vmin=None,
    vmax=None,
    missing_color=LIGHT_GREY,
    month_boundaries: bool = True,
    month_boundary_color=BLACK,
    annotate: bool = False,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    figsize=(12, 4),
    ax: Axes | None = None,
    show: bool = True,
    save_to=None,
):
    """Plot one participant's daily values in a week-by-week calendar.

    Parameters
    ----------
    trajectory
        Canonical daily trajectory indexed by ``(user_id, time)``. The input
        must contain no more than one row per calendar date.
    feature
        Column represented by cell color.
    user_id
        Participant to draw. It may be omitted for a single-user trajectory.
    start_date, end_date
        Optional half-open calendar bounds. Dates without a trajectory row are
        retained as missing cells.
    temporal
        Override automatic clock-time detection. Clock values are unwrapped
        across midnight and colorbar labels are formatted as ``HH:MM``.
    cmap, vmin, vmax
        Matplotlib colormap and optional fixed color limits.
    missing_color
        Color for dates inside the requested interval whose value is missing.
    month_boundaries
        Draw a visible outline around each calendar month.
    month_boundary_color
        Color used for the month outlines.
    annotate
        Write the value inside every observed cell.
    colorbar, colorbar_label
        Control the colorbar and its label.
    title, figsize, ax
        Optional title, new-figure size, or existing axis.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Notes
    -----
    Missing values use ``missing_color`` and remain distinct from numeric zero.

    Returns
    -------
    matplotlib.axes.Axes
        The calendar axis. Its figure also owns the optional colorbar.
    """
    frame, _, _ = _select_user(trajectory, user_id)
    _feature_list(frame, feature)
    normalized = frame.index.normalize()
    if normalized.duplicated().any():
        raise ValueError("calendar heatmaps require at most one row per date")
    is_clock = _is_clock_series(frame[feature], feature, temporal)
    anchor = _clock_anchor(frame[feature]) if is_clock else None
    values = (
        _clock_hours(frame[feature], anchor).to_numpy()
        if is_clock
        else pd.to_numeric(frame[feature], errors="coerce").to_numpy()
    )
    matrix, inside, start, end, monday = _calendar_matrix(
        frame.index, values, start_date, end_date
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for row, column in zip(*np.where(inside & np.isnan(matrix))):
        ax.add_patch(
            Rectangle(
                (column, row),
                1,
                1,
                facecolor=missing_color,
                edgecolor="white",
                linewidth=0.8,
            )
        )
    finite = matrix[np.isfinite(matrix)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1
    image = ax.pcolormesh(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidth=0.8,
        shading="flat",
    )
    if month_boundaries:
        _draw_month_boundaries(
            ax,
            start,
            end,
            monday,
            color=month_boundary_color,
            linewidth=1.5,
        )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_yticks(np.arange(7) + 0.5)
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    month_positions = {}
    for day in pd.date_range(start, end, inclusive="left"):
        key = (day.year, day.month)
        month_positions.setdefault(key, []).append((day - monday).days // 7 + 0.5)
    ax.set_xticks([np.mean(value) for value in month_positions.values()])
    ax.set_xticklabels(
        [
            pd.Timestamp(year=year, month=month, day=1).strftime("%b\n%Y")
            for year, month in month_positions
        ]
    )
    ax.tick_params(length=0)
    ax.set_title(title or feature)
    ax.spines[:].set_visible(False)
    if annotate:
        for row, column in zip(*np.where(np.isfinite(matrix))):
            label = (
                _clock_formatter(matrix[row, column], None)
                if is_clock
                else f"{matrix[row, column]:.1f}"
            )
            ax.text(
                column + 0.5,
                row + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=7,
            )
    if colorbar:
        bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        bar.set_label(colorbar_label or feature)
        if is_clock:
            bar.ax.yaxis.set_major_formatter(FuncFormatter(_clock_formatter))
    _finish_figure(fig, show=show, save_to=save_to)
    return ax


def _infer_interval_minutes(times: pd.DatetimeIndex) -> float:
    minutes = np.sort(
        pd.Series(times.hour * 60 + times.minute + times.second / 60).unique()
    )
    differences = np.diff(minutes)
    positive = differences[differences > 0]
    if not len(positive):
        return 60.0
    rounded = pd.Series(np.round(positive, 6))
    return float(rounded.mode().iloc[0])


def _time_of_day_hours(value) -> float:
    """Convert a clock-like value to fractional hours in ``[0, 24)``."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        hours = float(value)
    else:
        hours = time_utils.clock_seconds(pd.Timestamp(f"2000-01-01 {value}")) / 3600
    if not np.isfinite(hours) or not 0 <= hours <= 24:
        raise ValueError("clock values must fall between 00:00 and 24:00")
    return hours


def _shade_clock_window(ax, window, *, color="#C9D7EC", alpha=0.65):
    """Shade a local clock interval, including one that crosses midnight."""
    if window is None:
        return
    if len(window) != 2:
        raise ValueError("night_window must contain start and end clock times")
    start, end = (_time_of_day_hours(value) for value in window)
    if start == end:
        return
    spans = [(start, end)] if start < end else [(0, end), (start, 24)]
    for left, right in spans:
        ax.axvspan(left, right, color=color, alpha=alpha, linewidth=0, zorder=0)


def _shade_clock_windows(ax, windows, *, color="#C9D7EC", alpha=0.65):
    """Shade one clock interval or a sequence of non-wrapping intervals."""
    if windows is None:
        return
    intervals = (
        windows
        if len(windows) and isinstance(windows[0], (list, tuple))
        else [windows]
    )
    for window in intervals:
        _shade_clock_window(ax, window, color=color, alpha=alpha)


def _weekday_sleep_windows(loader, user_id, start_date, end_date):
    """Return mean night segments for each local weekday.

    Bedtime and wake-up are first averaged as offsets from the sleep calendar
    date.  The resulting mean session is then split at midnight.  This keeps
    the evening and morning pieces tied to the same average night instead of
    averaging those pieces from different subsets of sessions.
    """
    nightly = sleep.get_sleep_statistics(
        loader,
        statistic=[
            sleep.SLEEP_STATISTIC_BEDTIME,
            sleep.SLEEP_STATISTIC_WAKEUP,
        ],
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="nightly",
    )
    if nightly.empty:
        return {}
    nightly = nightly.xs(user_id, level=schema.USER_COL)
    if isinstance(nightly.index, pd.MultiIndex):
        time_levels = [
            name
            for name in nightly.index.names
            if name != schema.TIMEZONE_COL
        ]
        if len(time_levels) != 1:
            raise ValueError("nightly sleep must contain one calendar-date level")
        calendar_dates = pd.to_datetime(
            nightly.index.get_level_values(time_levels[0]), errors="coerce"
        )
    else:
        calendar_dates = pd.to_datetime(nightly.index, errors="coerce")

    sessions = {weekday: [] for weekday in range(7)}
    for calendar_date, row in zip(calendar_dates, nightly.itertuples()):
        bedtime = pd.Timestamp(getattr(row, sleep.SLEEP_STATISTIC_BEDTIME))
        wakeup = pd.Timestamp(getattr(row, sleep.SLEEP_STATISTIC_WAKEUP))
        if (
            pd.isna(calendar_date)
            or pd.isna(bedtime)
            or pd.isna(wakeup)
            or wakeup <= bedtime
        ):
            continue
        anchor = calendar_date.normalize()
        sessions[anchor.weekday()].append(
            (
                (bedtime - anchor).total_seconds() / 3600,
                (wakeup - anchor).total_seconds() / 3600,
            )
        )

    means = {
        weekday: (
            float(np.mean([value[0] for value in values])),
            float(np.mean([value[1] for value in values])),
        )
        for weekday, values in sessions.items()
        if values
    }
    windows = {}
    for weekday, label in enumerate(_WEEKDAY_NAMES):
        segments = []
        if weekday in means:
            bedtime, wakeup = means[weekday]
            morning_start = max(0.0, bedtime)
            morning_end = min(24.0, wakeup)
            if morning_end > morning_start:
                segments.append((morning_start, morning_end))
        following_weekday = (weekday + 1) % 7
        if following_weekday in means:
            following_bedtime, _ = means[following_weekday]
            if following_bedtime < 0:
                evening_start = max(0.0, 24.0 + following_bedtime)
                if evening_start < 24:
                    segments.append((evening_start, 24.0))
        if segments:
            windows[label] = segments
    return windows


def plot_coverage_heatmap(
    trajectory: pd.DataFrame,
    feature: str,
    *,
    user_id=None,
    interval=None,
    cmap: str | Colormap = "RdYlGn",
    missing_color=LIGHT_GREY,
    title: str | None = None,
    figsize=(12, 5),
    ax: Axes | None = None,
    show: bool = True,
    save_to=None,
):
    """Plot subdaily coverage by calendar date and local time of day.

    Parameters
    ----------
    trajectory
        Canonical subdaily trajectory indexed by ``(user_id, time)``.
    feature
        Feature whose ``<feature>__coverage`` column is visualized. The
        feature value itself is not used.
    user_id
        Participant to draw. It may be omitted for a single-user trajectory.
    interval
        Time-of-day bin width accepted by :class:`pandas.Timedelta`, such as
        ``"5min"`` or ``"1h"``. By default the most common native spacing is
        used. Multiple rows in a bin are averaged.
    cmap
        Matplotlib colormap. The default maps low coverage to red, partial
        coverage to yellow, and complete coverage to green.
    missing_color
        Color used where coverage is unknown.
    title, figsize, ax
        Optional title, new-figure size, or existing axis.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Notes
    -----
    Rows are calendar dates and columns are local time-of-day intervals. This
    plot shows the coverage semantics already present in the trajectory; it
    does not infer wearing time itself.

    Returns
    -------
    matplotlib.axes.Axes
        The heatmap axis. Its figure also owns the colorbar.
    """
    frame, _, _ = _select_user(trajectory, user_id)
    column = f"{feature}__coverage"
    if column not in frame:
        raise ValueError(f"trajectory is missing coverage column: {column}")
    interval_minutes = (
        _infer_interval_minutes(frame.index)
        if interval is None
        else pd.Timedelta(interval).total_seconds() / 60
    )
    if interval_minutes <= 0 or interval_minutes > 24 * 60:
        raise ValueError("interval must be between zero and one day")
    slots = int(np.ceil(24 * 60 / interval_minutes))
    days = pd.date_range(frame.index.min().normalize(), frame.index.max().normalize())
    matrix = np.full((len(days), slots), np.nan, dtype=float)
    records = pd.DataFrame(
        {
            "day": frame.index.normalize(),
            "slot": np.floor(
                (frame.index.hour * 60 + frame.index.minute + frame.index.second / 60)
                / interval_minutes
            ).astype(int),
            "coverage": pd.to_numeric(frame[column], errors="coerce").to_numpy(),
        }
    )
    grouped = records.groupby(["day", "slot"], dropna=False)["coverage"].mean()
    day_lookup = {day: index for index, day in enumerate(days)}
    for (day, slot), value in grouped.items():
        if day in day_lookup and 0 <= slot < slots:
            matrix[day_lookup[day], slot] = value
    cmap_object = mpl.colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    cmap_object.set_bad(missing_color)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, 24, len(days), 0),
        cmap=cmap_object,
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 3)])
    tick_count = min(10, len(days))
    positions = np.linspace(0.5, len(days) - 0.5, tick_count)
    indices = np.clip(np.floor(positions).astype(int), 0, len(days) - 1)
    ax.set_yticks(positions)
    ax.set_yticklabels([days[index].strftime("%b %d") for index in indices])
    ax.set(
        title=title or f"{feature} coverage",
        xlabel="Local time of day",
        ylabel="Calendar date",
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("Observed / expected")
    _finish_figure(fig, show=show, save_to=save_to)
    return ax


def plot_weekly_profile(
    trajectory: pd.DataFrame,
    feature: str,
    *,
    user_id=None,
    loader=None,
    aggregation: str = "mean",
    kind: str = "lines",
    interval=None,
    include_imputed: bool = False,
    cmap: str | Colormap = "viridis",
    night_window="auto",
    night_color="#C9D7EC",
    title: str | None = None,
    ylabel: str | None = None,
    figsize=(10, 9),
    ax=None,
    show: bool = True,
    save_to=None,
):
    """Plot a participant's typical within-day profile for each weekday.

    This descriptive view aggregates an already prepared subdaily trajectory
    by weekday and local-time slot. It answers questions such as when steps are
    usually accumulated or how heart rate varies through a typical weekday.
    It does not calculate calendar-period features.

    Parameters
    ----------
    trajectory
        Canonical subdaily trajectory indexed by ``(user_id, time)``.
    feature
        Numeric feature summarized within each weekday/time-of-day cell.
    user_id
        Participant to draw. It may be omitted for a single-user trajectory.
    loader
        Optional loader used when ``night_window="auto"``. Each observed sleep
        session is expressed relative to its sleep calendar date, averaged by
        that weekday, and only then split at local midnight. Consequently the
        evening segment on one panel and morning segment on the next panel are
        two parts of the same mean night.
    aggregation
        ``"mean"`` or ``"median"`` across matching weekday/slot values.
    kind
        ``"lines"`` (default) creates seven aligned small-multiple line plots.
        ``"heatmap"`` creates a compact weekday-by-time matrix.
    interval
        Time-of-day bin width accepted by :class:`pandas.Timedelta`. By
        default the most common native spacing is retained. This parameter
        aggregates existing rows; it does not reconstruct finer observations.
    include_imputed
        Include values marked by ``<feature>__is_imputed`` in the summary.
    cmap
        Colormap used only for ``kind="heatmap"``.
    night_window
        ``"auto"`` (default) uses weekday-specific local-day sleep segments
        when a loader is supplied. An explicit ``(start, end)`` clock pair
        shades the same interval in every panel. A window crossing midnight is
        supported; pass ``None`` to disable shading.
    night_color
        Color of the night shading.
    title, ylabel, figsize, ax
        Labels and layout. For line plots, ``ax`` must contain seven axes; for
        a heatmap it must be one axis.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Returns
    -------
    numpy.ndarray or matplotlib.axes.Axes
        Seven axes for line small multiples, or one heatmap axis.
    """
    frame, selected_user, _ = _select_user(trajectory, user_id)
    _feature_list(frame, feature)
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'")
    if kind not in {"heatmap", "lines"}:
        raise ValueError("kind must be 'heatmap' or 'lines'")
    values = pd.to_numeric(frame[feature], errors="coerce")
    flag = f"{feature}__is_imputed"
    if not include_imputed and flag in frame:
        values = values.mask(frame[flag].fillna(False).astype(bool))
    interval_minutes = (
        _infer_interval_minutes(frame.index)
        if interval is None
        else pd.Timedelta(interval).total_seconds() / 60
    )
    if interval_minutes <= 0 or interval_minutes > 24 * 60:
        raise ValueError("interval must be between zero and one day")
    slots = int(np.ceil(24 * 60 / interval_minutes))
    rows = pd.DataFrame(
        {
            "weekday": frame.index.weekday,
            "slot": np.floor(
                (frame.index.hour * 60 + frame.index.minute + frame.index.second / 60)
                / interval_minutes
            ).astype(int),
            "value": values.to_numpy(),
        }
    )
    grouped = getattr(rows.groupby(["weekday", "slot"])["value"], aggregation)()
    matrix = np.full((7, slots), np.nan, dtype=float)
    for (weekday, slot), value in grouped.items():
        if 0 <= slot < slots:
            matrix[weekday, slot] = value
    if night_window == "auto":
        night_windows = (
            _weekday_sleep_windows(
                loader,
                selected_user,
                frame.index.min().normalize(),
                frame.index.max().normalize() + pd.Timedelta(days=1),
            )
            if loader is not None
            else {}
        )
    elif night_window is None:
        night_windows = {}
    else:
        if isinstance(night_window, str):
            raise ValueError(
                "night_window must be 'auto', None, or a (start, end) pair"
            )
        night_windows = {name: night_window for name in _WEEKDAY_NAMES}
    if kind == "heatmap":
        if ax is None:
            fig, profile_ax = plt.subplots(figsize=figsize)
        elif isinstance(ax, Axes):
            profile_ax = ax
            fig = ax.figure
        else:
            raise ValueError("kind='heatmap' requires one Matplotlib Axes")
        cmap_object = (
            mpl.colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
        )
        cmap_object.set_bad(LIGHT_GREY)
        image = profile_ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(0, 24, 7, 0),
            cmap=cmap_object,
        )
        profile_ax.set_yticks(np.arange(7) + 0.5)
        profile_ax.set_yticklabels(_WEEKDAY_NAMES)
        profile_ax.set_xticks(np.arange(0, 25, 3))
        profile_ax.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 3)])
        profile_ax.set(
            title=title or f"Typical week: {feature} ({aggregation})",
            xlabel="Local time of day",
        )
        bar = fig.colorbar(image, ax=profile_ax, fraction=0.025, pad=0.02)
        bar.set_label(ylabel or feature)
        result = profile_ax
    else:
        if ax is None:
            fig, profile_axes = plt.subplots(
                7,
                1,
                figsize=figsize,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
        else:
            profile_axes = np.asarray(ax, dtype=object).reshape(-1)
            if len(profile_axes) != 7 or not all(
                isinstance(axis, Axes) for axis in profile_axes
            ):
                raise ValueError("kind='lines' requires seven Matplotlib axes")
            fig = profile_axes[0].figure
            if any(axis.figure is not fig for axis in profile_axes):
                raise ValueError("all weekly-profile axes must share a figure")
        x = (np.arange(slots) + 0.5) * interval_minutes / 60
        for weekday, profile_ax in enumerate(profile_axes):
            _shade_clock_windows(
                profile_ax,
                night_windows.get(_WEEKDAY_NAMES[weekday]),
                color=night_color,
            )
            profile_ax.plot(x, matrix[weekday], color=BLUE, linewidth=1.5)
            profile_ax.set_xlim(0, 24)
            profile_ax.set_ylabel(
                _WEEKDAY_NAMES[weekday][:3], rotation=0, labelpad=22
            )
            profile_ax.grid(axis="y", alpha=0.16)
            profile_ax.spines[["top", "right"]].set_visible(False)
        profile_axes[0].set_title(title or f"Typical week: {feature} ({aggregation})")
        profile_axes[-1].set_xticks(np.arange(0, 25, 3))
        profile_axes[-1].set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 3)])
        profile_axes[-1].set_xlabel("Local time of day")
        fig.supylabel(ylabel or feature)
        result = np.asarray(profile_axes)
    _finish_figure(fig, show=show, save_to=save_to)
    return result


def _normalize_cpd(cpd) -> list[str]:
    if cpd is None:
        return []
    requested = [cpd] if isinstance(cpd, str) else list(cpd)
    mapping = {
        "midpoint": sleep.SLEEP_STATISTIC_CPD_MIDPOINT,
        "duration": sleep.SLEEP_STATISTIC_CPD_DURATION,
        sleep.SLEEP_STATISTIC_CPD_MIDPOINT: sleep.SLEEP_STATISTIC_CPD_MIDPOINT,
        sleep.SLEEP_STATISTIC_CPD_DURATION: sleep.SLEEP_STATISTIC_CPD_DURATION,
    }
    unknown = [value for value in requested if value not in mapping]
    if unknown:
        raise ValueError(f"unsupported CPD panels: {unknown}")
    return list(dict.fromkeys(mapping[value] for value in requested))


def _sleep_clock_hour(value):
    if pd.isna(value):
        return np.nan
    return time_utils.unwrap_clock_seconds(
        time_utils.clock_seconds(value), 12 * 3600
    ) / 3600


def plot_sleep_summary(
    loader,
    user_id,
    start_date=None,
    end_date=None,
    *,
    show_scores: bool = True,
    show_boundaries: bool = True,
    cpd=None,
    cpd_trend: bool = False,
    cpd_trend_kwargs: Mapping | None = None,
    title: str | None = None,
    figsize=None,
    fontsize: float = 12,
    axes=None,
    show: bool = True,
    save_to=None,
):
    """Plot compact native hypnograms for a participant across many nights.

    Unlike the generic trajectory plots, this view accepts a loader because it
    combines sleep summaries, native stage intervals, calculated average sleep
    timing, provider scores, and optional composite-phase-deviation
    (CPD) panels.

    Parameters
    ----------
    loader
        Loader exposing canonical sleep summaries and sleep-stage intervals.
    user_id
        Participant identifier.
    start_date, end_date
        Inclusive sleep-calendar-date bounds. Missing dates remain visible as
        empty rows. A night assigned to ``start_date`` retains stages from the
        preceding civil date.
    show_scores
        Write the provider's nightly sleep score to the right of each row when
        available. Scores are colored using fixed quality bands.
    show_boundaries
        Draw dotted vertical lines at the participant's circular mean bedtime
        and wake-up time. It does not add a marker for every night.
    cpd
        Optional ``"midpoint"``, ``"duration"``, or sequence containing both.
        Each adds one deliberately narrow panel beside the hypnograms.
    cpd_trend
        Add a historical baseline and normal range to each CPD panel.
    cpd_trend_kwargs
        Keyword arguments forwarded to
        :func:`pywearable.trajectory.compute_trajectory_trend` for CPD panels.
    title, figsize, fontsize
        Figure title and optional size. The default grows vertically with the
        number of requested nights. ``fontsize`` controls titles, labels,
        ticks, scores, and the shared legend.
    axes
        Existing axes as a mapping keyed by ``"sleep"`` and requested CPD
        feature names, or as an ordered sequence.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Returns
    -------
    dict
        Axes keyed by ``"sleep"`` and any requested CPD feature names.

    Notes
    -----
    Stage intervals are drawn at native duration; no one-minute expansion is
    performed. The clock axis is cropped to the observed sleep range rather
    than showing an entire 24-hour day.
    """
    if not np.isfinite(fontsize) or fontsize <= 0:
        raise ValueError("fontsize must be a positive finite number")
    summary_start = (
        pd.Timestamp(start_date).normalize() if start_date is not None else None
    )
    summary_end = (
        pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        if end_date is not None
        else None
    )
    if summary_start is not None and summary_end is not None:
        if summary_end <= summary_start:
            raise ValueError("end_date must not precede start_date")
    summaries = schema.prepare_sleep_summary(
        loader.load_sleep_summary(user_id, summary_start, summary_end)
    )
    if summaries.empty:
        raise ValueError("no sleep summaries are available for the requested period")

    stage_start = summaries[schema.LOCAL_TIME_COL].min()
    stage_ends = (
        pd.to_datetime(summaries[schema.LOCAL_END_TIME_COL], errors="coerce")
        if schema.LOCAL_END_TIME_COL in summaries
        else pd.Series(pd.NaT, index=summaries.index, dtype="datetime64[ns]")
    )
    durations = pd.to_numeric(
        summaries[schema.DURATION_IN_MS_COL], errors="coerce"
    ).fillna(0)
    awake_durations = (
        pd.to_numeric(
            summaries[schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL],
            errors="coerce",
        ).fillna(0)
        if schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL in summaries
        else pd.Series(0.0, index=summaries.index)
    )
    reconstructed_ends = summaries[schema.LOCAL_TIME_COL] + pd.to_timedelta(
        durations + awake_durations, unit="ms"
    )
    stage_end = stage_ends.fillna(reconstructed_ends).max()
    if pd.isna(stage_end) or stage_end <= stage_start:
        raise ValueError("the selected sleep summaries have invalid session bounds")
    stages = schema.prepare_sleep_stages(
        loader.load_sleep_stage(user_id, stage_start, stage_end)
    )
    summary_ids = set(summaries[schema.SLEEP_SUMMARY_ID_COL])
    stages = stages[stages[schema.SLEEP_SUMMARY_ID_COL].isin(summary_ids)]

    start = (
        summary_start
        if summary_start is not None
        else summaries[schema.CALENDAR_DATE_COL].min().normalize()
    )
    end = (
        summary_end
        if summary_end is not None
        else summaries[schema.CALENDAR_DATE_COL].max().normalize()
        + pd.Timedelta(days=1)
    )
    dates = pd.date_range(start, end, inclusive="left")
    cpd_features = _normalize_cpd(cpd)
    statistics = []
    if show_scores:
        statistics.append(sleep.SLEEP_STATISTIC_SLEEP_SCORE)
    if show_boundaries:
        statistics.extend(
            [
                sleep.SLEEP_STATISTIC_BEDTIME,
                sleep.SLEEP_STATISTIC_WAKEUP,
            ]
        )
    statistics.extend(cpd_features)
    statistics = list(dict.fromkeys(statistics))
    metrics = (
        sleep.get_sleep_statistics(
            loader,
            statistic=statistics,
            user_id=user_id,
            start_date=start,
            end_date=end,
            resolution="nightly",
        ).xs(user_id, level=schema.USER_COL)
        if statistics
        else pd.DataFrame(index=dates)
    )

    axis_names = ["sleep", *cpd_features]
    if axes is None:
        if figsize is None:
            figsize = (
                8 + 0.9 * len(cpd_features),
                max(5.5, len(dates) * 0.34),
            )
        widths = [10, *([0.9] * len(cpd_features))]
        fig, created = plt.subplots(
            1,
            len(axis_names),
            figsize=figsize,
            sharey=True,
            gridspec_kw={"width_ratios": widths},
            constrained_layout=True,
            squeeze=False,
        )
        axes_map = dict(zip(axis_names, created[0]))
    elif isinstance(axes, Mapping):
        missing_axes = set(axis_names).difference(axes)
        if missing_axes:
            raise ValueError(f"axes is missing panels: {sorted(missing_axes)}")
        axes_map = dict(axes)
        fig = axes_map["sleep"].figure
    else:
        supplied = list(np.asarray(axes, dtype=object).flat)
        if len(supplied) != len(axis_names):
            raise ValueError(f"expected {len(axis_names)} axes")
        axes_map = dict(zip(axis_names, supplied))
        fig = supplied[0].figure

    main_ax = axes_map["sleep"]
    date_positions = {date: position for position, date in enumerate(dates)}
    summary_dates = summaries.set_index(schema.SLEEP_SUMMARY_ID_COL)[
        schema.CALENDAR_DATE_COL
    ].dt.normalize()
    for stage in stages.itertuples(index=False):
        summary_id = getattr(stage, schema.SLEEP_SUMMARY_ID_COL)
        calendar_date = summary_dates.get(summary_id)
        if calendar_date not in date_positions:
            continue
        start_hour = _sleep_clock_hour(getattr(stage, schema.LOCAL_TIME_COL))
        duration_hours = getattr(stage, schema.DURATION_IN_MS_COL) / 3_600_000
        stage_type = getattr(stage, schema.SLEEP_STAGE_SLEEP_TYPE_COL)
        main_ax.broken_barh(
            [(start_hour, duration_hours)],
            (date_positions[calendar_date] - 0.29, 0.58),
            facecolors=SLEEP_STAGE_COLORS[stage_type],
            edgecolors="none",
            alpha=SLEEP_STAGE_ALPHA.get(stage_type, 0.7),
        )

    boundary_styles = {
        sleep.SLEEP_STATISTIC_BEDTIME: (BLACK, "Average bedtime"),
        sleep.SLEEP_STATISTIC_WAKEUP: (GREEN, "Average wake-up"),
    }
    plotted_hours = []
    for stage in stages.itertuples(index=False):
        stage_start = _sleep_clock_hour(getattr(stage, schema.LOCAL_TIME_COL))
        duration = getattr(stage, schema.DURATION_IN_MS_COL) / 3_600_000
        if np.isfinite(stage_start) and np.isfinite(duration):
            plotted_hours.extend([stage_start, stage_start + duration])
    if show_boundaries:
        for feature, (color, label) in boundary_styles.items():
            if feature not in metrics:
                continue
            aligned = metrics[feature].reindex(dates)
            x = aligned.map(_sleep_clock_hour)
            if x.notna().any():
                average = float(x.mean())
                plotted_hours.append(average)
                main_ax.axvline(
                    average,
                    linestyle=":",
                    linewidth=1.7,
                    label=label,
                    zorder=4,
                    color=color,
                )
    scores = None
    if show_scores and sleep.SLEEP_STATISTIC_SLEEP_SCORE in metrics:
        scores = metrics[sleep.SLEEP_STATISTIC_SLEEP_SCORE].reindex(dates)

    def score_color(score):
        if pd.isna(score):
            return GREY
        if score >= 90:
            return "forestgreen"
        if score >= 80:
            return "limegreen"
        if score >= 60:
            return "darkorange"
        return "firebrick"

    if plotted_hours:
        x_min = np.floor(min(plotted_hours) - 0.5)
        x_max = np.ceil(max(plotted_hours) + 0.5)
    else:
        x_min, x_max = 20, 33
    score_x = x_max + 0.15
    if scores is not None:
        for position, score in enumerate(scores):
            if pd.notna(score):
                main_ax.text(
                    score_x,
                    position,
                    f"{score:.0f}",
                    va="center",
                    fontsize=fontsize - 1,
                    color=score_color(score),
                    weight="bold",
                )
        main_ax.text(
            score_x,
            -0.8,
            "Score",
            ha="left",
            fontsize=fontsize - 1,
            weight="bold",
        )
        x_max += 1.0
    main_ax.set_xlim(x_min, x_max)
    main_ax.set_xticks(np.arange(np.ceil(x_min), np.floor(x_max) + 1, 2))
    main_ax.xaxis.set_major_formatter(FuncFormatter(_clock_formatter))
    main_ax.set_yticks(np.arange(len(dates)))
    main_ax.set_yticklabels([date.strftime("%d/%m") for date in dates])
    main_ax.invert_yaxis()
    main_ax.set_title(title or "Sleep stages and score", fontsize=fontsize + 2)
    main_ax.set_xlabel("Participant-local clock time", fontsize=fontsize)
    main_ax.set_ylabel("Sleep calendar date", fontsize=fontsize)
    main_ax.tick_params(axis="both", labelsize=fontsize - 1)
    main_ax.xaxis.grid(True, color="#EEEEEE")
    main_ax.yaxis.grid(False)
    main_ax.set_axisbelow(True)
    main_ax.spines[["top", "right", "bottom"]].set_visible(False)
    main_ax.spines["left"].set_color("#DDDDDD")
    stage_handles = [
        Patch(
            color=color,
            alpha=SLEEP_STAGE_ALPHA.get(stage, 0.7),
            label=stage.title(),
        )
        for stage, color in SLEEP_STAGE_COLORS.items()
        if stage in set(stages.get(schema.SLEEP_STAGE_SLEEP_TYPE_COL, []))
    ]
    boundary_handles, boundary_labels = main_ax.get_legend_handles_labels()
    fig.legend(
        [*stage_handles, *boundary_handles],
        [handle.get_label() for handle in stage_handles] + boundary_labels,
        frameon=False,
        fontsize=fontsize - 1,
        ncol=max(1, min(4, len(stage_handles) + len(boundary_handles))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
    )

    trend_kwargs = dict(cpd_trend_kwargs or {})
    for feature in cpd_features:
        cpd_ax = axes_map[feature]
        values = pd.to_numeric(metrics[feature].reindex(dates), errors="coerce")
        cpd_ax.barh(
            np.arange(len(dates)),
            values,
            color=GREY,
            alpha=0.65,
            height=0.55,
            zorder=2,
        )
        if cpd_trend:
            cpd_frame = metrics[[feature, f"{feature}__coverage"]].copy()
            if not isinstance(cpd_frame.index, pd.MultiIndex):
                cpd_frame.index = pd.MultiIndex.from_arrays(
                    [[user_id] * len(cpd_frame), cpd_frame.index],
                    names=[schema.USER_COL, schema.CALENDAR_DATE_COL],
                )
            enriched = trajectory_analysis.compute_trajectory_trend(
                cpd_frame,
                features=feature,
                **trend_kwargs,
            ).xs(user_id, level=schema.USER_COL)
            prefix = f"{feature}__trend"
            baseline = enriched[f"{prefix}_baseline"].reindex(dates)
            lower = enriched[f"{prefix}_normal_lower"].reindex(dates)
            upper = enriched[f"{prefix}_normal_upper"].reindex(dates)
            cpd_ax.plot(
                baseline,
                np.arange(len(dates)),
                color=VERMILLION,
                linewidth=1.2,
                zorder=3,
            )
            cpd_ax.fill_betweenx(
                np.arange(len(dates)),
                lower,
                upper,
                color=GREEN,
                alpha=0.18,
                zorder=1,
            )
        label = "CPD\nmidpoint" if feature.endswith("midpoint") else "CPD\nduration"
        cpd_ax.set_title(label, fontsize=fontsize)
        cpd_ax.set_xlabel("Hours", fontsize=fontsize - 1)
        cpd_ax.tick_params(axis="both", labelsize=fontsize - 2)
        cpd_ax.tick_params(axis="y", labelleft=False)
        cpd_ax.grid(axis="x", color="#EEEEEE")
        cpd_ax.spines[["top", "right", "bottom"]].set_visible(False)
    _finish_figure(fig, show=show, save_to=save_to)
    return axes_map


def plot_sleep_heart_rate(
    loader,
    user_id,
    calendar_date,
    *,
    resolution="15min",
    show_stages: bool = True,
    stage_alpha: float | None = None,
    show_boundaries: bool = True,
    show_coverage: bool = False,
    title: str | None = None,
    figsize=(12, 5),
    ax=None,
    show: bool = True,
    save_to=None,
):
    """Plot raw and aggregated heart rate through one sleep session.

    Parameters
    ----------
    loader
        Loader exposing sleep summaries, sleep stages, and heart-rate samples.
    user_id
        Participant identifier.
    calendar_date
        Sleep calendar date selecting one main-sleep session.
    resolution
        Fixed window used for the overlaid heart-rate mean. The default is
        ``"15min"``; raw samples are always shown independently.
    show_stages
        Shade native sleep-stage intervals behind the heart-rate traces.
    stage_alpha
        Optional opacity applied to every stage background. By default each
        stage uses a subdued stage-specific opacity that keeps heart rate
        visually dominant while making wake and light sleep distinguishable.
        The external legend identifies every stage present in the session.
    show_boundaries
        Draw calculated bedtime, sleep-onset, wake-up, and get-up timestamps as
        dashed vertical lines when available. A sleep-onset line coincident
        with bedtime, or a get-up line coincident with wake-up, is suppressed.
    show_coverage
        Add a lower panel containing aggregated heart-rate coverage.
    title, figsize, ax
        Optional title, new-figure size, or existing axis/axes. Coverage mode
        requires two axes.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Returns
    -------
    matplotlib.axes.Axes or dict
        The heart-rate axis, or ``{"heart_rate", "coverage"}`` when coverage
        is requested.
    """
    if stage_alpha is not None and (
        not np.isfinite(stage_alpha) or not 0 <= stage_alpha <= 1
    ):
        raise ValueError("stage_alpha must lie between zero and one")
    date = pd.Timestamp(calendar_date).normalize()
    summaries = schema.prepare_sleep_summary(
        loader.load_sleep_summary(user_id, date, date + pd.Timedelta(days=1))
    )
    row = summaries[summaries[schema.CALENDAR_DATE_COL].dt.normalize().eq(date)]
    if row.empty:
        raise ValueError(f"no sleep summary is available for {date.date()}")
    summary = row.iloc[0]
    start = pd.Timestamp(summary[schema.LOCAL_TIME_COL])
    if schema.LOCAL_END_TIME_COL in row and pd.notna(
        summary[schema.LOCAL_END_TIME_COL]
    ):
        end = pd.Timestamp(summary[schema.LOCAL_END_TIME_COL])
    else:
        duration = pd.to_numeric(
            pd.Series([summary.get(schema.DURATION_IN_MS_COL)]), errors="coerce"
        ).iloc[0]
        awake = (
            pd.to_numeric(
                pd.Series(
                    [summary.get(schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL, 0)]
                ),
                errors="coerce",
            )
            .fillna(0)
            .iloc[0]
        )
        end = start + pd.to_timedelta(duration + awake, unit="ms")
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError("the selected sleep summary has invalid session bounds")

    raw = loader.load_heart_rate(
        user_id,
        start_date=start,
        end_date=end,
        time_col=schema.LOCAL_TIME_COL,
    ).copy()
    raw[schema.LOCAL_TIME_COL] = pd.to_datetime(
        raw[schema.LOCAL_TIME_COL], errors="coerce"
    )
    raw[schema.HEART_RATE_COL] = pd.to_numeric(
        raw[schema.HEART_RATE_COL], errors="coerce"
    )
    raw = raw.dropna(subset=[schema.LOCAL_TIME_COL, schema.HEART_RATE_COL]).sort_values(
        schema.LOCAL_TIME_COL
    )
    if raw.empty:
        raise ValueError(
            "no heart-rate samples are available during this sleep session"
        )
    aggregate = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
        user_id=user_id,
        start_date=start,
        end_date=end,
        resolution=resolution,
        time_reference="local",
    ).xs(user_id, level=schema.USER_COL)

    fig, main_ax, coverage_ax = _trajectory_axes(ax, show_coverage, figsize)
    present_stages = []
    if show_stages:
        stages = schema.prepare_sleep_stages(
            loader.load_sleep_stage(user_id, start_date=start, end_date=end)
        )
        stages = stages[
            stages[schema.SLEEP_SUMMARY_ID_COL].eq(summary[schema.SLEEP_SUMMARY_ID_COL])
        ]
        for stage in stages.itertuples(index=False):
            stage_start = pd.Timestamp(getattr(stage, schema.LOCAL_TIME_COL))
            stage_end = stage_start + pd.to_timedelta(
                getattr(stage, schema.DURATION_IN_MS_COL), unit="ms"
            )
            stage_type = getattr(stage, schema.SLEEP_STAGE_SLEEP_TYPE_COL)
            present_stages.append(stage_type)
            alpha = (
                SLEEP_HEART_RATE_STAGE_ALPHA.get(stage_type, 0.20)
                if stage_alpha is None
                else stage_alpha
            )
            main_ax.axvspan(
                stage_start,
                stage_end,
                color=SLEEP_STAGE_COLORS[stage_type],
                alpha=alpha,
                linewidth=0,
            )
    main_ax.scatter(
        raw[schema.LOCAL_TIME_COL],
        raw[schema.HEART_RATE_COL],
        color=GREY,
        alpha=0.35,
        s=10,
        label="Raw heart rate",
        zorder=2,
    )
    main_ax.step(
        aggregate.index,
        aggregate[heart_rate.HEART_RATE_STATISTIC_MEAN],
        where="post",
        color=BLUE,
        linewidth=2,
        label=f"{resolution} mean",
        zorder=3,
    )
    if show_boundaries:
        boundaries = sleep.get_sleep_statistics(
            loader,
            statistic=[
                sleep.SLEEP_STATISTIC_BEDTIME,
                sleep.SLEEP_STATISTIC_SLEEP_ONSET,
                sleep.SLEEP_STATISTIC_WAKEUP,
                sleep.SLEEP_STATISTIC_GETUP,
            ],
            user_id=user_id,
            start_date=date,
            end_date=date + pd.Timedelta(days=1),
        ).xs(user_id, level=schema.USER_COL)
        if date in boundaries.index:
            pairs = (
                (
                    (sleep.SLEEP_STATISTIC_BEDTIME, BLACK, "Bedtime"),
                    (sleep.SLEEP_STATISTIC_SLEEP_ONSET, BLUE, "Sleep onset"),
                ),
                (
                    (sleep.SLEEP_STATISTIC_WAKEUP, ORANGE, "Wake-up"),
                    (sleep.SLEEP_STATISTIC_GETUP, GREEN, "Get-up"),
                ),
            )
            for primary, secondary in pairs:
                primary_value = boundaries.loc[date, primary[0]]
                secondary_value = boundaries.loc[date, secondary[0]]
                if pd.notna(primary_value):
                    main_ax.axvline(
                        primary_value,
                        color=primary[1],
                        linestyle="--",
                        linewidth=1.2,
                        label=primary[2],
                    )
                if pd.notna(secondary_value) and (
                    pd.isna(primary_value)
                    or abs(pd.Timestamp(secondary_value) - pd.Timestamp(primary_value))
                    > pd.Timedelta(seconds=1)
                ):
                    main_ax.axvline(
                        secondary_value,
                        color=secondary[1],
                        linestyle="--",
                        linewidth=1.2,
                        label=secondary[2],
                    )
    main_ax.set(
        xlim=(start, end),
        title=title or f"Heart rate through sleep — {date:%Y-%m-%d}",
        ylabel="Heart rate (bpm)",
        xlabel=None if show_coverage else "Participant-local time",
    )
    main_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _style_axis(main_ax)
    handles, labels = main_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    stage_handles = [
        Patch(
            color=SLEEP_STAGE_COLORS[stage_type],
            alpha=(
                SLEEP_HEART_RATE_STAGE_ALPHA.get(stage_type, 0.20)
                if stage_alpha is None
                else stage_alpha
            ),
            label=f"{stage_type.title()} stage",
        )
        for stage_type in SLEEP_STAGE_COLORS
        if stage_type in set(present_stages)
    ]
    main_ax.legend(
        [*unique.values(), *stage_handles],
        [*unique.keys(), *(handle.get_label() for handle in stage_handles)],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    if coverage_ax is not None:
        coverage_column = f"{heart_rate.HEART_RATE_STATISTIC_MEAN}__coverage"
        coverage_ax.step(
            aggregate.index,
            aggregate[coverage_column],
            where="post",
            color=GREEN,
        )
        coverage_ax.set(
            xlim=(start, end),
            ylim=(-0.05, 1.05),
            ylabel="Coverage",
            xlabel="Participant-local time",
        )
        coverage_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        _style_axis(coverage_ax)
    _finish_figure(fig, show=show, save_to=save_to)
    return (
        {"heart_rate": main_ax, "coverage": coverage_ax}
        if coverage_ax is not None
        else main_ax
    )


def plot_period_comparison(
    trajectory: pd.DataFrame,
    feature: str,
    *,
    user_id=None,
    kind: str = "bar",
    show_percent_change: bool = True,
    favorable_direction: str | None = None,
    period_format: str = "%Y-%m-%d",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize=(11, 5),
    ax: Axes | None = None,
    show: bool = True,
    save_to=None,
):
    """Compare already-aggregated periods without re-aggregating the data.

    Values use one neutral color. The only annotations are signed consecutive
    percentage changes; absolute values remain readable from the y-axis.
    Missing periods are blank and break the comparison chain.

    Parameters
    ----------
    trajectory
        Canonical trajectory already calculated at the comparison resolution,
        such as weekly total steps or monthly mean resting heart rate.
    feature
        Numeric period-level feature to draw.
    user_id
        Participant to draw. It may be omitted for a single-user trajectory.
    kind
        ``"bar"`` or ``"line"``.
    show_percent_change
        Annotate change from the immediately preceding non-missing period.
        A missing period breaks the comparison chain.
    favorable_direction
        ``"higher"`` colors increases green and decreases vermillion;
        ``"lower"`` reverses that interpretation. With ``None``, green means
        an increase and vermillion a decrease, without implying benefit.
    period_format
        ``strftime`` format used for the x-axis labels.
    title, xlabel, ylabel, figsize, ax
        Labels and layout for a new or supplied axis.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Notes
    -----
    This function does not choose or calculate the period aggregation. Use the
    relevant domain hierarchy first so the meaning of each bar is explicit.

    Returns
    -------
    matplotlib.axes.Axes
        The period-comparison axis.
    """
    frame, _, time_level = _select_user(trajectory, user_id)
    _feature_list(frame, feature)
    if kind not in {"bar", "line"}:
        raise ValueError("kind must be 'bar' or 'line'")
    if favorable_direction not in {None, "higher", "lower"}:
        raise ValueError("favorable_direction must be None, 'higher', or 'lower'")
    values = pd.to_numeric(frame[feature], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError("the selected period trajectory has no numeric values")
    previous = values.shift(1)
    change = (values - previous) / previous.abs() * 100
    change = change.where(values.notna() & previous.notna() & previous.ne(0))
    positions = np.arange(len(frame))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if kind == "bar":
        artists = ax.bar(
            positions,
            values.fillna(0),
            width=0.72,
            color=BLUE,
            alpha=0.82,
        )
        coordinates = [
            (bar.get_x() + bar.get_width() / 2, bar.get_height()) for bar in artists
        ]
    else:
        ax.plot(
            positions,
            values,
            color=BLUE,
            marker="o",
            linewidth=2,
        )
        coordinates = list(zip(positions, values))
    finite = values.dropna()
    maximum = float(finite.max())
    minimum = float(finite.min())
    span = maximum - minimum
    for position, ((x, y), delta) in enumerate(zip(coordinates, change)):
        if pd.isna(values.iloc[position]):
            continue
        if (
            show_percent_change
            and position > 0
            and pd.notna(delta)
            and np.isfinite(delta)
        ):
            if favorable_direction is None:
                positive = delta >= 0
            else:
                positive = (
                    delta >= 0 if favorable_direction == "higher" else delta <= 0
                )
            color = GREEN if positive else VERMILLION
            arrow = "↑" if delta >= 0 else "↓"
            ax.annotate(
                f"{arrow} {abs(delta):.1f}%",
                (x, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                weight="bold",
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [pd.Timestamp(value).strftime(period_format) for value in frame.index],
        rotation=30,
        ha="right",
    )
    ax.set(
        title=title or feature,
        xlabel=xlabel or time_level.replace("_", " "),
        ylabel=ylabel,
    )
    if minimum >= 0:
        padding = 0.12 * (span if span > 0 else max(abs(maximum), 1))
        ax.set_ylim(0, maximum + padding)
    else:
        ax.margins(y=0.12)
    _style_axis(ax)
    _finish_figure(fig, show=show, save_to=save_to)
    return ax


def _resolution_hours(value) -> float:
    duration = pd.Timedelta(value)
    hours = duration.total_seconds() / 3600
    if not np.isfinite(hours) or hours <= 0 or hours > 24:
        raise ValueError("resolutions must be positive and no longer than one day")
    return hours


def _resolution_label(value) -> str:
    """Format a fixed resolution for a compact axis unit."""
    seconds = pd.Timedelta(value).total_seconds()
    for divisor, unit in ((86_400, "d"), (3_600, "h"), (60, "min")):
        if seconds >= divisor and seconds % divisor == 0:
            return f"{seconds / divisor:g} {unit}"
    return f"{seconds:g} s"


def _shade_missing_windows(ax, times, missing, width_hours):
    """Shade consecutive missing fixed windows on an elapsed-hour axis."""
    positions = np.flatnonzero(np.asarray(missing, dtype=bool))
    if not len(positions):
        return
    groups = np.split(positions, np.where(np.diff(positions) != 1)[0] + 1)
    for group in groups:
        left = times[group[0]]
        right = times[group[-1]] + width_hours
        ax.axvspan(left, right, color=GREY, alpha=0.22, linewidth=0, zorder=0)


def plot_weekly_overview(
    loader,
    user_id,
    start_date,
    end_date=None,
    *,
    start_hour: int = 12,
    heart_rate_resolution="5min",
    activity_resolution="15min",
    missing_threshold: float = 0.0,
    title: str | None = None,
    figsize=None,
    axes=None,
    show: bool = True,
    save_to=None,
):
    """Plot sleep stages, steps, heart rate, and missingness day by day.

    Each row is one 24-hour participant-local window. Windows begin at
    ``start_hour`` so a noon-to-noon layout keeps the night's sleep together.
    This loader-aware composition intentionally performs the required domain
    loading because the useful view depends on three synchronized streams.

    Parameters
    ----------
    loader
        Loader exposing timestamped activity samples, heart-rate samples, and
        sleep-stage intervals. Loaders with daily-only activity summaries
        cannot produce this subdaily view.
    user_id
        Participant identifier.
    start_date, end_date
        Calendar labels for the first and exclusive final row. If ``end_date``
        is omitted, seven rows are shown.
    start_hour
        Local hour at the left edge of every 24-hour row, from 0 through 23.
    heart_rate_resolution, activity_resolution
        Fixed aggregation windows passed to the corresponding domain APIs.
        Activity bars are totals per ``activity_resolution`` window, which is
        stated explicitly on the activity y-axis.
    missing_threshold
        A window is shaded grey when neither heart-rate nor activity coverage
        is greater than this value. Positive steps can therefore support a
        window even if heart rate is absent.
    title, figsize
        Optional figure title and size. No title is added when ``title`` is
        ``None``. Height grows with the number of rows.
    axes
        Optional sequence containing one primary axis per day. Heart-rate twin
        axes are created by the function.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Returns
    -------
    dict
        ``{"activity": axes, "heart_rate": twin_axes}``.

    Notes
    -----
    Sleep stages are the blue stepped hypnogram near the top of each row. Grey
    spans are explicit intervals without either supported stream, not periods
    of zero movement. Steps are dark bars and heart rate is a vermillion line.
    """
    if not isinstance(start_hour, (int, np.integer)) or not 0 <= start_hour < 24:
        raise ValueError("start_hour must be an integer from 0 through 23")
    if not np.isfinite(missing_threshold) or not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must lie between zero and one")
    if getattr(loader, "activity_source", None) != "samples":
        raise ValueError(
            "plot_weekly_overview requires timestamped activity samples; "
            "this loader exposes daily summaries"
        )

    start = pd.Timestamp(start_date).normalize()
    end = (
        start + pd.Timedelta(days=7)
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    if end <= start:
        raise ValueError("end_date must be after start_date")
    dates = pd.date_range(start, end, inclusive="left")
    physical_start = start + pd.Timedelta(hours=start_hour)
    physical_end = end + pd.Timedelta(hours=start_hour)
    hr_hours = _resolution_hours(heart_rate_resolution)
    step_hours = _resolution_hours(activity_resolution)

    hr = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
        user_id=user_id,
        start_date=physical_start,
        end_date=physical_end,
        resolution=heart_rate_resolution,
        time_reference="local",
    ).xs(user_id, level=schema.USER_COL)
    steps = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        user_id=user_id,
        start_date=physical_start,
        end_date=physical_end,
        resolution=activity_resolution,
        time_reference="local",
    ).xs(user_id, level=schema.USER_COL)
    stages = schema.prepare_sleep_stages(
        loader.load_sleep_stage(
            user_id,
            start_date=start - pd.Timedelta(days=1),
            end_date=end + pd.Timedelta(days=1),
        )
    )

    if axes is None:
        if figsize is None:
            figsize = (12, max(6, len(dates) * 1.45))
        fig, activity_axes = plt.subplots(
            len(dates),
            1,
            figsize=figsize,
            sharex=True,
            squeeze=False,
        )
        activity_axes = activity_axes[:, 0]
    else:
        activity_axes = np.asarray(axes, dtype=object).reshape(-1)
        if len(activity_axes) != len(dates) or not all(
            isinstance(axis, Axes) for axis in activity_axes
        ):
            raise ValueError("axes must contain one Matplotlib axis per day")
        fig = activity_axes[0].figure
        if any(axis.figure is not fig for axis in activity_axes):
            raise ValueError("all overview axes must share a figure")
    heart_axes = []
    stage_axes = []
    hr_feature = heart_rate.HEART_RATE_STATISTIC_MEAN
    step_feature = activity.ACTIVITY_STATISTIC_STEPS

    for row_number, (date, primary_ax) in enumerate(zip(dates, activity_axes)):
        window_start = date + pd.Timedelta(hours=start_hour)
        window_end = window_start + pd.Timedelta(days=1)
        hr_day = hr.loc[(hr.index >= window_start) & (hr.index < window_end)]
        step_day = steps.loc[(steps.index >= window_start) & (steps.index < window_end)]
        hr_x = (hr_day.index - window_start).total_seconds() / 3600
        step_x = (step_day.index - window_start).total_seconds() / 3600

        primary_ax.bar(
            step_x,
            pd.to_numeric(step_day[step_feature], errors="coerce"),
            width=step_hours * 0.82,
            align="edge",
            color=BLACK,
            alpha=0.6,
            linewidth=0,
            zorder=2,
        )
        heart_ax = primary_ax.twinx()
        heart_ax.plot(
            hr_x,
            pd.to_numeric(hr_day[hr_feature], errors="coerce"),
            color=VERMILLION,
            linewidth=1.0,
            zorder=3,
        )
        heart_axes.append(heart_ax)
        stage_ax = primary_ax.twinx()
        stage_ax.spines["right"].set_visible(False)
        stage_ax.spines["left"].set_position(("outward", 52))
        stage_ax.spines["left"].set_visible(True)
        stage_ax.spines["left"].set_color(BLUE)
        stage_ax.spines["left"].set_linewidth(0.8)
        stage_ax.yaxis.set_ticks_position("left")
        stage_ax.yaxis.set_label_position("left")
        stage_ax.set_ylim(-0.35, 3.35)
        stage_ax.set_yticks([0, 1, 2, 3])
        stage_ax.set_yticklabels(["deep", "light", "rem", "wake"])
        stage_ax.tick_params(
            axis="y",
            colors=BLUE,
            labelsize=7,
            length=3,
            width=0.8,
            direction="out",
            pad=3,
        )
        stage_ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
        stage_ax.patch.set_visible(False)
        stage_axes.append(stage_ax)

        hr_coverage = pd.to_numeric(
            hr_day.get(f"{hr_feature}__coverage"), errors="coerce"
        )
        step_coverage = pd.to_numeric(
            step_day.get(f"{step_feature}__coverage"), errors="coerce"
        )
        if len(hr_day):
            aligned_steps = step_coverage.reindex(
                hr_day.index,
                method="ffill",
                tolerance=pd.Timedelta(activity_resolution),
            )
            evidence = pd.concat(
                [hr_coverage.rename("hr"), aligned_steps.rename("steps")],
                axis=1,
            ).max(axis=1, skipna=True)
            missing = evidence.fillna(0).le(missing_threshold)
            _shade_missing_windows(
                primary_ax,
                np.asarray(hr_x),
                missing,
                hr_hours,
            )

        overlapping = stages[
            stages[schema.LOCAL_TIME_COL].lt(window_end)
            & (
                stages[schema.LOCAL_TIME_COL]
                + pd.to_timedelta(stages[schema.DURATION_IN_MS_COL], unit="ms")
            ).gt(window_start)
        ]
        overlapping = overlapping.sort_values(schema.LOCAL_TIME_COL)
        wake_level = SLEEP_STAGE_LEVELS[schema.SLEEP_STAGE_AWAKE_STAGE_VALUE][0]
        stage_x = [0.0]
        stage_y = [wake_level]
        unknown_intervals = []
        current_end = 0.0
        current_level = wake_level
        for stage in overlapping.itertuples(index=False):
            stage_start = max(
                pd.Timestamp(getattr(stage, schema.LOCAL_TIME_COL)), window_start
            )
            stage_end = min(
                pd.Timestamp(getattr(stage, schema.LOCAL_TIME_COL))
                + pd.to_timedelta(getattr(stage, schema.DURATION_IN_MS_COL), unit="ms"),
                window_end,
            )
            left = (stage_start - window_start).total_seconds() / 3600
            right = (stage_end - window_start).total_seconds() / 3600
            stage_type = getattr(stage, schema.SLEEP_STAGE_SLEEP_TYPE_COL)
            if right <= current_end:
                continue
            left = max(left, current_end)
            if left > current_end:
                if current_level != wake_level:
                    stage_x.append(current_end)
                    stage_y.append(wake_level)
                stage_x.append(left)
                stage_y.append(wake_level)
            if stage_type == schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE:
                stage_x.extend([left, np.nan, right])
                stage_y.extend([wake_level, np.nan, wake_level])
                unknown_intervals.append((left, right))
                current_level = wake_level
            else:
                level, _ = SLEEP_STAGE_LEVELS.get(
                    stage_type,
                    SLEEP_STAGE_LEVELS["asleep"],
                )
                stage_x.extend([left, right])
                stage_y.extend([level, level])
                current_level = level
            current_end = right
        if current_end < 24:
            if current_level != wake_level:
                stage_x.append(current_end)
                stage_y.append(wake_level)
            stage_x.append(24.0)
            stage_y.append(wake_level)
        if len(stage_x) > 1:
            transform = blended_transform_factory(
                primary_ax.transData, stage_ax.transData
            )
            primary_ax.plot(
                stage_x,
                stage_y,
                color=BLUE,
                linewidth=1.0,
                transform=transform,
                zorder=5,
            )
            for left, right in unknown_intervals:
                primary_ax.plot(
                    [left, right],
                    [1.5, 1.5],
                    color=GREY,
                    linewidth=3.0,
                    transform=transform,
                    zorder=6,
                )

        primary_ax.set_xlim(0, 24)
        primary_ax.set_ylabel(
            f"Steps / {_resolution_label(activity_resolution)}",
            color=BLACK,
            fontsize=8,
            labelpad=8,
        )
        primary_ax.tick_params(axis="y", colors=BLACK, labelsize=7)
        heart_ax.set_ylabel("HR", color=VERMILLION, fontsize=8)
        heart_ax.tick_params(axis="y", colors=VERMILLION, labelsize=7)
        if not pd.to_numeric(step_day[step_feature], errors="coerce").notna().any():
            primary_ax.set_ylim(0, 1)
            primary_ax.set_yticks([])
        if not pd.to_numeric(hr_day[hr_feature], errors="coerce").notna().any():
            heart_ax.set_ylim(0, 1)
            heart_ax.set_yticks([])
        primary_ax.text(
            0.995,
            0.98,
            date.strftime("%a %d %b"),
            transform=primary_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            weight="bold",
        )
        primary_ax.grid(axis="x", color="#EEEEEE")
        primary_ax.spines[["top", "right"]].set_visible(False)
        heart_ax.spines[["top", "left"]].set_visible(False)
        if row_number == 0:
            primary_ax.tick_params(axis="x", labeltop=True, top=True)
        elif row_number < len(dates) - 1:
            primary_ax.tick_params(axis="x", labelbottom=False)

    clock_ticks = np.arange(0, 25, 3)
    activity_axes[-1].set_xticks(clock_ticks)
    activity_axes[-1].set_xticklabels(
        [f"{(start_hour + hour) % 24:02d}:00" for hour in clock_ticks]
    )
    activity_axes[-1].set_xlabel("Participant-local time")
    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    _finish_figure(fig, show=show, save_to=save_to)
    return {
        "activity": np.asarray(activity_axes),
        "heart_rate": np.asarray(heart_axes),
    }


def _relative_adherence_rows(frame, periods):
    """Return in-study adherence rows with participant-relative day numbers."""
    time_level = _trajectory_time_level(frame)
    rows = frame.reset_index()
    timestamps = pd.to_datetime(rows[time_level])
    starts = rows[schema.USER_COL].map(
        {user: bounds[0] for user, bounds in periods.items()}
    )
    ends = rows[schema.USER_COL].map(
        {user: bounds[1] for user, bounds in periods.items()}
    )
    inside = starts.notna() & timestamps.ge(starts) & timestamps.lt(ends)
    rows = rows.loc[inside].copy()
    timestamps = timestamps.loc[inside]
    starts = starts.loc[inside]
    rows["study_day"] = (
        timestamps.dt.normalize() - starts.dt.normalize()
    ).dt.days.astype(int)
    rows["study_week"] = rows["study_day"] // 7
    rows["weekday"] = timestamps.dt.weekday
    rows["hour"] = timestamps.dt.hour
    return rows


def _daily_adherence_from_hourly(hourly, column):
    """Aggregate equal one-hour local windows to exact daily adherence."""
    time_level = _trajectory_time_level(hourly)
    rows = hourly[[column]].reset_index()
    rows[schema.CALENDAR_DATE_COL] = pd.to_datetime(
        rows[time_level], errors="coerce"
    ).dt.normalize()
    return (
        rows.groupby(
            [schema.USER_COL, schema.CALENDAR_DATE_COL],
            sort=True,
            dropna=False,
        )[column]
        .mean()
        .to_frame()
    )


def _weekly_adherence_counts(rows, column, users, number_of_weeks, threshold):
    """Count qualifying days in each participant-relative study week."""
    counts = np.full((len(users), number_of_weeks), np.nan)
    user_lookup = {value: position for position, value in enumerate(users)}
    for (user, week), group in rows.groupby(
        [schema.USER_COL, "study_week"], sort=False
    ):
        if week >= number_of_weeks:
            continue
        values = pd.to_numeric(group[column], errors="coerce")
        if values.notna().any():
            counts[user_lookup[user], int(week)] = int(values.ge(threshold).sum())
    return counts


def _column_means(matrix):
    """Average finite matrix values by column without empty-slice warnings."""
    valid = np.isfinite(matrix)
    denominator = valid.sum(axis=0)
    return np.divide(
        np.nansum(matrix, axis=0),
        denominator,
        out=np.full(matrix.shape[1], np.nan),
        where=denominator > 0,
    )


def plot_adherence_summary(
    loader,
    *,
    metric: str = adherence.ADHERENCE_HEART_RATE,
    user_id="all",
    follow_up_days: int | None = None,
    worn_day_threshold: float = 0.7,
    include_without_evidence: bool = False,
    title: str | None = None,
    figsize=None,
    axes: Mapping | None = None,
    show: bool = True,
    save_to=None,
):
    """Summarize aligned cohort adherence and nightly availability.

    Every participant is aligned to their first date with primary-stream or
    usable-sleep evidence. The default common follow-up is the longest observed
    trajectory among the requested participants. Shorter trajectories
    contribute trailing non-adherence to that shared denominator.

    Parameters
    ----------
    loader
        Loader used by :func:`pywearable.adherence.get_adherence`.
    metric
        ``"heart_rate"`` or ``"bbi"``. Sleep has no hourly stream and is not
        suitable as the primary wearing-time stream.
    user_id
        One participant, several participants, ``"all"``, or ``None``.
    follow_up_days
        Shared number of participant-relative days to display. With ``None``,
        use the maximum first-to-last evidence span across requested
        participants. A final partial study week is retained.
    worn_day_threshold
        Daily primary-stream adherence required for a day to count as worn.
    include_without_evidence
        Include requested participants with no primary or sleep evidence. They
        are aligned to the earliest inferred cohort start. By default they are
        excluded because their individual study start cannot be inferred.
    title, figsize
        Figure title and optional size. Height grows with cohort size.
    axes
        Optional mapping with axes named ``wearing_days``,
        ``available_nights``, ``weekly_wearing``, ``weekly_nights``,
        ``weekday_pattern``, and ``intraday``.
    show, save_to
        Display the figure and/or save it to a path. Saving does not close it.

    Returns
    -------
    dict
        Mapping of the six panel names to Matplotlib axes.

    Notes
    -----
    Participant-week heatmaps use the exact integer range 0–7. Grey cells mean
    that a stream is unsupported; zero qualifying days or nights is red. The
    final partial week can contain fewer than seven requested days. Weekday and
    intraday panels average raw primary-stream adherence.
    """
    if metric not in {adherence.ADHERENCE_HEART_RATE, adherence.ADHERENCE_BBI}:
        raise ValueError("metric must be 'heart_rate' or 'bbi'")
    if not np.isfinite(worn_day_threshold) or not 0 <= worn_day_threshold <= 1:
        raise ValueError("worn_day_threshold must lie between zero and one")
    if not isinstance(include_without_evidence, (bool, np.bool_)):
        raise ValueError("include_without_evidence must be a boolean")
    periods, follow_up_days, hourly, nights = (
        adherence._prepare_adherence_summary_data(
            loader,
            user_id,
            metric,
            follow_up_days,
            include_without_evidence,
        )
    )
    users = pd.Index(periods)
    column = f"{metric}_adherence"
    daily = _daily_adherence_from_hourly(hourly, column)
    night_column = f"{adherence.ADHERENCE_SLEEP}_adherence"
    daily_rows = _relative_adherence_rows(daily, periods)
    hourly_rows = _relative_adherence_rows(hourly, periods)
    night_rows = _relative_adherence_rows(nights, periods)
    number_of_weeks = int(np.ceil(follow_up_days / 7))
    weeks = np.arange(number_of_weeks)
    wearing_days = _weekly_adherence_counts(
        daily_rows,
        column,
        users,
        number_of_weeks,
        worn_day_threshold,
    )
    available_nights = _weekly_adherence_counts(
        night_rows,
        night_column,
        users,
        number_of_weeks,
        1.0,
    )

    if axes is None:
        if figsize is None:
            figsize = (15, max(9, 6 + len(users) * 0.15))
        fig, created = plt.subplot_mosaic(
            [
                ["wearing_days", "weekly_wearing"],
                ["available_nights", "weekly_nights"],
                ["weekday_pattern", "intraday"],
            ],
            figsize=figsize,
            width_ratios=(1.4, 2.6),
            height_ratios=(1, 1, 1.35),
            constrained_layout=True,
        )
        axes_map = created
    else:
        required = {
            "wearing_days",
            "available_nights",
            "weekly_wearing",
            "weekly_nights",
            "weekday_pattern",
            "intraday",
        }
        missing = required.difference(axes)
        if missing:
            raise ValueError(f"axes is missing panels: {sorted(missing)}")
        axes_map = dict(axes)
        fig = axes_map["wearing_days"].figure

    weekly_cmap = mpl.colors.ListedColormap(
        mpl.colormaps["RdYlGn"](np.linspace(0, 1, 8))
    )
    weekly_cmap.set_bad(LIGHT_GREY)
    weekly_images = []
    for axis_name, matrix, panel_title in (
        ("wearing_days", wearing_days, "A  Worn days by participant"),
        (
            "available_nights",
            available_nights,
            "B  Available nights by participant",
        ),
    ):
        panel = axes_map[axis_name]
        image = panel.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap=weekly_cmap,
            norm=mpl.colors.BoundaryNorm(np.arange(-0.5, 8.5), weekly_cmap.N),
        )
        weekly_images.append(image)
        panel.set(title=panel_title, xlabel="Study week", ylabel="Participant")
        tick_step = max(1, len(weeks) // 10)
        panel.set_xticks(np.arange(len(weeks))[::tick_step])
        panel.set_xticklabels((weeks + 1)[::tick_step])
        if len(users) <= 30:
            panel.set_yticks(np.arange(len(users)))
            panel.set_yticklabels(users)
        else:
            panel.set_yticks([])
    weekly_colorbar = fig.colorbar(
        weekly_images[0],
        ax=[axes_map["wearing_days"], axes_map["available_nights"]],
        ticks=np.arange(8),
        fraction=0.025,
        pad=0.02,
    )
    weekly_colorbar.set_label("Days / nights")

    for axis_name, matrix, panel_title, ylabel in (
        ("weekly_wearing", wearing_days, "C  Cohort mean worn days", "Worn days"),
        (
            "weekly_nights",
            available_nights,
            "D  Cohort mean available nights",
            "Available nights",
        ),
    ):
        panel = axes_map[axis_name]
        panel.plot(
            weeks + 1,
            _column_means(matrix),
            color=BLUE,
            marker="o",
            linewidth=1.8,
        )
        panel.set(
            title=panel_title,
            xlabel="Study week",
            ylabel=ylabel,
            xlim=(0.5, len(weeks) + 0.5),
            ylim=(-0.1, 7.1),
        )
        panel.set_yticks(np.arange(8))
        _style_axis(panel)

    weekday_matrix = np.full((7, len(weeks)), np.nan)
    grouped = daily_rows.groupby(["weekday", "study_week"])[column].mean()
    for (weekday, week), value in grouped.items():
        weekday_matrix[int(weekday), int(week)] = value
    continuous_cmap = mpl.colormaps["RdYlGn"].copy()
    continuous_cmap.set_bad("white")
    weekday_ax = axes_map["weekday_pattern"]
    weekday_ax.imshow(
        np.ma.masked_invalid(weekday_matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=continuous_cmap,
        vmin=0,
        vmax=1,
    )
    weekday_ax.set(
        title="E  Mean adherence by weekday and relative study week",
        xlabel="Study week",
        ylabel="Weekday",
    )
    weekday_ax.set_yticks(np.arange(7))
    weekday_ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    weekday_ax.set_xticks(np.arange(len(weeks))[:: max(1, len(weeks) // 10)])
    weekday_ax.set_xticklabels((weeks + 1)[:: max(1, len(weeks) // 10)])

    days = np.arange(follow_up_days)
    intraday_matrix = np.full((24, len(days)), np.nan)
    grouped = hourly_rows.groupby(["hour", "study_day"])[column].mean()
    for (hour, day), value in grouped.items():
        intraday_matrix[int(hour), int(day)] = value
    intraday_ax = axes_map["intraday"]
    intraday_image = intraday_ax.imshow(
        np.ma.masked_invalid(intraday_matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=continuous_cmap,
        vmin=0,
        vmax=1,
    )
    intraday_ax.set(
        title="F  Mean adherence by local hour and relative study day",
        xlabel="Study day",
        ylabel="Local hour",
    )
    intraday_ax.set_yticks(np.arange(0, 24, 4))
    intraday_ax.set_yticklabels([f"{hour:02d}:00" for hour in range(0, 24, 4)])
    tick_step = max(1, len(days) // 10)
    intraday_ax.set_xticks(np.arange(len(days))[::tick_step])
    intraday_ax.set_xticklabels((days + 1)[::tick_step])
    colorbar = fig.colorbar(
        intraday_image,
        ax=[weekday_ax, intraday_ax],
        fraction=0.02,
        pad=0.02,
    )
    colorbar.set_label("Adherence")
    fig.suptitle(title or f"{metric.replace('_', ' ').title()} adherence overview")
    _finish_figure(fig, show=show, save_to=save_to)
    return axes_map


__all__ = [
    "plot_adherence_summary",
    "plot_calendar_heatmap",
    "plot_coverage_heatmap",
    "plot_hierarchical_trajectory",
    "plot_period_comparison",
    "plot_sleep_heart_rate",
    "plot_sleep_summary",
    "plot_trajectory",
    "plot_trajectory_trend",
    "plot_weekly_profile",
    "plot_weekly_overview",
]
