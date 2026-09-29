"""Shared hierarchical aggregation for domain feature trajectories.

Domain modules compute the primary feature rows. This module validates the
resolution and aggregation specifications, constructs larger parent windows,
and aggregates each child trajectory with equal child weighting. Child rows
must use a ``(user_id, window_start)`` index and provide ``period_start`` and
``period_end`` columns. Cycle boundaries are supplied by :mod:`pywearable.cycle`.
"""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd

from . import schema
from .cycle import CYCLE_TYPES, get_intervals
from .utils import time as time_utils


STANDARD_AGGREGATIONS = ("mean", "std", "median", "min", "max", "p10", "p90")
CALENDAR_RESOLUTIONS = ("daily", "weekly", "monthly", "yearly")
RESOLUTION_ALIASES = {
    "day": "daily",
    "week": "weekly",
    "month": "monthly",
    "year": "yearly",
    "night": "nightly",
    "natural": "nightly",
}


def canonical_resolution(value) -> str:
    """Return the canonical name for one fixed, calendar, or cycle resolution."""
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        offset = pd.tseries.frequencies.to_offset(pd.Timedelta(value))
        return f"{offset.n}{offset.name}"
    if not isinstance(value, str):
        raise ValueError("each resolution must be a string or timedelta")
    return RESOLUTION_ALIASES.get(value.lower(), value.lower())


def normalize_resolution_chain(resolution) -> tuple[list[str], bool]:
    """Return canonical resolutions and whether multiple levels were requested.

    A sequence is validated to move strictly from smaller to larger windows.
    A scalar or one-item sequence represents a non-hierarchical request.
    """
    if isinstance(resolution, (str, pd.Timedelta, datetime.timedelta)):
        return [canonical_resolution(resolution)], False
    if not isinstance(resolution, Sequence) or not resolution:
        raise ValueError("resolution must be a string or a non-empty sequence")
    chain = [canonical_resolution(value) for value in resolution]
    _validate_resolution_chain(chain)
    return chain, len(chain) > 1


def fixed_duration(resolution: str) -> pd.Timedelta | None:
    """Return a fixed duration, or ``None`` for calendar and cycle windows."""
    if (
        resolution == "nightly"
        or resolution in CYCLE_TYPES
        or resolution in CALENDAR_RESOLUTIONS
    ):
        return None
    try:
        duration = pd.Timedelta(resolution)
    except ValueError as error:
        raise ValueError(f"Resolution {resolution!r} is not supported") from error
    if duration <= pd.Timedelta(0):
        raise ValueError("fixed resolutions must be positive")
    return duration


def _validate_resolution_chain(chain: list[str]) -> None:
    """Reject unsupported or non-increasing multi-resolution chains."""
    fixed = [fixed_duration(value) for value in chain]
    if len(chain) == 1:
        return
    if "nightly" in chain[1:]:
        raise ValueError("nightly can only be the primary sleep resolution")

    calendar_order = {"daily": 0, "weekly": 1, "monthly": 2, "yearly": 3}
    calendar_minimum = {
        "daily": pd.Timedelta(days=1),
        "weekly": pd.Timedelta(days=7),
        "monthly": pd.Timedelta(days=28),
        "yearly": pd.Timedelta(days=365),
    }
    for index, (left, right) in enumerate(zip(chain, chain[1:])):
        left_fixed = fixed[index]
        right_fixed = fixed[index + 1]
        valid = False
        if left_fixed is not None and right_fixed is not None:
            valid = left_fixed < right_fixed
        elif left_fixed is not None and right in CALENDAR_RESOLUTIONS:
            valid = left_fixed < calendar_minimum[right]
        elif left_fixed is not None and right in CYCLE_TYPES:
            valid = left_fixed < pd.Timedelta(days=1)
        elif left in CALENDAR_RESOLUTIONS and right in CALENDAR_RESOLUTIONS:
            valid = calendar_order[left] < calendar_order[right]
        elif left in CYCLE_TYPES and right in {"weekly", "monthly", "yearly"}:
            valid = True
        elif left == "nightly" and right in {"weekly", "monthly", "yearly"}:
            valid = True
        if not valid:
            raise ValueError("resolutions must move from smaller to larger windows")


def is_percentile(operation: str) -> bool:
    """Return whether ``operation`` is a percentile name from ``p0`` to ``p100``."""
    return (
        isinstance(operation, str)
        and operation.startswith("p")
        and operation[1:].isdigit()
        and 0 <= int(operation[1:]) <= 100
    )


def feature_percentile(statistic: str, family: str) -> float | None:
    """Return the quantile for a family-prefixed percentile feature."""
    prefix = f"{family}_"
    if not statistic.startswith(prefix):
        return None
    operation = statistic[len(prefix) :]
    if len(operation) > 4 or not is_percentile(operation):
        return None
    return int(operation[1:]) / 100


def family_statistics(statistics, family: str) -> list[str]:
    """Select validated feature names belonging to one prefixed family."""
    return [name for name in statistics if name.startswith(f"{family}_")]


def _normalize_operations(value, allow_sum: bool = False) -> list[str]:
    """Return a unique ordered list of supported aggregation operation names."""
    if value is None:
        return list(STANDARD_AGGREGATIONS)
    if isinstance(value, str):
        operations = [value]
    elif isinstance(value, (list, tuple)):
        operations = list(value)
    else:
        raise ValueError("aggregation operations must be strings or lists")
    operations = [operation.lower() for operation in operations]
    supported = (*STANDARD_AGGREGATIONS, "sum") if allow_sum else STANDARD_AGGREGATIONS
    if not operations or any(
        operation not in supported and not is_percentile(operation)
        for operation in operations
    ):
        supported_message = (
            "mean, std, median, min, max, sum"
            if allow_sum
            else "mean, std, median, min, or max"
        )
        raise ValueError(
            f"aggregation operations must be {supported_message}, or percentiles "
            "such as p10 and p90"
        )
    return list(dict.fromkeys(operations))


def normalize_aggregation_levels(aggregation, level_count: int):
    """Return one aggregation specification for every level above the primary.

    ``None`` selects :data:`STANDARD_AGGREGATIONS` at each level. For more than
    one upper level, branching operation lists must be nested by level.
    """
    if level_count == 0:
        return []
    if aggregation is None:
        return [None] * level_count
    if level_count == 1:
        return [aggregation]
    if isinstance(aggregation, str):
        return [aggregation] * level_count
    if not isinstance(aggregation, (list, tuple)) or len(aggregation) != level_count:
        raise ValueError(
            "multi-level aggregation must provide one specification per upper level"
        )
    # A nested sequence removes the ambiguity between branching operations and
    # one specification per hierarchy level.
    if all(isinstance(value, str) for value in aggregation):
        raise ValueError(
            "for more than two resolutions, nest operation lists by level"
        )
    return list(aggregation)


def normalize_coverage_thresholds(min_coverage, level_count: int):
    """Return one optional feature-coverage threshold per processing level.

    A scalar applies to every requested resolution. A sequence must contain
    one value for each resolution in the chain, including the primary level.
    ``None`` disables filtering globally or at an individual level.
    """
    if min_coverage is None:
        return [None] * level_count
    if isinstance(min_coverage, Real) and not isinstance(min_coverage, bool):
        thresholds = [float(min_coverage)] * level_count
    elif isinstance(min_coverage, Sequence) and not isinstance(min_coverage, str):
        if len(min_coverage) != level_count:
            raise ValueError(
                "min_coverage must provide one value per resolution"
            )
        thresholds = list(min_coverage)
    else:
        raise ValueError(
            "min_coverage must be None, a number, or a sequence of numbers"
        )

    for threshold in thresholds:
        if threshold is None:
            continue
        if (
            not isinstance(threshold, Real)
            or isinstance(threshold, bool)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("coverage thresholds must be between 0 and 1")
    return [None if value is None else float(value) for value in thresholds]


def normalize_valid_child_thresholds(min_valid_children, level_count: int):
    """Return one optional minimum valid-child count per processing level.

    The primary level has no child windows and therefore always uses ``None``.
    A scalar integer applies to every upper level. A sequence must contain one
    value per resolution, including ``None`` for the primary level.
    """
    if min_valid_children is None:
        return [None] * level_count
    if isinstance(min_valid_children, Integral) and not isinstance(
        min_valid_children, bool
    ):
        if level_count == 1:
            raise ValueError("min_valid_children requires an upper resolution")
        thresholds = [None] + [int(min_valid_children)] * (level_count - 1)
    elif isinstance(min_valid_children, Sequence) and not isinstance(
        min_valid_children, str
    ):
        if len(min_valid_children) != level_count:
            raise ValueError(
                "min_valid_children must provide one value per resolution"
            )
        thresholds = list(min_valid_children)
    else:
        raise ValueError(
            "min_valid_children must be None, an integer, or a sequence of integers"
        )

    if thresholds[0] is not None:
        raise ValueError(
            "the primary resolution has no child windows; its "
            "min_valid_children value must be None"
        )
    for threshold in thresholds[1:]:
        if threshold is None:
            continue
        if (
            not isinstance(threshold, Integral)
            or isinstance(threshold, bool)
            or threshold < 0
        ):
            raise ValueError(
                "min_valid_children thresholds must be non-negative integers"
            )
    return [None if value is None else int(value) for value in thresholds]


def resolve_feature_operations(features, specification, allow_sum: bool = False):
    """Apply one operation specification to every feature at the current level."""
    operations = _normalize_operations(specification, allow_sum=allow_sum)
    return {feature: operations for feature in features}


def warn_if_wide(features, aggregation_levels, threshold: int = 500) -> int:
    """Warn when branching produces many final feature trajectories.

    The returned width excludes metadata and coverage columns.
    """
    width = len(features)
    current = list(features)
    for specification in aggregation_levels:
        operations = resolve_feature_operations(current, specification)
        width = sum(len(values) for values in operations.values())
        current = [
            f"{feature}__{operation}"
            for feature, values in operations.items()
            for operation in values
        ]
    if width > threshold:
        warnings.warn(
            f"This hierarchy will produce {width:,} feature trajectories per user.",
            UserWarning,
            stacklevel=3,
        )
    return width


def aggregate_values(values: pd.Series, operation: str, temporal: bool = False):
    """Aggregate one feature over one parent window.

    Numeric features use ordinary scalar statistics. Temporal features use a
    circular mean and standard deviation; their ordered statistics are
    evaluated on a noon-to-noon clock so times around midnight remain adjacent.
    Temporal standard deviation is returned in minutes.
    """
    values = values.dropna()
    if values.empty:
        return np.nan if operation == "std" or not temporal else None
    if not temporal:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return np.nan
        if is_percentile(operation):
            return numeric.quantile(int(operation[1:]) / 100)
        return getattr(numeric, operation)()

    seconds = np.asarray(
        [time_utils.clock_seconds(value) for value in values], dtype=float
    )
    if operation in {"mean", "std"}:
        angles = seconds / (24 * 60 * 60) * 2 * np.pi
        mean_sine = np.sin(angles).mean()
        mean_cosine = np.cos(angles).mean()
        resultant = np.hypot(mean_sine, mean_cosine)
        if resultant < 1e-12:
            return np.nan if operation == "std" else None
        if operation == "std":
            return np.sqrt(-2 * np.log(resultant)) * (24 * 60) / (2 * np.pi)
        angle = np.arctan2(mean_sine, mean_cosine) % (2 * np.pi)
        return time_utils.clock_time(angle / (2 * np.pi) * 24 * 60 * 60)

    # Ordered sleep times use a noon-to-noon sleep-day clock so values around
    # midnight remain adjacent without allowing calendar dates to order them.
    ordered = np.where(seconds < 12 * 3600, seconds + 24 * 3600, seconds)
    if operation == "median":
        result = np.median(ordered)
    elif operation == "min":
        result = np.min(ordered)
    elif operation == "max":
        result = np.max(ordered)
    elif is_percentile(operation):
        result = np.percentile(ordered, int(operation[1:]))
    else:
        raise ValueError(f"unsupported temporal operation {operation!r}")
    return time_utils.clock_time(result)


def calendar_windows(
    users,
    resolution: str,
    user_bounds: dict,
) -> pd.DataFrame:
    """Construct per-user local calendar windows over exact requested bounds.

    Partial first and last periods are retained. The result is indexed by
    ``(user_id, window_start)`` and contains ``start_time`` and ``end_time``.
    """
    spec = time_utils.get_resampling_spec(resolution, time_reference="local")
    if not spec.calendar_based:
        raise ValueError("upper hierarchy resolutions must be calendar based")
    frames = {}
    for user in users:
        start, end = user_bounds[user]
        starts = time_utils.get_calendar_window_starts(start, end, spec.rule)
        ends = list(starts[1:]) + [pd.Timestamp(end)]
        frames[user] = pd.DataFrame(
            {"start_time": starts, "end_time": ends}, index=starts
        )
        frames[user].index.name = "window_start"
    return pd.concat(frames, names=["user_id", "window_start"])


def fixed_windows(users, resolution: str, user_bounds: dict) -> pd.DataFrame:
    """Construct fixed-duration windows covering each user's requested span.

    The first label is floored to the fixed-resolution grid. Domain extraction
    must ensure that samples outside the exact requested bounds cannot leak into
    the first or last partial window.
    """
    duration = pd.Timedelta(resolution)
    if duration <= pd.Timedelta(0):
        raise ValueError("fixed resolutions must be positive")
    frames = {}
    for user in users:
        start, end = user_bounds[user]
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        starts = pd.date_range(
            start=start.floor(duration),
            end=end,
            freq=duration,
            inclusive="left",
        )
        frames[user] = pd.DataFrame(
            {
                "start_time": starts.where(starts >= start, start),
                "end_time": (starts + duration).where(starts + duration <= end, end),
            },
            index=starts,
        )
        frames[user].index.name = "window_start"
    return pd.concat(frames, names=["user_id", "window_start"])


def windows_for_resolution(users, resolution: str, user_bounds: dict) -> pd.DataFrame:
    """Dispatch to fixed-duration or local-calendar parent-window construction."""
    if resolution in CALENDAR_RESOLUTIONS:
        return calendar_windows(users, resolution, user_bounds)
    return fixed_windows(users, resolution, user_bounds)


def observed_bounds(frame, time_col, start_date, end_date, resolution):
    """Use explicit bounds, or infer complete fixed/calendar periods from data."""
    start = schema.normalize_time_bound(start_date, time_col)
    end = schema.normalize_time_bound(end_date, time_col)
    resolution = canonical_resolution(resolution)
    if resolution in CALENDAR_RESOLUTIONS:
        spec = time_utils.get_resampling_spec(resolution, time_reference="local")

        def period_start(value):
            return time_utils.get_calendar_period_start(value, spec.rule)

        def period_end(value):
            beginning = period_start(value)
            return (
                beginning + pd.Timedelta(days=1)
                if spec.rule == "d"
                else pd.date_range(beginning, periods=2, freq=spec.rule)[1]
            )

    else:
        duration = pd.Timedelta(resolution)

        def period_start(value):
            return value.floor(duration)

        def period_end(value):
            return period_start(value) + duration

    if start is None and not frame.empty:
        start = period_start(frame[time_col].min())
    if end is None and not frame.empty:
        end = period_end(frame[time_col].max())
    if start is None or end is None:
        return None
    if end <= start:
        raise ValueError("end_date must be after start_date")
    return start, end


def empty_trajectory(statistics, include_coverage=False, time_col="window_start"):
    """Return an empty feature table using the shared hierarchy row shape."""
    index = pd.MultiIndex.from_arrays([[], []], names=[schema.USER_COL, time_col])
    columns = ["period_start", "period_end", *statistics]
    if include_coverage:
        columns.extend(f"{name}__coverage" for name in statistics)
    return pd.DataFrame(columns=columns, index=index)


def combine_trajectories(frames, features, include_coverage=False):
    """Merge independently built feature trajectories into one aligned table.

    Rows are matched by their complete ``(user_id, window_start)`` index, so
    data from different participants are never combined. For each matched row,
    ``period_start`` and ``period_end`` are taken from the first trajectory
    that supplies them. Feature and coverage columns are then aligned to the
    union of all rows, leaving missing values where a stream has no matching
    participant/window.
    """
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return empty_trajectory(features, include_coverage)
    index = nonempty[0].index
    for frame in nonempty[1:]:
        index = index.union(frame.index)
    result = pd.DataFrame(index=index)
    for metadata in ("period_start", "period_end"):
        values = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
        for frame in nonempty:
            values = values.fillna(frame[metadata].reindex(index))
        result[metadata] = values
    for frame in frames:
        for column in frame.columns:
            if column not in {"period_start", "period_end"}:
                result[column] = frame[column].reindex(index)
    for feature in features:
        if feature not in result:
            result[feature] = np.nan
        coverage_col = f"{feature}__coverage"
        if include_coverage and coverage_col not in result:
            result[coverage_col] = np.nan
    return result.sort_index()


def resolve_time_col(available, time_reference, chain, signal_name):
    """Choose a supported clock for fixed, calendar, or sleep-cycle windows."""
    if time_reference not in {"auto", "local", "utc"}:
        raise ValueError("time_reference must be 'auto', 'local', or 'utc'")
    if time_reference == "utc" and any(value in CYCLE_TYPES for value in chain):
        raise ValueError(f"sleep-defined {signal_name} cycles require local time")
    local_windows = any(
        value in CYCLE_TYPES or value in CALENDAR_RESOLUTIONS for value in chain
    )
    requested = (
        schema.LOCAL_TIME_COL
        if time_reference == "local" or (time_reference == "auto" and local_windows)
        else schema.UTC_TIME_COL
    )
    if requested in available:
        return requested
    if time_reference == "auto" and schema.LOCAL_TIME_COL in available:
        warnings.warn(
            f"UTC {signal_name} timestamps are unavailable; using local-time windows instead.",
            UserWarning,
            stacklevel=3,
        )
        return schema.LOCAL_TIME_COL
    raise ValueError(
        f"The loader does not provide {signal_name} timestamps in {requested}."
    )


def parent_windows_from_trajectory(
    loader,
    users,
    trajectory,
    target,
    lower,
    start_date=None,
    end_date=None,
    cycle_kwargs=None,
    cycle_intervals=None,
    prefer_explicit_bounds=False,
    cycle_label_bounds=True,
):
    """Build parent windows and choose how children belong to them."""
    if target in CYCLE_TYPES:
        parent = cycle_intervals
        if parent is None:
            parent = get_intervals(
                loader,
                cycle_type=target,
                user_id=users,
                start_date=start_date,
                end_date=end_date,
                **(cycle_kwargs or {}),
            )
        return (
            parent.rename_axis(index={schema.CALENDAR_DATE_COL: "window_start"}),
            "majority_overlap",
        )

    bounds = {}
    available = set(trajectory.index.get_level_values(0))
    for user in users:
        rows = trajectory.xs(user) if user in available else trajectory.iloc[0:0]
        if rows.empty and not (
            prefer_explicit_bounds and start_date is not None and end_date is not None
        ):
            continue
        if lower in CYCLE_TYPES and cycle_label_bounds and not rows.empty:
            lower_bound = pd.Timestamp(rows.index.min()).normalize()
            upper_bound = pd.Timestamp(rows.index.max()).normalize() + pd.Timedelta(
                days=1
            )
        else:
            lower_bound = rows["period_start"].min()
            upper_bound = rows["period_end"].max()
        if prefer_explicit_bounds:
            if start_date is not None:
                lower_bound = pd.Timestamp(start_date)
            if end_date is not None:
                upper_bound = pd.Timestamp(end_date)
            if lower in CYCLE_TYPES:
                lower_bound = lower_bound.normalize()
                upper_bound = upper_bound.normalize()
        if pd.isna(lower_bound) or pd.isna(upper_bound):
            continue
        bounds[user] = (lower_bound, upper_bound)
    if not bounds:
        return None, "label"
    parent = windows_for_resolution(list(bounds), target, bounds)
    membership = (
        "majority_overlap"
        if lower not in CALENDAR_RESOLUTIONS and lower not in CYCLE_TYPES
        else "label"
    )
    return parent, membership


def _majority_overlap_assignments(
    trajectory: pd.DataFrame, parent_windows: pd.DataFrame
) -> pd.Series:
    """Assign each child once when at least half overlaps a parent window."""
    assignments = pd.Series(index=trajectory.index, dtype="object")
    parents_by_user = {
        user: rows for user, rows in parent_windows.groupby(level=0, sort=False)
    }

    for user, children in trajectory.groupby(level=0, sort=False):
        parents = parents_by_user.get(user)
        if parents is None:
            continue
        parents = parents.droplevel(0).dropna(subset=["start_time", "end_time"])
        parents = parents[parents["end_time"] > parents["start_time"]].sort_values(
            "start_time", kind="stable"
        )
        if parents.empty:
            continue

        parent_rows = list(parents.itertuples())
        cursor = 0
        ordered_children = children.sort_values("period_start", kind="stable")
        for child_index, child in ordered_children.iterrows():
            child_start = child["period_start"]
            child_end = child["period_end"]
            if pd.isna(child_start) or pd.isna(child_end) or child_end <= child_start:
                continue

            while (
                cursor < len(parent_rows)
                and parent_rows[cursor].end_time <= child_start
            ):
                cursor += 1

            candidates = []
            position = cursor
            child_duration = child_end - child_start
            midpoint = child_start + child_duration / 2
            while (
                position < len(parent_rows)
                and parent_rows[position].start_time < child_end
            ):
                parent = parent_rows[position]
                overlap = min(child_end, parent.end_time) - max(
                    child_start, parent.start_time
                )
                if overlap > pd.Timedelta(0) and overlap * 2 >= child_duration:
                    contains_midpoint = (
                        parent.start_time <= midpoint < parent.end_time
                    )
                    candidates.append((overlap, contains_midpoint, position, parent))
                position += 1

            if not candidates:
                continue
            maximum_overlap = max(candidate[0] for candidate in candidates)
            candidates = [
                candidate
                for candidate in candidates
                if candidate[0] == maximum_overlap
            ]
            midpoint_candidates = [
                candidate for candidate in candidates if candidate[1]
            ]
            chosen = (midpoint_candidates or candidates)[0]
            assignments.at[child_index] = chosen[3].Index

    return assignments


def aggregate_level(
    trajectory: pd.DataFrame,
    parent_windows: pd.DataFrame,
    target_resolution: str,
    aggregation,
    temporal_features=frozenset(),
    membership: str = "label",
    min_coverage: float | None = None,
    min_valid_children: int | None = None,
    allow_sum: bool = False,
):
    """Aggregate one child trajectory into one larger resolution.

    With ``membership='label'``, child labels select their parent window. With
    ``membership='physical'``, only fully contained children are selected.
    ``membership='majority_overlap'`` assigns each child once when at least
    half of its duration overlaps a parent; exact ties use the parent containing
    the child's midpoint under half-open boundaries. Membership determines the
    expected children, while feature missingness determines which are valid.
    Each valid child has equal weight. Output names append
    ``__{target_resolution}_{operation}``, and each output currently receives
    immediate ``valid / expected`` coverage. When ``min_coverage`` or
    ``min_valid_children`` is set, an output failing either threshold is
    missing while its measured coverage remains available. This missing value
    then propagates as an invalid child at the next hierarchy level.

    Returns the aggregated trajectory and the names of outputs that remain
    temporal for a possible next level.
    """
    if membership not in {"label", "physical", "majority_overlap"}:
        raise ValueError(
            "membership must be 'label', 'physical', or 'majority_overlap'"
        )
    min_coverage = normalize_coverage_thresholds(min_coverage, 1)[0]
    if min_valid_children is not None and (
        not isinstance(min_valid_children, Integral)
        or isinstance(min_valid_children, bool)
        or min_valid_children < 0
    ):
        raise ValueError("min_valid_children must be a non-negative integer")
    metadata = {"period_start", "period_end", "interval_valid", "end_source"}
    features = [
        column
        for column in trajectory.columns
        if column not in metadata and not column.endswith("__coverage")
    ]
    operations = resolve_feature_operations(features, aggregation, allow_sum=allow_sum)
    records = []
    next_temporal = set()
    empty_children = trajectory.iloc[0:0]
    children_by_user = {
        user: rows for user, rows in trajectory.groupby(level=0, sort=False)
    }
    overlap_assignments = (
        _majority_overlap_assignments(trajectory, parent_windows)
        if membership == "majority_overlap"
        else None
    )

    for (user, label), parent in parent_windows.iterrows():
        children = children_by_user.get(user, empty_children)
        if membership == "physical":
            selected = children[
                children["period_start"].ge(parent["start_time"])
                & children["period_end"].le(parent["end_time"])
            ]
        elif membership == "majority_overlap":
            assigned = overlap_assignments.reindex(children.index)
            selected = children[assigned.eq(label).to_numpy()]
        else:
            child_labels = children.index.get_level_values(1)
            selected = children[
                (child_labels >= parent["start_time"])
                & (child_labels < parent["end_time"])
            ]
        expected_count = len(selected)
        record = {
            "user_id": user,
            "window_start": label,
            "period_start": parent["start_time"],
            "period_end": parent["end_time"],
        }
        for feature in features:
            valid_count = int(selected[feature].notna().sum())
            coverage = (
                valid_count / expected_count if expected_count else np.nan
            )
            for operation in operations[feature]:
                output = f"{feature}__{target_resolution}_{operation}"
                is_temporal = feature in temporal_features
                value = aggregate_values(
                    selected[feature], operation, temporal=is_temporal
                )
                if (
                    min_coverage is not None and coverage < min_coverage
                ) or (
                    min_valid_children is not None
                    and valid_count < min_valid_children
                ):
                    value = np.nan
                record[output] = value
                record[f"{output}__coverage"] = coverage
                if is_temporal and operation != "std":
                    next_temporal.add(output)
        records.append(record)

    if not records:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=["user_id", "window_start"]
        )
        return pd.DataFrame(index=index), next_temporal
    result = (
        pd.DataFrame(records)
        .set_index(["user_id", "window_start"])
        .sort_index()
    )
    return result, next_temporal
