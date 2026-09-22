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

import numpy as np
import pandas as pd

from .cycle import CYCLE_TYPES
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


def _resolution_rank(resolution: str) -> float:
    """Return an ordering rank used to validate resolution chains."""
    if resolution == "nightly" or resolution in CYCLE_TYPES:
        return 1.0
    if resolution in CALENDAR_RESOLUTIONS:
        return {"daily": 1.0, "weekly": 2.0, "monthly": 3.0, "yearly": 4.0}[
            resolution
        ]
    try:
        duration = pd.Timedelta(resolution)
    except ValueError as error:
        raise ValueError(f"Resolution {resolution!r} is not supported") from error
    if duration <= pd.Timedelta(0):
        raise ValueError("fixed resolutions must be positive")
    return duration / pd.Timedelta(days=1)


def _validate_resolution_chain(chain: list[str]) -> None:
    """Reject unsupported or non-increasing multi-resolution chains."""
    if len(chain) == 1:
        _resolution_rank(chain[0])
        return
    ranks = [_resolution_rank(value) for value in chain]
    if any(right <= left for left, right in zip(ranks, ranks[1:])):
        raise ValueError("resolutions must move from smaller to larger windows")
    if "nightly" in chain[1:]:
        raise ValueError("nightly can only be the primary sleep resolution")
    for index, value in enumerate(chain):
        if value in CYCLE_TYPES:
            if index == len(chain) - 1:
                continue
            if chain[index + 1] not in {"weekly", "monthly", "yearly"}:
                raise ValueError(
                    "a cycle resolution can only be followed by weekly, monthly, or yearly"
                )


def is_percentile(operation: str) -> bool:
    """Return whether ``operation`` is a percentile name from ``p0`` to ``p100``."""
    return (
        isinstance(operation, str)
        and operation.startswith("p")
        and operation[1:].isdigit()
        and 0 <= int(operation[1:]) <= 100
    )


def _normalize_operations(value) -> list[str]:
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
    if not operations or any(
        operation not in STANDARD_AGGREGATIONS and not is_percentile(operation)
        for operation in operations
    ):
        raise ValueError(
            "aggregation operations must be mean, std, median, min, max, "
            "or percentiles such as p10 and p90"
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


def resolve_feature_operations(features, specification):
    """Apply one operation specification to every feature at the current level."""
    operations = _normalize_operations(specification)
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


def _clock_seconds(value):
    """Convert a timestamp or clock time to seconds since midnight."""
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


def _clock_time(seconds):
    """Convert seconds since midnight to a wrapped :class:`datetime.time`."""
    if pd.isna(seconds):
        return None
    seconds = float(seconds) % (24 * 60 * 60)
    whole = int(round(seconds)) % (24 * 60 * 60)
    hour, remainder = divmod(whole, 3600)
    minute, second = divmod(remainder, 60)
    return datetime.time(hour, minute, second)


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

    seconds = np.asarray([_clock_seconds(value) for value in values], dtype=float)
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
        return _clock_time(angle / (2 * np.pi) * 24 * 60 * 60)

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
    return _clock_time(result)


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
                "start_time": starts,
                "end_time": starts + duration,
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


def aggregate_level(
    trajectory: pd.DataFrame,
    parent_windows: pd.DataFrame,
    target_resolution: str,
    aggregation,
    temporal_features=frozenset(),
    membership: str = "label",
):
    """Aggregate one child trajectory into one larger resolution.

    With ``membership='label'``, child labels select their parent window. With
    ``membership='physical'``, only children fully contained between the
    parent's physical boundaries are selected. Each valid child has equal
    weight. Output names append ``__{target_resolution}_{operation}``, and each
    output currently receives immediate ``valid / expected`` coverage.

    Returns the aggregated trajectory and the names of outputs that remain
    temporal for a possible next level.
    """
    if membership not in {"label", "physical"}:
        raise ValueError("membership must be either 'label' or 'physical'")
    metadata = {"period_start", "period_end", "interval_valid", "end_source"}
    features = [
        column
        for column in trajectory.columns
        if column not in metadata and not column.endswith("__coverage")
    ]
    operations = resolve_feature_operations(features, aggregation)
    records = []
    next_temporal = set()
    empty_children = trajectory.iloc[0:0]
    children_by_user = {
        user: rows for user, rows in trajectory.groupby(level=0, sort=False)
    }

    for (user, label), parent in parent_windows.iterrows():
        children = children_by_user.get(user, empty_children)
        if membership == "physical":
            selected = children[
                children["period_start"].ge(parent["start_time"])
                & children["period_end"].le(parent["end_time"])
            ]
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
            for operation in operations[feature]:
                output = f"{feature}__{target_resolution}_{operation}"
                is_temporal = feature in temporal_features
                record[output] = aggregate_values(
                    selected[feature], operation, temporal=is_temporal
                )
                record[f"{output}__coverage"] = (
                    valid_count / expected_count if expected_count else np.nan
                )
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
