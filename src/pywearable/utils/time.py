from dataclasses import dataclass

import pandas as pd


@dataclass
class ResamplingSpec:
    rule: str
    time_col: str
    calendar_based: bool


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
