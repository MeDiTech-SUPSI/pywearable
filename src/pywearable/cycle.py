"""Define sleep-based intervals without calculating domain features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pandas as pd

from . import sleep_schema

if TYPE_CHECKING:
    from .loader.base import BaseLoader


CYCLE_NIGHT = "cycle_night"
CYCLE_AWAKE = "cycle_awake"
CYCLE_PHYSIOLOGICAL = "cycle_physiological"
CYCLE_TYPES = {CYCLE_NIGHT, CYCLE_AWAKE, CYCLE_PHYSIOLOGICAL}

INTERVAL_COLUMNS = [
    "start_time",
    "end_time",
    "interval_valid",
    "end_source",
]


def _user_intervals(
    loader: BaseLoader,
    user_id: str,
    cycle_type: str,
    start_date=None,
    end_date=None,
    missing_end: str = "invalidate",
) -> pd.DataFrame:
    if missing_end not in {"invalidate", "midnight"}:
        raise ValueError("missing_end must be 'invalidate' or 'midnight'")

    requested_start = None if start_date is None else pd.Timestamp(start_date).normalize()
    requested_end = None if end_date is None else pd.Timestamp(end_date).normalize()
    if requested_start is not None and requested_end is not None:
        if requested_end <= requested_start:
            raise ValueError("end_date must be after start_date")

    load_start = (
        None
        if requested_start is None
        else requested_start - pd.Timedelta(days=1)
    )
    load_end = (
        None if requested_end is None else requested_end + pd.Timedelta(days=1)
    )
    sessions = sleep_schema.prepare_sleep_sessions(
        loader.load_sleep_sessions(user_id, load_start, load_end)
    )
    sessions = sessions.loc[sessions["main_sleep"]].copy()
    if not sessions.empty:
        # Prefer the longest provider-designated main session if a day has more than one.
        sessions = (
            sessions.sort_values(
                ["calendar_date", "time_in_bed_seconds", "local_time"],
                kind="stable",
                na_position="first",
            )
            .drop_duplicates("calendar_date", keep="last")
            .sort_values("calendar_date", kind="stable")
        )

    if requested_start is None:
        if sessions.empty:
            return pd.DataFrame(columns=INTERVAL_COLUMNS)
        requested_start = sessions["calendar_date"].min()
    if requested_end is None:
        if sessions.empty:
            return pd.DataFrame(columns=INTERVAL_COLUMNS)
        requested_end = sessions["calendar_date"].max() + pd.Timedelta(days=1)

    dates = pd.date_range(requested_start, requested_end, freq="D", inclusive="left")
    indexed = sessions.set_index("calendar_date")
    current = indexed.reindex(dates)
    if cycle_type == CYCLE_NIGHT:
        starts = current["local_time"]
        ends = current["local_end_time"]
        end_sources = pd.Series("session_end", index=dates).where(starts.notna())
    else:
        start_column = (
            "local_end_time" if cycle_type == CYCLE_AWAKE else "local_time"
        )
        starts = current[start_column]
        following_dates = dates + pd.Timedelta(days=1)
        ends = indexed["local_time"].reindex(following_dates)
        ends.index = dates
        end_sources = pd.Series("next_bedtime", index=dates).where(ends.notna())
        if missing_end == "midnight":
            fallback = starts.notna() & ends.isna()
            ends.loc[fallback] = following_dates[fallback]
            end_sources.loc[fallback] = "midnight"

    result = pd.DataFrame(
        {
            "start_time": pd.to_datetime(starts),
            "end_time": pd.to_datetime(ends),
            "end_source": pd.array(end_sources, dtype="string"),
        },
        index=dates,
    )
    result.index.name = "calendar_date"
    result["interval_valid"] = (
        result["start_time"].notna()
        & result["end_time"].notna()
        & result["end_time"].gt(result["start_time"])
    )
    return result[INTERVAL_COLUMNS]


def get_intervals(
    loader: BaseLoader,
    cycle_type: str,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    missing_end: str = "invalidate",
) -> pd.DataFrame:
    """Return expected local-time rows for one requested cycle type.

    ``cycle_night`` is the provider main-sleep session. ``cycle_awake`` runs
    from its getup/session end to the following night's bedtime, and
    ``cycle_physiological`` runs from one bedtime to the following bedtime.
    Missing following nights invalidate the latter two by default. Setting
    ``missing_end='midnight'`` closes a cycle with a known start at the next
    local midnight; it never invents a missing starting boundary. Naps remain
    inside the latter two cycles and do not alter their boundaries.
    """
    if cycle_type not in CYCLE_TYPES:
        raise ValueError(
            f"cycle_type must be one of {sorted(CYCLE_TYPES)}, not {cycle_type!r}"
        )
    if user_id == "all" or user_id is None:
        users = loader.get_user_ids()
    elif isinstance(user_id, str):
        users = [user_id]
    else:
        users = list(user_id)

    frames = {}
    for user in users:
        frame = _user_intervals(
            loader,
            user,
            cycle_type,
            start_date=start_date,
            end_date=end_date,
            missing_end=missing_end,
        )
        if not frame.empty:
            frames[user] = frame
    if not frames:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=["user_id", "calendar_date"]
        )
        return pd.DataFrame(columns=INTERVAL_COLUMNS, index=index)
    return pd.concat(frames, names=["user_id", "calendar_date"]).sort_index()


__all__ = [
    "CYCLE_AWAKE",
    "CYCLE_NIGHT",
    "CYCLE_PHYSIOLOGICAL",
    "CYCLE_TYPES",
    "get_intervals",
]
