"""Continuous local-time sleep/wake trajectories and regularity metrics."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from . import sleep_schema
from .loader.base import BaseLoader


TRAJECTORY_COLUMNS = [
    "calendar_date",
    "state",
    "sleep_stage",
    "observed",
    "state_coverage",
    "state_source",
    "session_id",
    "main_sleep",
    "session_type",
]


def _fixed_resolution(resolution: str) -> pd.Timedelta:
    try:
        value = pd.Timedelta(resolution)
    except (TypeError, ValueError) as error:
        raise ValueError("resolution must be a fixed pandas timedelta") from error
    if value <= pd.Timedelta(0):
        raise ValueError("resolution must be positive")
    return value


def _normalize_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "id",
        "calendar_date",
        "type",
        "main_sleep",
        "local_time",
        "local_end_time",
        "time_in_bed_seconds",
    ]
    if not isinstance(sessions, pd.DataFrame):
        raise ValueError("sleep session data must be a pandas DataFrame")
    if sessions.empty:
        return pd.DataFrame(columns=columns)
    missing = set(columns[:-1]).difference(sessions.columns)
    if missing:
        raise ValueError(
            "sleep session data is missing required columns: "
            + ", ".join(sorted(missing))
        )
    sessions = sessions.copy()
    sessions["calendar_date"] = pd.to_datetime(
        sessions["calendar_date"], errors="coerce"
    ).dt.normalize()
    sessions["local_time"] = pd.to_datetime(
        sessions["local_time"], errors="coerce"
    )
    sessions["local_end_time"] = pd.to_datetime(
        sessions["local_end_time"], errors="coerce"
    )
    if sessions[["id", "calendar_date", "local_time", "local_end_time"]].isna().any().any():
        raise ValueError("sleep sessions require valid identifiers and local boundaries")
    if sessions["local_end_time"].le(sessions["local_time"]).any():
        raise ValueError("sleep session end times must follow their start times")
    sessions["main_sleep"] = sessions["main_sleep"].astype(bool)
    if "time_in_bed_seconds" not in sessions:
        sessions["time_in_bed_seconds"] = (
            sessions["local_end_time"] - sessions["local_time"]
        ).dt.total_seconds()
    return (
        sessions[columns]
        .drop_duplicates(subset="id", keep="last")
        .sort_values("local_time", kind="stable")
        .reset_index(drop=True)
    )


def _empty_trajectory(user_id: str) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [[], []], names=[sleep_schema.USER_COL, sleep_schema.ISODATE_COL]
    )
    return pd.DataFrame(columns=TRAJECTORY_COLUMNS, index=index)


def _bin_slice(
    grid_start: pd.Timestamp,
    resolution: pd.Timedelta,
    size: int,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> tuple[int, int]:
    start = max(interval_start, grid_start)
    grid_end = grid_start + size * resolution
    end = min(interval_end, grid_end)
    if end <= start:
        return 0, 0
    first = max(0, int(np.floor((start - grid_start) / resolution)))
    last = min(size, int(np.ceil((end - grid_start) / resolution)))
    return first, last


def _user_sleep_wake_trajectory(
    loader: BaseLoader,
    user_id: str,
    start_date=None,
    end_date=None,
    resolution: str = "1min",
    include_naps: bool = True,
) -> pd.DataFrame:
    resolution_delta = _fixed_resolution(resolution)
    requested_start = None if start_date is None else pd.Timestamp(start_date)
    requested_end = None if end_date is None else pd.Timestamp(end_date)
    if requested_start is not None and requested_end is not None:
        if requested_end <= requested_start:
            raise ValueError("end_date must be after start_date")

    load_start = None
    load_end = None
    if requested_start is not None:
        load_start = requested_start - pd.Timedelta(days=1)
    if requested_end is not None:
        load_end = requested_end + pd.Timedelta(days=1)
    sessions = _normalize_sessions(
        loader.load_sleep_sessions(user_id, load_start, load_end)
    )
    stages = sleep_schema.prepare_sleep_stages(
        loader.load_sleep_session_stages(user_id, load_start, load_end)
    )

    if not include_naps and not sessions.empty:
        main_ids = set(sessions.loc[sessions["main_sleep"], "id"])
        sessions = sessions[sessions["id"].isin(main_ids)].reset_index(drop=True)
        stages = stages[
            stages[sleep_schema.SLEEP_SUMMARY_ID_COL].isin(main_ids)
        ].reset_index(drop=True)

    available_starts = []
    available_ends = []
    if not sessions.empty:
        available_starts.append(sessions["local_time"].min())
        available_ends.append(sessions["local_end_time"].max())
    if not stages.empty:
        available_starts.append(stages[sleep_schema.ISODATE_COL].min())
        available_ends.append(
            (
                stages[sleep_schema.ISODATE_COL]
                + pd.to_timedelta(
                    stages[sleep_schema.SLEEP_STAGE_DURATION_IN_MS_COL], unit="ms"
                )
            ).max()
        )
    if requested_start is None:
        if not available_starts:
            return _empty_trajectory(user_id)
        requested_start = min(available_starts).normalize()
    if requested_end is None:
        if not available_ends:
            return _empty_trajectory(user_id)
        requested_end = max(available_ends).normalize() + pd.Timedelta(days=1)

    grid = pd.date_range(
        requested_start, requested_end, freq=resolution_delta, inclusive="left"
    )
    size = len(grid)
    if size == 0:
        return _empty_trajectory(user_id)
    bin_ms = resolution_delta.total_seconds() * 1000
    sleep_ms = np.zeros(size, dtype=float)
    awake_ms = np.zeros(size, dtype=float)
    inferred_awake = np.zeros(size, dtype=bool)
    staged_ms = np.zeros(size, dtype=float)
    stage_overlap: dict[str, np.ndarray] = {}
    session_for_bin = np.full(size, None, dtype=object)

    main_sessions = sessions[sessions["main_sleep"]].sort_values(
        "calendar_date", kind="stable"
    )
    for current, following in zip(
        main_sessions.itertuples(index=False),
        main_sessions.iloc[1:].itertuples(index=False),
    ):
        if following.calendar_date - current.calendar_date != pd.Timedelta(days=1):
            continue
        if following.local_time < current.local_end_time:
            continue
        first, last = _bin_slice(
            requested_start,
            resolution_delta,
            size,
            current.local_end_time,
            following.local_time,
        )
        for index in range(first, last):
            bin_start = grid[index]
            bin_end = bin_start + resolution_delta
            overlap = min(bin_end, following.local_time) - max(
                bin_start, current.local_end_time
            )
            if overlap > pd.Timedelta(0):
                awake_ms[index] += overlap.total_seconds() * 1000
                inferred_awake[index] = True

    # A recorded session interrupts inferred wake. Its state remains missing
    # unless the provider supplies stage intervals for it. Subtract only the
    # exact overlap so coarse bins can retain observed wake outside a session.
    for session in sessions.itertuples(index=False):
        first, last = _bin_slice(
            requested_start,
            resolution_delta,
            size,
            session.local_time,
            session.local_end_time,
        )
        for index in range(first, last):
            bin_start = grid[index]
            bin_end = bin_start + resolution_delta
            overlap = min(bin_end, session.local_end_time) - max(
                bin_start, session.local_time
            )
            if overlap <= pd.Timedelta(0):
                continue
            awake_ms[index] = max(
                0,
                awake_ms[index] - overlap.total_seconds() * 1000,
            )
            inferred_awake[index] = awake_ms[index] > 0

    session_lookup = sessions.set_index("id") if not sessions.empty else sessions
    for stage in stages.itertuples(index=False):
        if hasattr(stage, "local_time_uncertain") and stage.local_time_uncertain:
            continue
        stage_start = getattr(stage, sleep_schema.ISODATE_COL)
        stage_end = stage_start + pd.to_timedelta(
            getattr(stage, sleep_schema.SLEEP_STAGE_DURATION_IN_MS_COL), unit="ms"
        )
        session_id = getattr(stage, sleep_schema.SLEEP_STAGE_SLEEP_SUMMARY_ID_COL)
        if not sessions.empty and session_id in session_lookup.index:
            session = session_lookup.loc[session_id]
            stage_start = max(stage_start, session["local_time"])
            stage_end = min(stage_end, session["local_end_time"])
            if stage_end <= stage_start:
                continue
        stage_type = getattr(stage, sleep_schema.SLEEP_STAGE_SLEEP_TYPE_COL)
        if stage_type == sleep_schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE:
            continue
        state = (
            "awake"
            if stage_type == sleep_schema.SLEEP_STAGE_AWAKE_STAGE_VALUE
            else "sleep"
        )
        if stage_type not in stage_overlap:
            stage_overlap[stage_type] = np.zeros(size, dtype=float)
        first, last = _bin_slice(
            requested_start, resolution_delta, size, stage_start, stage_end
        )
        for index in range(first, last):
            bin_start = grid[index]
            bin_end = bin_start + resolution_delta
            overlap = min(bin_end, stage_end) - max(bin_start, stage_start)
            if overlap <= pd.Timedelta(0):
                continue
            overlap_ms = overlap.total_seconds() * 1000
            if state == "sleep":
                sleep_ms[index] += overlap_ms
            else:
                awake_ms[index] += overlap_ms
            staged_ms[index] += overlap_ms
            stage_overlap[stage_type][index] += overlap_ms
            session_for_bin[index] = session_id

    state = np.full(size, None, dtype=object)
    state[sleep_ms > awake_ms] = "sleep"
    state[awake_ms > sleep_ms] = "awake"
    observed_ms = np.maximum(sleep_ms, awake_ms)
    coverage = np.minimum(100, observed_ms / bin_ms * 100)
    source = np.full(size, None, dtype=object)
    has_state = pd.notna(state)
    source[has_state & inferred_awake & (staged_ms == 0)] = "between_main_sleeps"
    source[has_state & ~inferred_awake & (staged_ms > 0)] = "stage"
    source[has_state & inferred_awake & (staged_ms > 0)] = "mixed"

    detailed_stage = np.full(size, None, dtype=object)
    if stage_overlap:
        stage_names = list(stage_overlap)
        overlap_matrix = np.vstack([stage_overlap[name] for name in stage_names])
        winners = overlap_matrix.argmax(axis=0)
        has_stage = overlap_matrix.max(axis=0) > 0
        detailed_stage[has_stage] = np.asarray(stage_names, dtype=object)[
            winners[has_stage]
        ]
    detailed_stage[(state == "awake") & pd.isna(detailed_stage)] = "awake"

    main_sleep = np.full(size, pd.NA, dtype=object)
    session_type = np.full(size, None, dtype=object)
    if not sessions.empty:
        for index, session_id in enumerate(session_for_bin):
            if session_id is None or session_id not in session_lookup.index:
                continue
            session = session_lookup.loc[session_id]
            main_sleep[index] = bool(session["main_sleep"])
            session_type[index] = session["type"]

    result = pd.DataFrame(
        {
            "calendar_date": grid.normalize(),
            "state": pd.array(state, dtype="string"),
            "sleep_stage": pd.array(detailed_stage, dtype="string"),
            "observed": pd.notna(state),
            "state_coverage": coverage,
            "state_source": pd.array(source, dtype="string"),
            "session_id": pd.array(session_for_bin, dtype="string"),
            "main_sleep": pd.array(main_sleep, dtype="boolean"),
            "session_type": pd.array(session_type, dtype="string"),
        },
        index=grid,
    )
    result.index.name = sleep_schema.ISODATE_COL
    return pd.concat(
        {user_id: result}, names=[sleep_schema.USER_COL, sleep_schema.ISODATE_COL]
    )


def get_sleep_wake_trajectory(
    loader: BaseLoader,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    resolution: str = "1min",
    include_naps: bool = True,
) -> pd.DataFrame:
    """Return a regular local-time sleep/wake trajectory.

    Wake between consecutive main-sleep sessions is inferred. Recorded
    session intervals are classified only where stage data are available;
    unmeasurable intervals and gaps remain missing. Non-main sessions are
    included by default when the loader exposes them.
    """
    if user_id == "all" or user_id is None:
        users = loader.get_user_ids()
    elif isinstance(user_id, str):
        users = [user_id]
    else:
        users = list(user_id)
    trajectories = [
        _user_sleep_wake_trajectory(
            loader,
            user,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
            include_naps=include_naps,
        )
        for user in users
    ]
    trajectories = [trajectory for trajectory in trajectories if not trajectory.empty]
    if not trajectories:
        return _empty_trajectory("")
    return pd.concat(trajectories).sort_index()


def get_sleep_regularity(
    loader: BaseLoader,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    resolution: str = "1min",
    include_naps: bool = True,
) -> pd.DataFrame:
    """Return daily SRI and the percentage of comparable 24-hour epochs."""
    resolution_delta = _fixed_resolution(resolution)
    periods_per_day = pd.Timedelta(days=1) / resolution_delta
    if periods_per_day != int(periods_per_day):
        raise ValueError("resolution must divide evenly into 24 hours for SRI")
    periods_per_day = int(periods_per_day)

    trajectory_start = (
        None
        if start_date is None
        else pd.Timestamp(start_date).normalize() - pd.Timedelta(days=1)
    )
    trajectory_end = (
        None if end_date is None else pd.Timestamp(end_date).normalize()
    )
    trajectory = get_sleep_wake_trajectory(
        loader,
        user_id=user_id,
        start_date=trajectory_start,
        end_date=trajectory_end,
        resolution=resolution,
        include_naps=include_naps,
    )
    records = []
    for user, user_trajectory in trajectory.groupby(level=sleep_schema.USER_COL):
        states = user_trajectory.droplevel(sleep_schema.USER_COL)["state"]
        if states.empty:
            continue
        first_date = states.index.min().normalize()
        last_date = states.index.max().normalize()
        requested_first = (
            first_date + pd.Timedelta(days=1)
            if start_date is None
            else pd.Timestamp(start_date).normalize()
        )
        requested_last = (
            last_date + pd.Timedelta(days=1)
            if end_date is None
            else pd.Timestamp(end_date).normalize()
        )
        for day in pd.date_range(
            requested_first, requested_last, freq="D", inclusive="left"
        ):
            current_index = pd.date_range(
                day, day + pd.Timedelta(days=1), freq=resolution_delta, inclusive="left"
            )
            current = states.reindex(current_index)
            previous = states.reindex(current_index - pd.Timedelta(days=1))
            previous.index = current_index
            comparable = current.notna() & previous.notna()
            comparable_count = int(comparable.sum())
            sri = np.nan
            if comparable_count:
                agreement = current[comparable].eq(previous[comparable]).mean()
                sri = -100 + 200 * agreement
            records.append(
                {
                    sleep_schema.USER_COL: user,
                    sleep_schema.CALENDAR_DATE_COL: day,
                    "SRI": sri,
                    "sri_coverage": comparable_count / periods_per_day * 100,
                }
            )
    if not records:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=[sleep_schema.USER_COL, sleep_schema.CALENDAR_DATE_COL]
        )
        return pd.DataFrame(columns=["SRI", "sri_coverage"], index=index)
    return (
        pd.DataFrame(records)
        .set_index([sleep_schema.USER_COL, sleep_schema.CALENDAR_DATE_COL])
        .sort_index()
    )
