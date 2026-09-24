"""Canonical columns and validation shared by loaders and domain modules."""

import pandas as pd

USER_COL = "user_id"
UTC_TIME_COL = "utc_time"
LOCAL_TIME_COL = "local_time"
TIMEZONE_COL = "timezone"
LOCAL_END_TIME_COL = "local_end_time"
UTC_END_TIME_COL = "utc_end_time"
UNIXTIMESTAMP_IN_MS_COL = "unix_timestamp_in_ms"
TIMEZONEOFFSET_IN_MS_COL = "timezone_offset_in_ms"
CALENDAR_DATE_COL = "calendar_date"
DURATION_IN_MS_COL = "duration_in_ms"
IS_SLEEPING_COL = "is_sleeping"

HEART_RATE_COL = "heart_rate"
TIMESTAMP_CONFLICT_COL = "timestamp_conflict"
HEART_RATE_TIME_COLUMNS = (UTC_TIME_COL, LOCAL_TIME_COL)
HEART_RATE_REQUIRED_COLUMNS = (
    UTC_TIME_COL,
    LOCAL_TIME_COL,
    TIMEZONE_COL,
    HEART_RATE_COL,
)

BBI_COL = "bbi"
BBI_TIME_COLUMNS = (UTC_TIME_COL, LOCAL_TIME_COL)
BBI_REQUIRED_COLUMNS = (
    UTC_TIME_COL,
    LOCAL_TIME_COL,
    TIMEZONE_COL,
    BBI_COL,
)

HRV_RMSSD_COL = "hrv_rmssd"
HRV_VLF_POWER_COL = "hrv_vlf_power"
HRV_LF_POWER_COL = "hrv_lf_power"
HRV_HF_POWER_COL = "hrv_hf_power"
HRV_COVERAGE_COL = "hrv_coverage"

SLEEP_SUMMARY_ID_COL = "sleep_summary_id"
SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL = "overall_sleep_score"
SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL = "light_sleep_duration_in_ms"
SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL = "deep_sleep_duration_in_ms"
SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL = "rem_sleep_duration_in_ms"
SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL = "awake_duration_in_ms"
SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL = "unmeasurable_sleep_duration_in_ms"
SLEEP_STAGE_SLEEP_TYPE_COL = "type"
SLEEP_STAGE_REM_STAGE_VALUE = "rem"
SLEEP_STAGE_LIGHT_STAGE_VALUE = "light"
SLEEP_STAGE_DEEP_STAGE_VALUE = "deep"
SLEEP_STAGE_AWAKE_STAGE_VALUE = "awake"
SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE = "unmeasurable"
SLEEP_STAGE_ALLOWED_VALUES = {
    SLEEP_STAGE_REM_STAGE_VALUE,
    SLEEP_STAGE_LIGHT_STAGE_VALUE,
    SLEEP_STAGE_DEEP_STAGE_VALUE,
    SLEEP_STAGE_AWAKE_STAGE_VALUE,
    SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
    "asleep",
}

SLEEP_SESSION_COLUMNS = (
    "id",
    CALENDAR_DATE_COL,
    "type",
    "main_sleep",
    LOCAL_TIME_COL,
    LOCAL_END_TIME_COL,
    "time_in_bed_seconds",
)
SLEEP_SESSION_REQUIRED_COLUMNS = SLEEP_SESSION_COLUMNS[:-1]

SLEEP_SUMMARY_REQUIRED_COLUMNS = (
    SLEEP_SUMMARY_ID_COL,
    CALENDAR_DATE_COL,
    LOCAL_TIME_COL,
)
SLEEP_SUMMARY_FEATURE_COLUMNS = (
    DURATION_IN_MS_COL,
    SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
)
SLEEP_STAGE_REQUIRED_COLUMNS = (
    SLEEP_SUMMARY_ID_COL,
    LOCAL_TIME_COL,
    DURATION_IN_MS_COL,
    SLEEP_STAGE_SLEEP_TYPE_COL,
)

_COLUMN_MAP = {
    "isoDate": LOCAL_TIME_COL,
    "unixTimestampInMs": UNIXTIMESTAMP_IN_MS_COL,
    "timezoneOffsetInMs": TIMEZONEOFFSET_IN_MS_COL,
    "calendarDate": CALENDAR_DATE_COL,
    "sleepSummaryId": SLEEP_SUMMARY_ID_COL,
    "durationInMs": DURATION_IN_MS_COL,
    "isSleeping": IS_SLEEPING_COL,
    "overallSleepScore": SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL,
    "lightSleepDurationInMs": SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
    "deepSleepDurationInMs": SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
    "remSleepDurationInMs": SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
    "awakeDurationInMs": SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
    "unmeasurableSleepDurationInMs": SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
}


def validate_heart_rate_time_col(time_col: str) -> str:
    """Validate the timestamp used to filter heart-rate samples."""
    if time_col not in HEART_RATE_TIME_COLUMNS:
        raise ValueError(
            f"time_col must be one of {HEART_RATE_TIME_COLUMNS}, not {time_col}"
        )
    return time_col


def normalize_time_bound(value, time_col: str):
    """Return a timezone-naive bound in the selected clock."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        if time_col == UTC_TIME_COL:
            timestamp = timestamp.tz_convert("UTC")
        timestamp = timestamp.tz_localize(None)
    return timestamp


def filter_heart_rate_by_time(
    frame: pd.DataFrame, start_date=None, end_date=None, time_col: str = UTC_TIME_COL
) -> pd.DataFrame:
    """Filter canonical heart-rate samples over ``[start_date, end_date)``."""
    validate_heart_rate_time_col(time_col)
    start = normalize_time_bound(start_date, time_col)
    end = normalize_time_bound(end_date, time_col)
    if start is not None:
        frame = frame[frame[time_col] >= start]
    if end is not None:
        frame = frame[frame[time_col] < end]
    return frame


def empty_heart_rate_frame(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical heart-rate columns."""
    return pd.DataFrame(columns=[*HEART_RATE_REQUIRED_COLUMNS, *extra_columns])


def empty_bbi_frame(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical BBI columns."""
    return pd.DataFrame(columns=[*BBI_REQUIRED_COLUMNS, *extra_columns])


def prepare_bbi(frame):
    """Validate and order beat-to-beat intervals in the shared schema."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("BBI data must be a pandas DataFrame")
    frame = frame.copy()
    aliases = {
        "isoDate": LOCAL_TIME_COL,
        "timezoneOffsetInMs": TIMEZONEOFFSET_IN_MS_COL,
        "unixTimestampInMs": UNIXTIMESTAMP_IN_MS_COL,
    }
    frame = frame.rename(
        columns={source: target for source, target in aliases.items() if target not in frame}
    )
    if "timezone_offset" in frame.columns and TIMEZONE_COL not in frame.columns:
        frame = frame.rename(columns={"timezone_offset": TIMEZONE_COL})
    if UTC_TIME_COL not in frame.columns and UNIXTIMESTAMP_IN_MS_COL in frame.columns:
        frame[UTC_TIME_COL] = pd.to_datetime(
            frame[UNIXTIMESTAMP_IN_MS_COL], unit="ms", errors="coerce", utc=True
        ).dt.tz_localize(None)
    if LOCAL_TIME_COL in frame.columns:
        frame[LOCAL_TIME_COL] = pd.to_datetime(
            frame[LOCAL_TIME_COL], errors="coerce", format="mixed"
        ).map(
            lambda value: value.tz_localize(None)
            if pd.notna(value) and value.tzinfo is not None
            else value
        )
    if TIMEZONE_COL not in frame.columns:
        if TIMEZONEOFFSET_IN_MS_COL in frame.columns:
            frame[TIMEZONE_COL] = pd.to_timedelta(
                pd.to_numeric(frame[TIMEZONEOFFSET_IN_MS_COL], errors="coerce"),
                unit="ms",
            )
        elif UTC_TIME_COL in frame.columns and LOCAL_TIME_COL in frame.columns:
            frame[TIMEZONE_COL] = (
                pd.to_datetime(frame[LOCAL_TIME_COL], errors="coerce")
                - pd.to_datetime(frame[UTC_TIME_COL], errors="coerce")
            )
    if frame.empty:
        for column in BBI_REQUIRED_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame
    missing = [column for column in BBI_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("BBI data is missing required columns: " + ", ".join(missing))
    frame[UTC_TIME_COL] = pd.to_datetime(
        frame[UTC_TIME_COL], errors="coerce", utc=True
    ).dt.tz_localize(None)
    frame[LOCAL_TIME_COL] = pd.to_datetime(frame[LOCAL_TIME_COL], errors="coerce")
    frame[BBI_COL] = pd.to_numeric(frame[BBI_COL], errors="coerce")
    frame = frame[
        frame[BBI_COL].gt(0)
        & (frame[UTC_TIME_COL].notna() | frame[LOCAL_TIME_COL].notna())
    ]
    return frame.sort_values(
        UTC_TIME_COL if frame[UTC_TIME_COL].notna().any() else LOCAL_TIME_COL,
        kind="stable",
    ).reset_index(drop=True)


def normalize_sleep_columns(frame):
    """Normalize a provider's sleep output without altering its source data."""
    return frame.rename(
        columns={name: _COLUMN_MAP[name] for name in frame.columns if name in _COLUMN_MAP}
    )


def prepare_sleep_summary(frame):
    """Validate and order a loader result for nightly feature extraction."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sleep summary data must be a pandas DataFrame")

    frame = normalize_sleep_columns(frame.copy())
    if frame.empty:
        return frame

    missing = [
        column for column in SLEEP_SUMMARY_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "sleep summary data is missing required columns: " + ", ".join(missing)
        )

    for column in SLEEP_SUMMARY_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(float("nan"), index=frame.index, dtype=float)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[CALENDAR_DATE_COL] = pd.to_datetime(
        frame[CALENDAR_DATE_COL], errors="coerce"
    ).dt.normalize()
    frame[LOCAL_TIME_COL] = pd.to_datetime(frame[LOCAL_TIME_COL], errors="coerce")
    if UTC_TIME_COL in frame.columns:
        frame[UTC_TIME_COL] = pd.to_datetime(
            frame[UTC_TIME_COL], errors="coerce", utc=True
        ).dt.tz_localize(None)
    elif UNIXTIMESTAMP_IN_MS_COL in frame.columns:
        frame[UTC_TIME_COL] = pd.to_datetime(
            frame[UNIXTIMESTAMP_IN_MS_COL], unit="ms", errors="coerce", utc=True
        ).dt.tz_localize(None)
    if LOCAL_END_TIME_COL in frame.columns:
        frame[LOCAL_END_TIME_COL] = pd.to_datetime(
            frame[LOCAL_END_TIME_COL], errors="coerce"
        )
    if UTC_END_TIME_COL in frame.columns:
        frame[UTC_END_TIME_COL] = pd.to_datetime(
            frame[UTC_END_TIME_COL], errors="coerce", utc=True
        ).dt.tz_localize(None)

    if frame[[CALENDAR_DATE_COL, LOCAL_TIME_COL, SLEEP_SUMMARY_ID_COL]].isna().any().any():
        raise ValueError(
            "sleep summary data requires a valid calendar_date, local_time, "
            "and sleep_summary_id for every night"
        )
    if frame[CALENDAR_DATE_COL].duplicated().any():
        duplicates = frame.loc[
            frame[CALENDAR_DATE_COL].duplicated(keep=False), CALENDAR_DATE_COL
        ].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "sleep summary data must contain one row per calendar_date; duplicates: "
            + ", ".join(duplicates.unique())
        )
    if frame[SLEEP_SUMMARY_ID_COL].duplicated().any():
        raise ValueError("sleep_summary_id must uniquely identify each night")

    return frame.sort_values(CALENDAR_DATE_COL, kind="stable").reset_index(drop=True)


def prepare_sleep_stages(frame):
    """Validate and order a loader result for sleep-stage feature extraction."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sleep stage data must be a pandas DataFrame")

    frame = normalize_sleep_columns(frame.copy())
    if frame.empty:
        for column in SLEEP_STAGE_REQUIRED_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame

    missing = [
        column for column in SLEEP_STAGE_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "sleep stage data is missing required columns: " + ", ".join(missing)
        )

    frame[LOCAL_TIME_COL] = pd.to_datetime(frame[LOCAL_TIME_COL], errors="coerce")
    if UTC_TIME_COL in frame.columns:
        frame[UTC_TIME_COL] = pd.to_datetime(
            frame[UTC_TIME_COL], errors="coerce", utc=True
        ).dt.tz_localize(None)
    elif UNIXTIMESTAMP_IN_MS_COL in frame.columns:
        frame[UTC_TIME_COL] = pd.to_datetime(
            frame[UNIXTIMESTAMP_IN_MS_COL], unit="ms", errors="coerce", utc=True
        ).dt.tz_localize(None)
    frame[DURATION_IN_MS_COL] = pd.to_numeric(
        frame[DURATION_IN_MS_COL], errors="coerce"
    )
    if frame[
        [
            SLEEP_SUMMARY_ID_COL,
            LOCAL_TIME_COL,
            DURATION_IN_MS_COL,
            SLEEP_STAGE_SLEEP_TYPE_COL,
        ]
    ].isna().any().any():
        raise ValueError(
            "sleep stage data requires a valid sleep_summary_id, local_time, "
            "duration_in_ms, and type for every stage"
        )
    if frame[DURATION_IN_MS_COL].le(0).any():
        raise ValueError("sleep stage duration_in_ms must be positive")
    unknown_types = set(frame[SLEEP_STAGE_SLEEP_TYPE_COL].unique()).difference(
        SLEEP_STAGE_ALLOWED_VALUES
    )
    if unknown_types:
        raise ValueError(
            "sleep stage type contains unsupported values: "
            + ", ".join(sorted(map(str, unknown_types)))
        )

    frame = frame.sort_values(
        [SLEEP_SUMMARY_ID_COL, LOCAL_TIME_COL], kind="stable"
    ).reset_index(drop=True)
    return coalesce_sleep_stage_runs(frame)


def prepare_sleep_sessions(frame):
    """Validate and order sleep sessions used by full-day trajectories."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sleep session data must be a pandas DataFrame")

    frame = frame.copy()
    if frame.empty:
        for column in SLEEP_SESSION_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame[list(SLEEP_SESSION_COLUMNS)]

    missing = [
        column for column in SLEEP_SESSION_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "sleep session data is missing required columns: " + ", ".join(missing)
        )

    frame[CALENDAR_DATE_COL] = pd.to_datetime(
        frame[CALENDAR_DATE_COL], errors="coerce"
    ).dt.normalize()
    frame[LOCAL_TIME_COL] = pd.to_datetime(frame[LOCAL_TIME_COL], errors="coerce")
    frame[LOCAL_END_TIME_COL] = pd.to_datetime(
        frame[LOCAL_END_TIME_COL], errors="coerce"
    )
    required_values = ["id", CALENDAR_DATE_COL, LOCAL_TIME_COL, LOCAL_END_TIME_COL]
    if frame[required_values].isna().any().any():
        raise ValueError(
            "sleep sessions require valid identifiers, calendar dates, and local boundaries"
        )
    if frame[LOCAL_END_TIME_COL].le(frame[LOCAL_TIME_COL]).any():
        raise ValueError("sleep session end times must follow their start times")
    if frame["id"].duplicated().any():
        raise ValueError("sleep session ids must be unique")
    if not frame["main_sleep"].isin([True, False]).all():
        raise ValueError("sleep session main_sleep values must be boolean")
    frame["main_sleep"] = frame["main_sleep"].astype(bool)

    reconstructed_time_in_bed = (
        frame[LOCAL_END_TIME_COL] - frame[LOCAL_TIME_COL]
    ).dt.total_seconds()
    if "time_in_bed_seconds" not in frame.columns:
        frame["time_in_bed_seconds"] = reconstructed_time_in_bed
    else:
        frame["time_in_bed_seconds"] = pd.to_numeric(
            frame["time_in_bed_seconds"], errors="coerce"
        ).fillna(reconstructed_time_in_bed)

    return (
        frame[list(SLEEP_SESSION_COLUMNS)]
        .sort_values(LOCAL_TIME_COL, kind="stable")
        .reset_index(drop=True)
    )


def coalesce_sleep_stage_runs(frame):
    """Merge exactly adjacent intervals with the same stage label."""
    if frame.empty:
        return frame

    frame = frame.copy()
    order_time = frame[LOCAL_TIME_COL].copy()
    if UTC_TIME_COL in frame.columns:
        complete_utc = frame[UTC_TIME_COL].notna().groupby(
            frame[SLEEP_SUMMARY_ID_COL]
        ).transform("all")
        order_time.loc[complete_utc] = frame.loc[complete_utc, UTC_TIME_COL]
    frame["__sleep_order_time"] = order_time
    frame = frame.sort_values(
        [SLEEP_SUMMARY_ID_COL, "__sleep_order_time"], kind="stable"
    )
    end_time = frame["__sleep_order_time"] + pd.to_timedelta(
        frame[DURATION_IN_MS_COL], unit="ms"
    )
    starts_new_run = (
        frame[SLEEP_SUMMARY_ID_COL].ne(
            frame[SLEEP_SUMMARY_ID_COL].shift()
        )
        | frame[SLEEP_STAGE_SLEEP_TYPE_COL].ne(
            frame[SLEEP_STAGE_SLEEP_TYPE_COL].shift()
        )
        | frame["__sleep_order_time"].ne(end_time.shift())
    )
    run_id = starts_new_run.cumsum()
    result = frame.loc[starts_new_run].copy()
    result[DURATION_IN_MS_COL] = frame.groupby(run_id, sort=False)[
        DURATION_IN_MS_COL
    ].sum().to_numpy()
    return result.drop(columns="__sleep_order_time").reset_index(drop=True)
