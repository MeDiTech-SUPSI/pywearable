"""Canonical sleep columns returned by loaders and consumed by sleep features."""

import pandas as pd

USER_COL = "user_id"
ISODATE_COL = "local_time"
UTC_TIME_COL = "utc_time"
UNIXTIMESTAMP_IN_MS_COL = "unix_timestamp_in_ms"
TIMEZONEOFFSET_IN_MS_COL = "timezone_offset_in_ms"
CALENDAR_DATE_COL = "calendar_date"
SLEEP_SUMMARY_ID_COL = "sleep_summary_id"
DURATION_IN_MS_COL = "duration_in_ms"
IS_SLEEPING_COL = "is_sleeping"
SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL = "overall_sleep_score"
SLEEP_SUMMARY_DURATION_IN_MS_COL = "duration_in_ms"
SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL = "light_sleep_duration_in_ms"
SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL = "deep_sleep_duration_in_ms"
SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL = "rem_sleep_duration_in_ms"
SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL = "awake_duration_in_ms"
SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL = "unmeasurable_sleep_duration_in_ms"
SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL = "sleep_summary_id"
SLEEP_STAGE_SLEEP_TYPE_COL = "type"
SLEEP_STAGE_DURATION_IN_MS_COL = "duration_in_ms"
SLEEP_STAGE_SLEEP_SUMMARY_ID_COL = "sleep_summary_id"
SLEEP_STAGE_REM_STAGE_VALUE = "rem"
SLEEP_STAGE_LIGHT_STAGE_VALUE = "light"
SLEEP_STAGE_DEEP_STAGE_VALUE = "deep"
SLEEP_STAGE_AWAKE_STAGE_VALUE = "awake"
SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE = "unmeasurable"

SLEEP_SUMMARY_REQUIRED_COLUMNS = (
    SLEEP_SUMMARY_ID_COL,
    CALENDAR_DATE_COL,
    ISODATE_COL,
)
SLEEP_SUMMARY_FEATURE_COLUMNS = (
    SLEEP_SUMMARY_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
    SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
)
SLEEP_STAGE_REQUIRED_COLUMNS = (
    SLEEP_STAGE_SLEEP_SUMMARY_ID_COL,
    ISODATE_COL,
    SLEEP_STAGE_DURATION_IN_MS_COL,
    SLEEP_STAGE_SLEEP_TYPE_COL,
)

_COLUMN_MAP = {
    "isoDate": ISODATE_COL,
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
    frame[ISODATE_COL] = pd.to_datetime(frame[ISODATE_COL], errors="coerce")
    if "local_end_time" in frame.columns:
        frame["local_end_time"] = pd.to_datetime(
            frame["local_end_time"], errors="coerce"
        )

    if frame[[CALENDAR_DATE_COL, ISODATE_COL, SLEEP_SUMMARY_ID_COL]].isna().any().any():
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

    frame[ISODATE_COL] = pd.to_datetime(frame[ISODATE_COL], errors="coerce")
    if UTC_TIME_COL in frame.columns:
        frame[UTC_TIME_COL] = pd.to_datetime(frame[UTC_TIME_COL], errors="coerce")
    frame[SLEEP_STAGE_DURATION_IN_MS_COL] = pd.to_numeric(
        frame[SLEEP_STAGE_DURATION_IN_MS_COL], errors="coerce"
    )
    if frame[[SLEEP_STAGE_SLEEP_SUMMARY_ID_COL, ISODATE_COL]].isna().any().any():
        raise ValueError(
            "sleep stage data requires a valid sleep_summary_id and local_time "
            "for every stage"
        )

    return frame.sort_values(
        [SLEEP_STAGE_SLEEP_SUMMARY_ID_COL, ISODATE_COL], kind="stable"
    ).reset_index(drop=True)
