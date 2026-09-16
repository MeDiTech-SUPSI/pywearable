"""Canonical sleep columns returned by loaders and consumed by sleep features."""

USER_COL = "user_id"
ISODATE_COL = "local_time"
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
