"""Canonical columns and validation shared by loaders and domain modules."""

import numpy as np
import pandas as pd

USER_COL = "user_id"
UTC_TIME_COL = "utc_time"
LOCAL_TIME_COL = "local_time"
TIMEZONE_COL = "timezone"
LOCAL_END_TIME_COL = "local_end_time"
UTC_END_TIME_COL = "utc_end_time"
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

RESPIRATORY_RATE_COL = "respiratory_rate"
RESPIRATORY_RATE_MEAN_COL = "respiratory_rate_mean"
RESPIRATORY_RATE_STD_COL = "respiratory_rate_std"
SPO2_COL = "spo2"
SPO2_MEAN_COL = "spo2_mean"
SPO2_LOWER_BOUND_COL = "spo2_lower_bound"
SPO2_UPPER_BOUND_COL = "spo2_upper_bound"
RESPIRATION_TIME_COLUMNS = (UTC_TIME_COL, LOCAL_TIME_COL)
RESPIRATION_SAMPLE_REQUIRED_COLUMNS = (
    UTC_TIME_COL,
    LOCAL_TIME_COL,
    TIMEZONE_COL,
    RESPIRATORY_RATE_COL,
)
SPO2_SAMPLE_REQUIRED_COLUMNS = (
    UTC_TIME_COL,
    LOCAL_TIME_COL,
    TIMEZONE_COL,
    SPO2_COL,
)

ACTIVITY_STEPS_COL = "activity_steps"
ACTIVITY_TIME_COLUMNS = (UTC_TIME_COL, LOCAL_TIME_COL)
ACTIVITY_SAMPLE_REQUIRED_COLUMNS = (
    UTC_TIME_COL,
    LOCAL_TIME_COL,
    TIMEZONE_COL,
    DURATION_IN_MS_COL,
    ACTIVITY_STEPS_COL,
)
ACTIVITY_SUMMARY_REQUIRED_COLUMNS = (
    CALENDAR_DATE_COL,
    ACTIVITY_STEPS_COL,
)

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


def _empty_frame(column_dtypes, extra_columns=()) -> pd.DataFrame:
    """Return an empty frame with stable canonical dtypes."""
    frame = pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)
            for column, dtype in column_dtypes.items()
        }
    )
    for column in extra_columns:
        if column not in frame:
            frame[column] = pd.Series(dtype=object)
    return frame


def _local_datetimes(values) -> pd.Series:
    """Return timezone-naive participant-local timestamps."""
    return pd.to_datetime(values, errors="coerce", format="mixed").map(
        lambda value: value.tz_localize(None)
        if pd.notna(value) and value.tzinfo is not None
        else value
    )


def _utc_datetimes(values) -> pd.Series:
    """Return timezone-naive UTC timestamps."""
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)


def _timedeltas(values) -> pd.Series:
    """Return canonical UTC offsets, coercing datetime-typed missing values."""
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        if values.notna().any():
            raise ValueError("timezone values must be time offsets, not timestamps")
        return pd.Series(pd.NaT, index=values.index, dtype="timedelta64[ns]")
    return pd.to_timedelta(values, errors="coerce")


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
    return _empty_frame(
        {
            UTC_TIME_COL: "datetime64[ns]",
            LOCAL_TIME_COL: "datetime64[ns]",
            TIMEZONE_COL: "timedelta64[ns]",
            HEART_RATE_COL: "float64",
        },
        extra_columns,
    )


def empty_bbi_frame(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical BBI columns."""
    return _empty_frame(
        {
            UTC_TIME_COL: "datetime64[ns]",
            LOCAL_TIME_COL: "datetime64[ns]",
            TIMEZONE_COL: "timedelta64[ns]",
            BBI_COL: "float64",
        },
        extra_columns,
    )


def empty_respiration_frame(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical respiration columns."""
    return _empty_frame(
        {
            UTC_TIME_COL: "datetime64[ns]",
            LOCAL_TIME_COL: "datetime64[ns]",
            TIMEZONE_COL: "timedelta64[ns]",
            RESPIRATORY_RATE_COL: "float64",
        },
        extra_columns,
    )


def empty_spo2_frame(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical SpO2 columns."""
    return _empty_frame(
        {
            UTC_TIME_COL: "datetime64[ns]",
            LOCAL_TIME_COL: "datetime64[ns]",
            TIMEZONE_COL: "timedelta64[ns]",
            SPO2_COL: "float64",
        },
        extra_columns,
    )


def empty_activity_frame(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical activity-sample columns."""
    return _empty_frame(
        {
            UTC_TIME_COL: "datetime64[ns]",
            LOCAL_TIME_COL: "datetime64[ns]",
            TIMEZONE_COL: "timedelta64[ns]",
            DURATION_IN_MS_COL: "float64",
            ACTIVITY_STEPS_COL: "float64",
        },
        extra_columns,
    )


def empty_activity_summary(extra_columns=()) -> pd.DataFrame:
    """Return an empty table containing the canonical daily-activity columns."""
    return _empty_frame(
        {
            CALENDAR_DATE_COL: "datetime64[ns]",
            ACTIVITY_STEPS_COL: "float64",
        },
        extra_columns,
    )


def prepare_heart_rate_samples(frame) -> pd.DataFrame:
    """Validate and normalize canonical timestamped heart-rate samples.

    Loaders must translate vendor columns before calling this function.
    Duplicate and conflicting exports are preserved so each loader can apply
    its documented reconciliation policy.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("heart-rate data must be a pandas DataFrame")
    frame = frame.copy()
    for column in (UTC_TIME_COL, LOCAL_TIME_COL):
        if column not in frame:
            frame[column] = pd.NaT
    frame[UTC_TIME_COL] = _utc_datetimes(frame[UTC_TIME_COL])
    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    if TIMEZONE_COL not in frame:
        frame[TIMEZONE_COL] = frame[LOCAL_TIME_COL] - frame[UTC_TIME_COL]
    else:
        frame[TIMEZONE_COL] = _timedeltas(frame[TIMEZONE_COL])
    if HEART_RATE_COL not in frame:
        if frame.empty:
            frame[HEART_RATE_COL] = pd.Series(dtype=float)
        else:
            raise ValueError(
                f"heart-rate data is missing required column: {HEART_RATE_COL}"
            )
    frame[HEART_RATE_COL] = pd.to_numeric(frame[HEART_RATE_COL], errors="coerce")
    if TIMESTAMP_CONFLICT_COL in frame:
        frame[TIMESTAMP_CONFLICT_COL] = frame[TIMESTAMP_CONFLICT_COL].astype(
            "boolean"
        )
    frame = frame[
        frame[HEART_RATE_COL].gt(0)
        & (frame[UTC_TIME_COL].notna() | frame[LOCAL_TIME_COL].notna())
    ]
    order_col = (
        UTC_TIME_COL if frame[UTC_TIME_COL].notna().any() else LOCAL_TIME_COL
    )
    return frame.sort_values(order_col, kind="stable").reset_index(drop=True)


def _prepare_physiological_samples(frame, value_col, label):
    """Normalize one timestamped physiological stream into shared columns."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} data must be a pandas DataFrame")
    frame = frame.copy()
    if LOCAL_TIME_COL not in frame.columns:
        frame[LOCAL_TIME_COL] = pd.NaT
    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    if UTC_TIME_COL not in frame.columns:
        frame[UTC_TIME_COL] = pd.NaT
    frame[UTC_TIME_COL] = _utc_datetimes(frame[UTC_TIME_COL])
    if TIMEZONE_COL not in frame.columns:
        frame[TIMEZONE_COL] = frame[LOCAL_TIME_COL] - frame[UTC_TIME_COL]
    else:
        frame[TIMEZONE_COL] = _timedeltas(frame[TIMEZONE_COL])
    if value_col not in frame.columns:
        if frame.empty:
            frame[value_col] = pd.Series(dtype=float)
        else:
            raise ValueError(f"{label} data is missing required column: {value_col}")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    if IS_SLEEPING_COL in frame.columns:
        frame[IS_SLEEPING_COL] = frame[IS_SLEEPING_COL].astype("boolean")
    if CALENDAR_DATE_COL in frame.columns:
        frame[CALENDAR_DATE_COL] = pd.to_datetime(
            frame[CALENDAR_DATE_COL], errors="coerce"
        ).dt.normalize()
    frame = frame[
        frame[value_col].gt(0)
        & (frame[UTC_TIME_COL].notna() | frame[LOCAL_TIME_COL].notna())
    ]
    if value_col == SPO2_COL:
        frame = frame[frame[value_col].le(100)]
    order_col = (
        UTC_TIME_COL if frame[UTC_TIME_COL].notna().any() else LOCAL_TIME_COL
    )
    return frame.sort_values(order_col, kind="stable").reset_index(drop=True)


def prepare_respiration_samples(frame):
    """Validate and normalize timestamped respiratory-rate samples."""
    return _prepare_physiological_samples(
        frame,
        RESPIRATORY_RATE_COL,
        "respiration",
    )


def prepare_spo2_samples(frame):
    """Validate and normalize timestamped oxygen-saturation samples."""
    return _prepare_physiological_samples(
        frame,
        SPO2_COL,
        "SpO2",
    )


def _prepare_provider_summary(frame, value_columns, label):
    """Normalize a canonical summary while preserving additional fields."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} summary must be a pandas DataFrame")
    frame = frame.copy()
    if CALENDAR_DATE_COL not in frame.columns:
        if frame.empty:
            frame[CALENDAR_DATE_COL] = pd.Series(dtype="datetime64[ns]")
        else:
            raise ValueError(
                f"{label} summary is missing required column: {CALENDAR_DATE_COL}"
            )
    frame[CALENDAR_DATE_COL] = _local_datetimes(
        frame[CALENDAR_DATE_COL]
    ).dt.normalize()
    for column in value_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(float("nan"), index=frame.index, dtype=float)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (LOCAL_TIME_COL, LOCAL_END_TIME_COL):
        if column in frame.columns:
            frame[column] = _local_datetimes(frame[column])
    frame = frame[frame[CALENDAR_DATE_COL].notna()].copy()
    for column in value_columns:
        valid = (
            frame[column].ge(0)
            if column == RESPIRATORY_RATE_STD_COL
            else frame[column].gt(0)
        )
        if column in (SPO2_MEAN_COL, SPO2_LOWER_BOUND_COL, SPO2_UPPER_BOUND_COL):
            valid &= frame[column].le(100)
        frame.loc[~valid, column] = float("nan")
    if frame[CALENDAR_DATE_COL].duplicated().any():
        raise ValueError(
            f"{label} summary must contain at most one row per calendar_date"
        )
    return frame.sort_values(CALENDAR_DATE_COL, kind="stable").reset_index(drop=True)


def prepare_respiration_summary(frame):
    """Validate provider-native nightly respiratory-rate summaries."""
    return _prepare_provider_summary(
        frame,
        (RESPIRATORY_RATE_MEAN_COL, RESPIRATORY_RATE_STD_COL),
        "respiration",
    )


def prepare_spo2_summary(frame):
    """Validate provider-native nightly oxygen-saturation summaries."""
    return _prepare_provider_summary(
        frame,
        (SPO2_MEAN_COL, SPO2_LOWER_BOUND_COL, SPO2_UPPER_BOUND_COL),
        "SpO2",
    )


def prepare_activity_samples(frame) -> pd.DataFrame:
    """Normalize canonical timestamped incremental step intervals.

    Each row represents a half-open interval beginning at ``utc_time`` or
    ``local_time`` with ``duration_in_ms`` and an incremental
    ``activity_steps`` count. Zero-step intervals are valid observations.
    Exact duplicates are coalesced; conflicting duplicate intervals retain one
    row with a missing step count.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("activity samples must be a pandas DataFrame")
    frame = frame.copy()
    for column in (UTC_TIME_COL, LOCAL_TIME_COL):
        if column not in frame.columns:
            frame[column] = pd.NaT
    frame[UTC_TIME_COL] = _utc_datetimes(frame[UTC_TIME_COL])
    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    if TIMEZONE_COL not in frame.columns:
        frame[TIMEZONE_COL] = frame[LOCAL_TIME_COL] - frame[UTC_TIME_COL]
    else:
        frame[TIMEZONE_COL] = _timedeltas(frame[TIMEZONE_COL])
    if DURATION_IN_MS_COL not in frame.columns:
        if frame.empty:
            frame[DURATION_IN_MS_COL] = pd.Series(dtype=float)
        else:
            raise ValueError(
                "activity samples are missing required column: duration_in_ms"
            )
    if ACTIVITY_STEPS_COL not in frame.columns:
        if frame.empty:
            frame[ACTIVITY_STEPS_COL] = pd.Series(dtype=float)
        else:
            raise ValueError(
                "activity samples are missing required column: activity_steps"
            )
    frame[DURATION_IN_MS_COL] = pd.to_numeric(
        frame[DURATION_IN_MS_COL], errors="coerce"
    )
    frame[ACTIVITY_STEPS_COL] = pd.to_numeric(
        frame[ACTIVITY_STEPS_COL], errors="coerce"
    )
    frame.loc[frame[ACTIVITY_STEPS_COL].lt(0), ACTIVITY_STEPS_COL] = float("nan")
    frame = frame[
        frame[DURATION_IN_MS_COL].gt(0)
        & (frame[UTC_TIME_COL].notna() | frame[LOCAL_TIME_COL].notna())
    ].copy()
    order_col = "__activity_time"
    frame[order_col] = frame[UTC_TIME_COL].where(
        frame[UTC_TIME_COL].notna(), frame[LOCAL_TIME_COL]
    )
    frame = frame.sort_values(order_col, kind="stable")
    interval_versions = frame.groupby(order_col, dropna=False)[
        [ACTIVITY_STEPS_COL, DURATION_IN_MS_COL]
    ].nunique(dropna=True)
    conflicting_times = interval_versions.index[
        interval_versions.gt(1).any(axis=1)
    ]
    frame.loc[
        frame[order_col].isin(conflicting_times), ACTIVITY_STEPS_COL
    ] = float("nan")
    return (
        frame.drop_duplicates(order_col, keep="last")
        .drop(columns=order_col)
        .reset_index(drop=True)
    )


def prepare_activity_summary(frame) -> pd.DataFrame:
    """Normalize canonical calendar-day step totals."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("activity summary must be a pandas DataFrame")
    frame = frame.copy()
    if CALENDAR_DATE_COL not in frame.columns:
        if frame.empty:
            frame[CALENDAR_DATE_COL] = pd.Series(dtype="datetime64[ns]")
        else:
            raise ValueError(
                "activity summary is missing required column: calendar_date"
            )
    if ACTIVITY_STEPS_COL not in frame.columns:
        if frame.empty:
            frame[ACTIVITY_STEPS_COL] = pd.Series(dtype=float)
        else:
            raise ValueError(
                "activity summary is missing required column: activity_steps"
            )
    frame[CALENDAR_DATE_COL] = _local_datetimes(
        frame[CALENDAR_DATE_COL]
    ).dt.normalize()
    frame[ACTIVITY_STEPS_COL] = pd.to_numeric(
        frame[ACTIVITY_STEPS_COL], errors="coerce"
    )
    frame.loc[frame[ACTIVITY_STEPS_COL].lt(0), ACTIVITY_STEPS_COL] = float("nan")
    frame = frame[frame[CALENDAR_DATE_COL].notna()].copy()
    frame = frame.drop_duplicates([CALENDAR_DATE_COL, ACTIVITY_STEPS_COL], keep="last")
    if frame[CALENDAR_DATE_COL].duplicated().any():
        raise ValueError(
            "activity summary must contain at most one step total per calendar_date"
        )
    return frame.sort_values(CALENDAR_DATE_COL, kind="stable").reset_index(drop=True)


def prepare_provider_feature_samples(frame, feature: str) -> pd.DataFrame:
    """Validate canonical timestamped samples for one provider-native feature.

    Loaders must translate vendor columns before calling this function. Exact
    timestamp duplicates collapse to one row. If repeated rows disagree about
    the feature value, one row is retained with that value missing; downstream
    coverage therefore records the conflict without selecting an arbitrary
    export revision.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("provider feature samples must be a pandas DataFrame")
    frame = frame.copy()
    if feature not in frame.columns:
        if frame.empty:
            frame[feature] = pd.Series(dtype=float)
        else:
            raise ValueError(
                f"provider feature samples are missing required column: {feature}"
            )
    for column in (UTC_TIME_COL, LOCAL_TIME_COL):
        if column not in frame.columns:
            frame[column] = pd.NaT
    frame[UTC_TIME_COL] = _utc_datetimes(frame[UTC_TIME_COL])
    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    if TIMEZONE_COL not in frame.columns:
        frame[TIMEZONE_COL] = frame[LOCAL_TIME_COL] - frame[UTC_TIME_COL]
    else:
        frame[TIMEZONE_COL] = _timedeltas(frame[TIMEZONE_COL])
    frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame = frame[
        frame[UTC_TIME_COL].notna() | frame[LOCAL_TIME_COL].notna()
    ].copy()
    order_col = "__provider_feature_time"
    frame[order_col] = frame[UTC_TIME_COL].where(
        frame[UTC_TIME_COL].notna(), frame[LOCAL_TIME_COL]
    )
    frame = frame.sort_values(order_col, kind="stable")
    conflicts = frame.groupby(order_col, dropna=False)[feature].transform(
        lambda values: values.dropna().nunique() > 1
    )
    frame.loc[conflicts, feature] = float("nan")
    return (
        frame.drop_duplicates(order_col, keep="last")
        .drop(columns=order_col)
        .reset_index(drop=True)
    )


def prepare_provider_feature_summary(
    frame, feature: str, *, numeric: bool = True
) -> pd.DataFrame:
    """Validate one canonical provider feature at native daily grain.

    The input must already use ``calendar_date`` and the canonical feature
    name. The loader must resolve exact export duplicates or revisions first;
    this validator rejects any remaining duplicate date rather than choosing
    one provider value silently.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("provider feature summary must be a pandas DataFrame")
    frame = frame.copy()
    if CALENDAR_DATE_COL not in frame.columns:
        if frame.empty:
            frame[CALENDAR_DATE_COL] = pd.Series(dtype="datetime64[ns]")
        else:
            raise ValueError(
                "provider feature summary is missing required column: "
                f"{CALENDAR_DATE_COL}"
            )
    if feature not in frame.columns:
        if frame.empty:
            frame[feature] = pd.Series(dtype=float if numeric else object)
        else:
            raise ValueError(
                f"provider feature summary is missing required column: {feature}"
            )
    frame[CALENDAR_DATE_COL] = _local_datetimes(
        frame[CALENDAR_DATE_COL]
    ).dt.normalize()
    frame = frame[frame[CALENDAR_DATE_COL].notna()].copy()
    if numeric:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    if frame[CALENDAR_DATE_COL].duplicated().any():
        raise ValueError(
            "provider feature summary must contain at most one row per "
            "calendar_date"
        )
    return frame.sort_values(CALENDAR_DATE_COL, kind="stable").reset_index(drop=True)


def prepare_bbi(frame):
    """Validate and normalize canonical beat-to-beat interval samples."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("BBI data must be a pandas DataFrame")
    frame = frame.copy()
    for column in (UTC_TIME_COL, LOCAL_TIME_COL):
        if column not in frame:
            frame[column] = pd.NaT
    frame[UTC_TIME_COL] = _utc_datetimes(frame[UTC_TIME_COL])
    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    if TIMEZONE_COL not in frame.columns:
        frame[TIMEZONE_COL] = frame[LOCAL_TIME_COL] - frame[UTC_TIME_COL]
    else:
        frame[TIMEZONE_COL] = _timedeltas(frame[TIMEZONE_COL])
    if BBI_COL not in frame:
        if frame.empty:
            frame[BBI_COL] = pd.Series(dtype=float)
        else:
            raise ValueError(f"BBI data is missing required column: {BBI_COL}")
    frame[BBI_COL] = pd.to_numeric(frame[BBI_COL], errors="coerce")
    frame = frame[
        np.isfinite(frame[BBI_COL]) & frame[BBI_COL].gt(0)
        & (frame[UTC_TIME_COL].notna() | frame[LOCAL_TIME_COL].notna())
    ]
    physical_times = frame[UTC_TIME_COL].fillna(
        frame[LOCAL_TIME_COL] - frame[TIMEZONE_COL]
    )
    keys = pd.DataFrame({
        "time": physical_times.fillna(frame[LOCAL_TIME_COL]),
        "utc_known": physical_times.notna(),
        "bbi": frame[BBI_COL],
    })
    frame = frame.loc[~keys.duplicated()]
    return frame.sort_values(
        UTC_TIME_COL if frame[UTC_TIME_COL].notna().any() else LOCAL_TIME_COL,
        kind="stable",
    ).reset_index(drop=True)


def prepare_sleep_summary(frame):
    """Validate and normalize a canonical nightly sleep summary."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sleep summary data must be a pandas DataFrame")

    frame = frame.copy()
    if frame.empty:
        required_dtypes = {
            SLEEP_SUMMARY_ID_COL: object,
            CALENDAR_DATE_COL: "datetime64[ns]",
            LOCAL_TIME_COL: "datetime64[ns]",
            **{column: "float64" for column in SLEEP_SUMMARY_FEATURE_COLUMNS},
        }
        empty = _empty_frame(required_dtypes)
        for column in frame:
            if column not in empty:
                empty[column] = frame[column]
        return empty

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
        frame.loc[frame[column].lt(0), column] = float("nan")

    frame[CALENDAR_DATE_COL] = _local_datetimes(
        frame[CALENDAR_DATE_COL]
    ).dt.normalize()
    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    if UTC_TIME_COL in frame.columns:
        frame[UTC_TIME_COL] = _utc_datetimes(frame[UTC_TIME_COL])
    if LOCAL_END_TIME_COL in frame.columns:
        frame[LOCAL_END_TIME_COL] = _local_datetimes(frame[LOCAL_END_TIME_COL])
    if UTC_END_TIME_COL in frame.columns:
        frame[UTC_END_TIME_COL] = _utc_datetimes(frame[UTC_END_TIME_COL])

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
    """Validate and normalize canonical sleep-stage intervals."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sleep stage data must be a pandas DataFrame")

    frame = frame.copy()
    if frame.empty:
        required_dtypes = {
            SLEEP_SUMMARY_ID_COL: object,
            LOCAL_TIME_COL: "datetime64[ns]",
            DURATION_IN_MS_COL: "float64",
            SLEEP_STAGE_SLEEP_TYPE_COL: object,
        }
        empty = _empty_frame(required_dtypes)
        for column in frame:
            if column not in empty:
                empty[column] = frame[column]
        return empty

    missing = [
        column for column in SLEEP_STAGE_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "sleep stage data is missing required columns: " + ", ".join(missing)
        )

    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    if UTC_TIME_COL in frame.columns:
        frame[UTC_TIME_COL] = _utc_datetimes(frame[UTC_TIME_COL])
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
    return _coalesce_sleep_stage_runs(frame)


def prepare_sleep_sessions(frame):
    """Validate and order sleep sessions used by full-day trajectories."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sleep session data must be a pandas DataFrame")

    frame = frame.copy()
    if frame.empty:
        return _empty_frame(
            {
                "id": object,
                CALENDAR_DATE_COL: "datetime64[ns]",
                "type": object,
                "main_sleep": "bool",
                LOCAL_TIME_COL: "datetime64[ns]",
                LOCAL_END_TIME_COL: "datetime64[ns]",
                "time_in_bed_seconds": "float64",
            }
        )

    missing = [
        column for column in SLEEP_SESSION_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "sleep session data is missing required columns: " + ", ".join(missing)
        )

    frame[CALENDAR_DATE_COL] = _local_datetimes(
        frame[CALENDAR_DATE_COL]
    ).dt.normalize()
    frame[LOCAL_TIME_COL] = _local_datetimes(frame[LOCAL_TIME_COL])
    frame[LOCAL_END_TIME_COL] = _local_datetimes(frame[LOCAL_END_TIME_COL])
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


def _coalesce_sleep_stage_runs(frame):
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


__all__ = [
    "ACTIVITY_SAMPLE_REQUIRED_COLUMNS",
    "ACTIVITY_STEPS_COL",
    "ACTIVITY_SUMMARY_REQUIRED_COLUMNS",
    "ACTIVITY_TIME_COLUMNS",
    "BBI_COL",
    "BBI_REQUIRED_COLUMNS",
    "BBI_TIME_COLUMNS",
    "CALENDAR_DATE_COL",
    "DURATION_IN_MS_COL",
    "HEART_RATE_COL",
    "HEART_RATE_REQUIRED_COLUMNS",
    "HEART_RATE_TIME_COLUMNS",
    "HRV_COVERAGE_COL",
    "HRV_HF_POWER_COL",
    "HRV_LF_POWER_COL",
    "HRV_RMSSD_COL",
    "HRV_VLF_POWER_COL",
    "IS_SLEEPING_COL",
    "LOCAL_END_TIME_COL",
    "LOCAL_TIME_COL",
    "RESPIRATION_SAMPLE_REQUIRED_COLUMNS",
    "RESPIRATION_TIME_COLUMNS",
    "RESPIRATORY_RATE_COL",
    "RESPIRATORY_RATE_MEAN_COL",
    "RESPIRATORY_RATE_STD_COL",
    "SLEEP_SESSION_COLUMNS",
    "SLEEP_SESSION_REQUIRED_COLUMNS",
    "SLEEP_STAGE_ALLOWED_VALUES",
    "SLEEP_STAGE_AWAKE_STAGE_VALUE",
    "SLEEP_STAGE_DEEP_STAGE_VALUE",
    "SLEEP_STAGE_LIGHT_STAGE_VALUE",
    "SLEEP_STAGE_REM_STAGE_VALUE",
    "SLEEP_STAGE_REQUIRED_COLUMNS",
    "SLEEP_STAGE_SLEEP_TYPE_COL",
    "SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE",
    "SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL",
    "SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL",
    "SLEEP_SUMMARY_FEATURE_COLUMNS",
    "SLEEP_SUMMARY_ID_COL",
    "SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL",
    "SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL",
    "SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL",
    "SLEEP_SUMMARY_REQUIRED_COLUMNS",
    "SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL",
    "SPO2_COL",
    "SPO2_LOWER_BOUND_COL",
    "SPO2_MEAN_COL",
    "SPO2_SAMPLE_REQUIRED_COLUMNS",
    "SPO2_UPPER_BOUND_COL",
    "TIMESTAMP_CONFLICT_COL",
    "TIMEZONE_COL",
    "USER_COL",
    "UTC_END_TIME_COL",
    "UTC_TIME_COL",
    "empty_activity_frame",
    "empty_activity_summary",
    "empty_bbi_frame",
    "empty_heart_rate_frame",
    "empty_respiration_frame",
    "empty_spo2_frame",
    "filter_heart_rate_by_time",
    "normalize_time_bound",
    "prepare_activity_samples",
    "prepare_activity_summary",
    "prepare_bbi",
    "prepare_heart_rate_samples",
    "prepare_provider_feature_samples",
    "prepare_provider_feature_summary",
    "prepare_respiration_samples",
    "prepare_respiration_summary",
    "prepare_sleep_sessions",
    "prepare_sleep_stages",
    "prepare_sleep_summary",
    "prepare_spo2_samples",
    "prepare_spo2_summary",
    "validate_heart_rate_time_col",
]
