import pandas as pd

from pywearable import schema


def test_bbi_cleaning_rejects_nonfinite_values_and_deduplicates_pairs():
    times = pd.to_datetime(
        ["2024-01-01", "2024-01-01", "2024-01-01",
         "2024-01-01 00:00:01", "2024-01-01 00:00:02",
         "2024-01-01 00:00:03", "2024-01-01 00:00:04"], format="mixed",
    )
    frame = pd.DataFrame({
        "utc_time": times,
        "bbi": [800, 800, 900, 800, float("inf"), float("-inf"), 0],
    })
    result = schema.prepare_bbi(frame)
    assert result.bbi.tolist() == [800, 900, 800]
    assert result.utc_time.tolist() == times[[0, 2, 3]].tolist()


def test_bbi_deduplication_uses_offsets_without_confusing_repeated_local_times():
    frame = pd.DataFrame({
        "utc_time": pd.to_datetime(["2024-10-27 00:30", None, None]),
        "local_time": pd.to_datetime(["2024-10-27 02:30"] * 3),
        "timezone": pd.to_timedelta([2, 2, 1], unit="h"),
        "bbi": [800, 800, 800],
    })
    result = schema.prepare_bbi(frame)
    assert len(result) == 2
    assert result.timezone.tolist() == [pd.Timedelta(hours=2), pd.Timedelta(hours=1)]


def test_bbi_local_only_duplicates_keep_distinct_timestamps():
    frame = pd.DataFrame({
        "local_time": pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-01-01 00:00:01"], format="mixed",
        ),
        "bbi": [800, 800, 800],
    })
    result = schema.prepare_bbi(frame)
    assert len(result) == 2


def test_empty_stream_frames_have_stable_canonical_dtypes():
    frames = [
        schema.empty_heart_rate_frame(),
        schema.empty_bbi_frame(),
        schema.empty_respiration_frame(),
        schema.empty_spo2_frame(),
        schema.empty_activity_frame(),
    ]

    for frame in frames:
        assert pd.api.types.is_datetime64_ns_dtype(frame[schema.UTC_TIME_COL])
        assert pd.api.types.is_datetime64_ns_dtype(frame[schema.LOCAL_TIME_COL])
        assert pd.api.types.is_timedelta64_dtype(frame[schema.TIMEZONE_COL])

    assert pd.api.types.is_float_dtype(
        frames[0][schema.HEART_RATE_COL]
    )
    assert pd.api.types.is_float_dtype(frames[1][schema.BBI_COL])
    assert pd.api.types.is_float_dtype(
        frames[2][schema.RESPIRATORY_RATE_COL]
    )
    assert pd.api.types.is_float_dtype(frames[3][schema.SPO2_COL])
    assert pd.api.types.is_float_dtype(
        frames[4][schema.ACTIVITY_STEPS_COL]
    )


def test_prepare_heart_rate_samples_normalizes_without_reconciling_duplicates():
    frame = pd.DataFrame(
        {
            schema.UTC_TIME_COL: [
                "2024-01-01T22:00:00Z",
                "2024-01-01T22:00:00Z",
                "2024-01-01T23:00:00Z",
            ],
            schema.LOCAL_TIME_COL: [
                "2024-01-01T23:00:00+01:00",
                "2024-01-01T23:00:00+01:00",
                "2024-01-02T00:00:00+01:00",
            ],
            schema.HEART_RATE_COL: [60, 60, -1],
            "source": "test",
        }
    )

    result = schema.prepare_heart_rate_samples(frame)

    assert len(result) == 2
    assert result[schema.HEART_RATE_COL].tolist() == [60, 60]
    assert result[schema.LOCAL_TIME_COL].dt.tz is None
    assert result[schema.UTC_TIME_COL].dt.tz is None
    assert result[schema.TIMEZONE_COL].eq(pd.Timedelta(hours=1)).all()
    assert result["source"].tolist() == ["test", "test"]


def test_provider_feature_samples_normalize_supplied_timezone():
    frame = pd.DataFrame(
        {
            schema.UTC_TIME_COL: ["2024-01-01T22:00:00Z"],
            schema.LOCAL_TIME_COL: ["2024-01-01T23:00:00+01:00"],
            schema.TIMEZONE_COL: ["0 days 01:00:00"],
            "feature": [3],
        }
    )

    result = schema.prepare_provider_feature_samples(frame, "feature")

    assert result.loc[0, schema.TIMEZONE_COL] == pd.Timedelta(hours=1)
    assert pd.api.types.is_timedelta64_dtype(result[schema.TIMEZONE_COL])


def test_empty_sleep_preparers_return_complete_typed_schemas():
    summaries = schema.prepare_sleep_summary(pd.DataFrame())
    stages = schema.prepare_sleep_stages(pd.DataFrame())
    sessions = schema.prepare_sleep_sessions(pd.DataFrame())

    assert set(schema.SLEEP_SUMMARY_REQUIRED_COLUMNS) <= set(summaries)
    assert set(schema.SLEEP_SUMMARY_FEATURE_COLUMNS) <= set(summaries)
    assert pd.api.types.is_datetime64_ns_dtype(
        summaries[schema.CALENDAR_DATE_COL]
    )
    assert pd.api.types.is_datetime64_ns_dtype(
        summaries[schema.LOCAL_TIME_COL]
    )
    assert pd.api.types.is_float_dtype(summaries[schema.DURATION_IN_MS_COL])

    assert tuple(stages) == schema.SLEEP_STAGE_REQUIRED_COLUMNS
    assert pd.api.types.is_datetime64_ns_dtype(stages[schema.LOCAL_TIME_COL])
    assert pd.api.types.is_float_dtype(stages[schema.DURATION_IN_MS_COL])

    assert tuple(sessions) == schema.SLEEP_SESSION_COLUMNS
    assert pd.api.types.is_bool_dtype(sessions["main_sleep"])
    assert pd.api.types.is_float_dtype(sessions["time_in_bed_seconds"])


def test_prepare_sleep_summary_normalizes_local_time_and_invalid_duration():
    frame = pd.DataFrame(
        {
            schema.SLEEP_SUMMARY_ID_COL: ["night-1"],
            schema.CALENDAR_DATE_COL: ["2024-01-02T00:00:00+01:00"],
            schema.LOCAL_TIME_COL: ["2024-01-01T23:00:00+01:00"],
            schema.DURATION_IN_MS_COL: [-1],
        }
    )

    result = schema.prepare_sleep_summary(frame)

    assert result.loc[0, schema.CALENDAR_DATE_COL] == pd.Timestamp("2024-01-02")
    assert result.loc[0, schema.LOCAL_TIME_COL] == pd.Timestamp(
        "2024-01-01 23:00"
    )
    assert pd.isna(result.loc[0, schema.DURATION_IN_MS_COL])
