from types import SimpleNamespace

import pandas as pd
import pytest

from pywearable import sleep, sleep_schema


def test_legacy_sleep_columns_are_normalized():
    raw = pd.DataFrame(
        {
            "calendarDate": ["2024-01-02"],
            "sleepSummaryId": ["night-1"],
            "durationInMs": [1],
            "REMValue": [2],
        }
    )
    result = sleep_schema.normalize_sleep_columns(raw)
    assert list(result) == [
        "calendar_date",
        "sleep_summary_id",
        "duration_in_ms",
        "REMValue",
    ]
    assert list(raw) == ["calendarDate", "sleepSummaryId", "durationInMs", "REMValue"]


def test_sleep_trajectory_preserves_feature_names_and_keeps_missing_score():
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-2"],
            "calendar_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "local_time": pd.to_datetime(["2024-01-01 23:00", "2024-01-02 23:00"]),
            "duration_in_ms": [60 * 60_000, 60 * 60_000],
            "awake_duration_in_ms": [0, 0],
            "deep_sleep_duration_in_ms": [20 * 60_000, 20 * 60_000],
            "light_sleep_duration_in_ms": [30 * 60_000, 30 * 60_000],
            "rem_sleep_duration_in_ms": [10 * 60_000, 10 * 60_000],
            "overall_sleep_score": [80, None],
        }
    )
    stages = pd.DataFrame(columns=["sleep_summary_id", "local_time", "duration_in_ms", "type"])
    loader = SimpleNamespace(
        get_user_ids=lambda: ["person-1"],
        load_sleep_summary=lambda **kwargs: summaries.copy(),
        load_sleep_stage=lambda **kwargs: stages.copy(),
    )
    result = sleep.get_sleep_statistics(
        loader,
        user_id="person-1",
        start_date=None,
        end_date=None,
        statistic=["sleep_score", "TIB", "TST", "count_awake"],
    )
    assert list(result) == ["sleep_score", "TIB", "TST", "count_awake"]
    assert result.loc[("person-1", pd.Timestamp("2024-01-02")), "TIB"] == 60
    assert result.loc[("person-1", pd.Timestamp("2024-01-02")), "TST"] == 60
    assert pd.isna(
        result.loc[("person-1", pd.Timestamp("2024-01-02")), "count_awake"]
    )
    assert pd.isna(result.loc[("person-1", pd.Timestamp("2024-01-03")), "sleep_score"])
    score = sleep.get_sleep_statistics(
        loader, "person-1", None, None, statistic="sleep_score"
    )
    assert isinstance(score, pd.Series)
    assert score.index.names == ["user_id", "calendar_date"]
    assert score.loc[("person-1", pd.Timestamp("2024-01-02"))] == 80

    timestamps = sleep.get_sleep_timestamps(loader, "person-1")
    assert isinstance(timestamps, pd.DataFrame)
    assert timestamps.index.names == ["user_id", "calendar_date"]
    assert list(timestamps) == ["bedtime", "wakeup"]

    with pytest.raises(ValueError, match="statistic must be one among"):
        sleep.get_sleep_statistics(
            loader, "person-1", None, None, statistic="unknown_feature"
        )


def test_sleep_trajectory_is_sorted_and_accepts_omitted_bounds():
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["night-2", "night-1"],
            "calendar_date": ["2024-01-03 12:00", "2024-01-02 12:00"],
            "local_time": ["2024-01-02 23:00", "2024-01-01 23:00"],
            "duration_in_ms": [60 * 60_000, 60 * 60_000],
            "awake_duration_in_ms": [0, 0],
            "deep_sleep_duration_in_ms": [20 * 60_000, 20 * 60_000],
            "light_sleep_duration_in_ms": [30 * 60_000, 30 * 60_000],
            "rem_sleep_duration_in_ms": [10 * 60_000, 10 * 60_000],
            "unmeasurable_sleep_duration_in_ms": [0, 0],
        }
    )
    loader = SimpleNamespace(
        get_user_ids=lambda: ["person-1"],
        load_sleep_summary=lambda **kwargs: summaries.copy(),
        load_sleep_stage=lambda **kwargs: pd.DataFrame(),
    )

    result = sleep.get_sleep_statistics(
        loader, user_id="person-1", statistic=["TIB", "SPT"]
    )

    assert result.index.tolist() == [
        ("person-1", pd.Timestamp("2024-01-02")),
        ("person-1", pd.Timestamp("2024-01-03")),
    ]
    assert result["TIB"].tolist() == [60, 60]
    assert result["SPT"].isna().all()


def test_sleep_trajectory_rejects_more_than_one_summary_per_night():
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-2"],
            "calendar_date": ["2024-01-02", "2024-01-02"],
            "local_time": ["2024-01-01 22:00", "2024-01-01 23:00"],
            "duration_in_ms": [60_000, 60_000],
            "awake_duration_in_ms": [0, 0],
            "deep_sleep_duration_in_ms": [60_000, 60_000],
            "light_sleep_duration_in_ms": [0, 0],
            "rem_sleep_duration_in_ms": [0, 0],
            "unmeasurable_sleep_duration_in_ms": [0, 0],
        }
    )
    loader = SimpleNamespace(
        get_user_ids=lambda: ["person-1"],
        load_sleep_summary=lambda **kwargs: summaries.copy(),
        load_sleep_stage=lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="one row per calendar_date"):
        sleep.get_sleep_statistics(loader, user_id="person-1", statistic="TIB")


def test_sleep_period_time_uses_datetimes_without_unix_milliseconds():
    summaries = pd.DataFrame(
        {"sleep_summary_id": ["night-1"]},
        index=pd.DatetimeIndex(["2024-03-31"], name="calendar_date"),
    )
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-1"],
            "local_time": pd.to_datetime(["2024-03-31 03:00", "2024-03-31 01:30"]),
            "utc_time": pd.to_datetime(["2024-03-31 01:00", "2024-03-31 00:30"]),
            "duration_in_ms": [30 * 60_000, 30 * 60_000],
            "type": ["light", "deep"],
        }
    )

    result = sleep._compute_sleep_period_time(summaries, stages)

    assert result.loc[pd.Timestamp("2024-03-31")] == 60


def test_cpd_is_undefined_on_first_night_and_after_gap():
    summaries = pd.DataFrame(
        {
            "calendar_date": pd.to_datetime(["2024-01-04", "2024-01-01", "2024-01-02"]),
            "local_time": pd.to_datetime(
                ["2024-01-03 23:00", "2023-12-31 23:00", "2024-01-01 23:00"]
            ),
            "duration_in_ms": [60 * 60_000, 60 * 60_000, 60 * 60_000],
            "awake_duration_in_ms": [0, 0, 0],
            "deep_sleep_duration_in_ms": [20 * 60_000] * 3,
            "light_sleep_duration_in_ms": [30 * 60_000] * 3,
            "rem_sleep_duration_in_ms": [10 * 60_000] * 3,
        },
        index=["gap", "first", "second"],
    )
    for compute in (sleep._compute_cpd_midpoint, sleep._compute_cpd_duration):
        result = compute(summaries, chronotype=("23:00", "00:00"))
        assert pd.isna(result.loc["first"])
        assert pd.notna(result.loc["second"])
        assert pd.isna(result.loc["gap"])
        assert list(result.index) == list(summaries.index)
