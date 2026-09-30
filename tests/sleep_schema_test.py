from types import SimpleNamespace

import pandas as pd
import pytest

from pywearable import hierarchy, sleep, schema


def test_sleep_feature_names_are_family_prefixed():
    assert all(name.startswith("sleep_") for name in sleep.SLEEP_STATISTICS_DICT)
    assert sleep.SLEEP_STATISTIC_SLEEP_ONSET == "sleep_sleep_onset"
    assert hierarchy.family_statistics(
        [sleep.SLEEP_STATISTIC_TIB, "spo2_mean", sleep.SLEEP_STATISTIC_SLEEP_SCORE],
        "sleep",
    ) == [sleep.SLEEP_STATISTIC_TIB, sleep.SLEEP_STATISTIC_SLEEP_SCORE]


def test_sleep_schema_requires_canonical_column_names():
    raw = pd.DataFrame(
        {
            "calendarDate": ["2024-01-02"],
            "sleepSummaryId": ["night-1"],
            "durationInMs": [1],
            "REMValue": [2],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        schema.prepare_sleep_summary(raw)


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
        statistic=["sleep_score", "sleep_TIB", "sleep_TST", "sleep_count_awake"],
    )
    assert list(result) == [
        "sleep_score",
        "sleep_score__coverage",
        "sleep_TIB",
        "sleep_TIB__coverage",
        "sleep_TST",
        "sleep_TST__coverage",
        "sleep_count_awake",
        "sleep_count_awake__coverage",
    ]
    assert result.loc[("person-1", pd.Timestamp("2024-01-02")), "sleep_TIB"] == 60
    assert result.loc[("person-1", pd.Timestamp("2024-01-02")), "sleep_TST"] == 60
    assert pd.isna(
        result.loc[("person-1", pd.Timestamp("2024-01-02")), "sleep_count_awake"]
    )
    assert pd.isna(result.loc[("person-1", pd.Timestamp("2024-01-03")), "sleep_score"])
    score = sleep.get_sleep_statistics(
        loader, statistic="sleep_score", user_id="person-1"
    )
    assert isinstance(score, pd.DataFrame)
    assert score.index.names == ["user_id", "calendar_date"]
    assert (
        score.loc[("person-1", pd.Timestamp("2024-01-02")), "sleep_score"]
        == 80
    )

    timestamps = sleep.get_sleep_timestamps(loader, "person-1")
    assert isinstance(timestamps, pd.DataFrame)
    assert timestamps.index.names == ["user_id", "calendar_date"]
    assert list(timestamps) == [
        "sleep_bedtime",
        "sleep_bedtime__coverage",
        "sleep_sleep_onset",
        "sleep_sleep_onset__coverage",
        "sleep_wakeup",
        "sleep_wakeup__coverage",
        "sleep_getup",
        "sleep_getup__coverage",
    ]
    assert timestamps[["sleep_sleep_onset", "sleep_wakeup"]].isna().all().all()
    assert list(sleep.get_sleep_onset(loader, "person-1")) == [
        "sleep_sleep_onset",
        "sleep_sleep_onset__coverage",
    ]

    with pytest.raises(ValueError, match="statistic must be one among"):
        sleep.get_sleep_statistics(
            loader, statistic="unknown_feature", user_id="person-1"
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
        loader, user_id="person-1", statistic=["sleep_TIB", "sleep_SPT"]
    )

    assert result.index.tolist() == [
        ("person-1", pd.Timestamp("2024-01-02")),
        ("person-1", pd.Timestamp("2024-01-03")),
    ]
    assert result["sleep_TIB"].tolist() == [60, 60]
    assert result["sleep_SPT"].isna().all()


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
        sleep.get_sleep_statistics(loader, user_id="person-1", statistic="sleep_TIB")


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


def test_stage_features_use_sleep_boundaries_bouts_and_utc_elapsed_time():
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"],
            "local_time": pd.to_datetime(["2024-03-31 01:30"]),
            "utc_time": pd.to_datetime(["2024-03-31 00:30"]),
        },
        index=pd.DatetimeIndex(["2024-03-31"], name="calendar_date"),
    )
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"] * 8,
            "local_time": pd.to_datetime(
                [
                    "2024-03-31 01:30",
                    "2024-03-31 01:40",
                    "2024-03-31 03:00",
                    "2024-03-31 03:20",
                    "2024-03-31 03:25",
                    "2024-03-31 03:30",
                    "2024-03-31 03:50",
                    "2024-03-31 04:00",
                ]
            ),
            "utc_time": pd.to_datetime(
                [
                    "2024-03-31 00:30",
                    "2024-03-31 00:40",
                    "2024-03-31 01:00",
                    "2024-03-31 01:20",
                    "2024-03-31 01:25",
                    "2024-03-31 01:30",
                    "2024-03-31 01:50",
                    "2024-03-31 02:00",
                ]
            ),
            "duration_in_ms": [10, 20, 20, 5, 5, 20, 10, 10],
            "type": [
                "unmeasurable",
                "awake",
                "light",
                "awake",
                "awake",
                "deep",
                "unmeasurable",
                "awake",
            ],
        }
    )
    stages["duration_in_ms"] *= 60_000

    spt = sleep._compute_sleep_period_time(summaries, stages)
    waso = sleep._compute_wake_after_sleep_onset(summaries, stages)
    sol = sleep._compute_sleep_onset_latency(summaries, stages)
    latencies = sleep._compute_latencies(summaries, stages)
    counts = sleep._compute_stage_count(summaries, stages)
    sme = sleep._compute_sleep_maintenance_efficiency(summaries, stages)

    assert spt.iloc[0] == 50
    assert waso.iloc[0] == 10
    assert sol.iloc[0] == 30
    assert latencies.iloc[0]["light"] == 30
    assert latencies.iloc[0]["deep"] == 60
    assert counts.iloc[0]["awake"] == 3
    assert sme.iloc[0] == 80


def test_sleep_maintenance_efficiency_uses_one_stage_trajectory():
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"],
            # The exact provider summary and its five-minute stage grid differ.
            "duration_in_ms": [63 * 60_000],
            "unmeasurable_sleep_duration_in_ms": [0],
        },
        index=pd.DatetimeIndex(["2024-01-02"], name="calendar_date"),
    )
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-1"],
            "local_time": pd.to_datetime(["2024-01-01 23:00", "2024-01-02 00:00"]),
            "duration_in_ms": [60 * 60_000, 5 * 60_000],
            "type": ["light", "awake"],
        }
    )

    result = sleep._compute_sleep_maintenance_efficiency(summaries, stages)

    assert result.iloc[0] == 100


def test_wakeup_is_missing_after_an_unexplained_terminal_stage_gap():
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"],
            "local_time": pd.to_datetime(["2024-01-01 23:00"]),
            "local_end_time": pd.to_datetime(["2024-01-02 07:00"]),
            "duration_in_ms": [8 * 60 * 60_000],
            "awake_duration_in_ms": [0],
        },
        index=pd.DatetimeIndex(["2024-01-02"], name="calendar_date"),
    )
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-1"],
            "local_time": pd.to_datetime(
                ["2024-01-01 23:00", "2024-01-02 06:00"]
            ),
            "duration_in_ms": [7 * 60 * 60_000, 10 * 60_000],
            "type": ["light", "awake"],
        }
    )

    wakeup = sleep._compute_wakeup_time(summaries, stages)

    assert pd.isna(wakeup.iloc[0])


def test_sleep_stage_contract_rejects_unknown_stage_labels():
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"],
            "local_time": pd.to_datetime(["2024-01-01 23:00"]),
            "duration_in_ms": [60_000],
            "type": ["mystery"],
        }
    )

    with pytest.raises(ValueError, match="unsupported values: mystery"):
        schema.prepare_sleep_stages(stages)


def test_cpd_is_undefined_on_first_night_and_after_gap():
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["gap", "first", "second"],
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
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["gap", "first", "second"],
            "local_time": summaries["local_time"].to_numpy(),
            "duration_in_ms": [60 * 60_000] * 3,
            "type": ["light"] * 3,
        }
    )
    results = (
        sleep._compute_cpd_midpoint(
            summaries, chronotype=("23:00", "00:00"), sleep_stages=stages
        ),
        sleep._compute_cpd_duration(summaries, chronotype=("23:00", "00:00")),
    )
    for result in results:
        assert pd.isna(result.loc["first"])
        assert pd.notna(result.loc["second"])
        assert pd.isna(result.loc["gap"])
        assert list(result.index) == list(summaries.index)
