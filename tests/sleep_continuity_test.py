import numpy as np
import pandas as pd
import pytest

from pywearable import sleep


def _summary(duration_minutes=60, awake_minutes=30):
    return pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"],
            "local_time": pd.to_datetime(["2024-01-01 22:50"]),
            "local_end_time": pd.to_datetime(["2024-01-02 00:20"]),
            "utc_time": pd.to_datetime(["2024-01-01 22:50"]),
            "utc_end_time": pd.to_datetime(["2024-01-02 00:20"]),
            "duration_in_ms": [duration_minutes * 60_000],
            "awake_duration_in_ms": [awake_minutes * 60_000],
            "unmeasurable_sleep_duration_in_ms": [10 * 60_000],
            "deep_sleep_duration_in_ms": [10 * 60_000],
            "light_sleep_duration_in_ms": [30 * 60_000],
            "rem_sleep_duration_in_ms": [10 * 60_000],
        },
        index=pd.DatetimeIndex(["2024-01-02"], name="calendar_date"),
    )


def _stages():
    durations = [10, 10, 10, 5, 10, 10, 20, 15]
    return pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"] * len(durations),
            "local_time": pd.to_datetime(
                [
                    "2024-01-01 22:50",
                    "2024-01-01 23:00",
                    "2024-01-01 23:10",
                    "2024-01-01 23:20",
                    "2024-01-01 23:25",
                    "2024-01-01 23:35",
                    "2024-01-01 23:45",
                    "2024-01-02 00:05",
                ]
            ),
            "utc_time": pd.to_datetime(
                [
                    "2024-01-01 22:50",
                    "2024-01-01 23:00",
                    "2024-01-01 23:10",
                    "2024-01-01 23:20",
                    "2024-01-01 23:25",
                    "2024-01-01 23:35",
                    "2024-01-01 23:45",
                    "2024-01-02 00:05",
                ]
            ),
            "duration_in_ms": np.array(durations) * 60_000,
            "type": [
                "awake",
                "light",
                "deep",
                "awake",
                "rem",
                "unmeasurable",
                "light",
                "awake",
            ],
        }
    )


def test_continuity_metrics_use_internal_bouts_and_do_not_bridge_missing_stages():
    summary = _summary()
    stages = _stages()

    assert sleep._compute_awakening_count(summary, stages).iloc[0] == 1
    assert sleep._compute_awakening_index(summary, stages).iloc[0] == pytest.approx(1.2)
    assert sleep._compute_median_awakening_duration(summary, stages).iloc[0] == 5
    assert sleep._compute_stage_transition_count(summary, stages).iloc[0] == 1
    assert sleep._compute_stage_transition_index(summary, stages).iloc[0] == pytest.approx(1.2)
    assert sleep._compute_stage_coverage(summary, stages).iloc[0] == pytest.approx(
        100 * 80 / 90
    )


def test_sleep_timestamps_distinguish_session_and_observed_sleep_boundaries():
    summary = _summary()
    stages = _stages()

    assert sleep._compute_bedtime(summary).iloc[0] == pd.Timestamp("2024-01-01 22:50")
    assert sleep._compute_sleep_onset(summary, stages).iloc[0] == pd.Timestamp(
        "2024-01-01 23:00"
    )
    assert sleep._compute_wakeup_time(summary, stages).iloc[0] == pd.Timestamp(
        "2024-01-02 00:05"
    )
    assert sleep._compute_getup_time(summary).iloc[0] == pd.Timestamp(
        "2024-01-02 00:20"
    )
    assert sleep._compute_getup_latency(summary, stages).iloc[0] == 15
    assert sleep._compute_sleep_midpoint(summary, stages).iloc[0] == pd.Timestamp(
        "2024-01-01 23:32:30"
    )


def test_trailing_unmeasurable_stage_does_not_guess_wakeup():
    summary = _summary(duration_minutes=30, awake_minutes=0)
    summary["local_end_time"] = pd.Timestamp("2024-01-01 23:20")
    summary["utc_end_time"] = pd.Timestamp("2024-01-01 23:20")
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-1"],
            "local_time": pd.to_datetime(
                ["2024-01-01 22:50", "2024-01-01 23:10"]
            ),
            "utc_time": pd.to_datetime(
                ["2024-01-01 22:50", "2024-01-01 23:10"]
            ),
            "duration_in_ms": [20 * 60_000, 10 * 60_000],
            "type": ["light", "unmeasurable"],
        }
    )

    assert pd.isna(sleep._compute_wakeup_time(summary, stages).iloc[0])
    assert pd.isna(sleep._compute_sleep_midpoint(summary, stages).iloc[0])
    assert pd.isna(sleep._compute_getup_latency(summary, stages).iloc[0])
    assert sleep._compute_stage_coverage(summary, stages).iloc[0] == pytest.approx(
        100 * 20 / 30
    )


def test_generic_asleep_trajectory_has_no_detailed_stage_transition_metric():
    summary = _summary(duration_minutes=60, awake_minutes=10)
    summary["local_end_time"] = pd.Timestamp("2024-01-02 00:00")
    summary["utc_end_time"] = pd.Timestamp("2024-01-02 00:00")
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-1"],
            "local_time": pd.to_datetime(
                ["2024-01-01 22:50", "2024-01-01 23:50"]
            ),
            "duration_in_ms": [60 * 60_000, 10 * 60_000],
            "type": ["asleep", "awake"],
        }
    )

    assert pd.isna(sleep._compute_stage_transition_count(summary, stages).iloc[0])
    assert pd.isna(sleep._compute_stage_transition_index(summary, stages).iloc[0])
    assert sleep._compute_awakening_count(summary, stages).iloc[0] == 0
    assert sleep._compute_stage_coverage(summary, stages).iloc[0] == 100


def test_all_unmeasurable_trajectory_has_zero_stage_coverage():
    summary = _summary(duration_minutes=30, awake_minutes=0)
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"],
            "local_time": pd.to_datetime(["2024-01-01 22:50"]),
            "duration_in_ms": [30 * 60_000],
            "type": ["unmeasurable"],
        }
    )

    assert sleep._compute_stage_coverage(summary, stages).iloc[0] == 0
