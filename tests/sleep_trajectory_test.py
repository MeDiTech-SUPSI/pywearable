from types import SimpleNamespace

import pandas as pd
import pytest

from pywearable import sleep


def _trajectory_loader(
    include_nap=True,
    missing_interval=False,
    stage_overrun=False,
    nap_stages=True,
):
    sessions = pd.DataFrame(
        [
            {
                "id": "night-1",
                "calendar_date": "2024-01-02",
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": "2024-01-01 23:00",
                "local_end_time": "2024-01-02 07:00",
                "time_in_bed_seconds": 8 * 3600,
            },
            {
                "id": "night-2",
                "calendar_date": "2024-01-03",
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": "2024-01-02 23:00",
                "local_end_time": "2024-01-03 07:00",
                "time_in_bed_seconds": 8 * 3600,
            },
            {
                "id": "night-3",
                "calendar_date": "2024-01-04",
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": "2024-01-03 23:00",
                "local_end_time": "2024-01-04 07:00",
                "time_in_bed_seconds": 8 * 3600,
            },
        ]
    )
    stages = pd.DataFrame(
        [
            {
                "sleep_summary_id": row.id,
                "local_time": row.local_time,
                "duration_in_ms": 8 * 3600 * 1000,
                "type": "asleep",
            }
            for row in sessions.itertuples()
        ]
    )
    if missing_interval:
        stages.loc[stages["sleep_summary_id"] == "night-2", "duration_in_ms"] = (
            7 * 3600 * 1000
        )
    if stage_overrun:
        stages.loc[stages["sleep_summary_id"] == "night-1", "duration_in_ms"] = (
            9 * 3600 * 1000
        )
    if include_nap:
        nap = pd.DataFrame(
            [
                {
                    "id": "nap-1",
                    "calendar_date": "2024-01-02",
                    "type": "nap",
                    "main_sleep": False,
                    "local_time": "2024-01-02 13:00",
                    "local_end_time": "2024-01-02 14:00",
                    "time_in_bed_seconds": 3600,
                }
            ]
        )
        sessions = pd.concat([sessions, nap], ignore_index=True)
        if nap_stages:
            stages = pd.concat(
                [
                    stages,
                    pd.DataFrame(
                        {
                            "sleep_summary_id": ["nap-1"],
                            "local_time": pd.to_datetime(["2024-01-02 13:00"]),
                            "duration_in_ms": [3600 * 1000],
                            "type": ["asleep"],
                        }
                    ),
                ],
                ignore_index=True,
            )
    return SimpleNamespace(
        get_user_ids=lambda: ["person-1"],
        load_sleep_sessions=lambda *args, **kwargs: sessions.copy(),
        load_sleep_session_stages=lambda *args, **kwargs: stages.copy(),
    )


def test_full_day_trajectory_infers_wake_only_between_consecutive_main_sleeps():
    trajectory = sleep.get_sleep_wake_trajectory(
        _trajectory_loader(),
        "person-1",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution="1h",
    ).xs("person-1")

    assert len(trajectory) == 24
    assert trajectory.loc["2024-01-02 01:00", "state"] == "sleep"
    assert trajectory.loc["2024-01-02 08:00", "state"] == "awake"
    assert trajectory.loc["2024-01-02 13:00", "state"] == "sleep"
    assert trajectory.loc["2024-01-02 13:00", "main_sleep"] == False
    assert trajectory.loc["2024-01-02 08:00", "state_source"] == (
        "between_main_sleeps"
    )
    assert trajectory["state_available"].all()


def test_full_day_trajectory_can_exclude_naps_without_losing_inferred_wake():
    trajectory = sleep.get_sleep_wake_trajectory(
        _trajectory_loader(),
        "person-1",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution="1h",
        include_naps=False,
    ).xs("person-1")

    assert trajectory.loc["2024-01-02 13:00", "state"] == "awake"
    assert pd.isna(trajectory.loc["2024-01-02 13:00", "session_id"])


def test_session_boundaries_clip_provider_stage_grids():
    trajectory = sleep.get_sleep_wake_trajectory(
        _trajectory_loader(include_nap=False, stage_overrun=True),
        "person-1",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution="1h",
    ).xs("person-1")

    assert trajectory.loc["2024-01-02 07:00", "state"] == "awake"
    assert trajectory.loc["2024-01-02 07:00", "state_source"] == (
        "between_main_sleeps"
    )


def test_sleep_wins_a_nonempty_tie_and_coverage_means_classified_time():
    loader = _trajectory_loader(include_nap=False)
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-1"],
            "local_time": pd.to_datetime(
                ["2024-01-02 00:00", "2024-01-02 00:30"]
            ),
            "duration_in_ms": [30 * 60_000, 30 * 60_000],
            "type": ["asleep", "awake"],
        }
    )
    loader.load_sleep_session_stages = lambda *args, **kwargs: stages.copy()

    trajectory = sleep.get_sleep_wake_trajectory(
        loader,
        "person-1",
        start_date="2024-01-02 00:00",
        end_date="2024-01-02 02:00",
        resolution="1h",
    ).xs("person-1")

    assert trajectory.loc["2024-01-02 00:00", "state"] == "sleep"
    assert trajectory.loc["2024-01-02 00:00", "state_coverage"] == 100
    assert pd.isna(trajectory.loc["2024-01-02 01:00", "state"])
    assert trajectory.loc["2024-01-02 01:00", "state_coverage"] == 0


def test_partial_stage_data_reports_partial_state_coverage():
    loader = _trajectory_loader(include_nap=False)
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1"],
            "local_time": pd.to_datetime(["2024-01-02 00:00"]),
            "duration_in_ms": [30 * 60_000],
            "type": ["asleep"],
        }
    )
    loader.load_sleep_session_stages = lambda *args, **kwargs: stages.copy()

    trajectory = sleep.get_sleep_wake_trajectory(
        loader,
        "person-1",
        start_date="2024-01-02 00:00",
        end_date="2024-01-02 01:00",
        resolution="1h",
    ).xs("person-1")

    assert trajectory.iloc[0]["state"] == "sleep"
    assert trajectory.iloc[0]["state_coverage"] == 50


def test_session_metadata_is_retained_when_session_stages_are_missing():
    trajectory = sleep.get_sleep_wake_trajectory(
        _trajectory_loader(nap_stages=False),
        "person-1",
        start_date="2024-01-02 12:00",
        end_date="2024-01-02 15:00",
        resolution="1h",
    ).xs("person-1")

    nap = trajectory.loc["2024-01-02 13:00"]
    assert pd.isna(nap["state"])
    assert nap["session_id"] == "nap-1"
    assert nap["main_sleep"] == False
    assert nap["session_type"] == "nap"
    assert nap["state_coverage"] == 0


def test_missing_nights_leave_the_intervening_trajectory_unknown():
    sessions = pd.DataFrame(
        [
            {
                "id": "night-1",
                "calendar_date": "2024-01-01",
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": "2023-12-31 23:00",
                "local_end_time": "2024-01-01 07:00",
                "time_in_bed_seconds": 8 * 3600,
            },
            {
                "id": "night-4",
                "calendar_date": "2024-01-04",
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": "2024-01-03 23:00",
                "local_end_time": "2024-01-04 07:00",
                "time_in_bed_seconds": 8 * 3600,
            },
        ]
    )
    stages = pd.DataFrame(
        {
            "sleep_summary_id": ["night-1", "night-4"],
            "local_time": pd.to_datetime(
                ["2023-12-31 23:00", "2024-01-03 23:00"]
            ),
            "duration_in_ms": [8 * 3600 * 1000, 8 * 3600 * 1000],
            "type": ["asleep", "asleep"],
        }
    )
    loader = SimpleNamespace(
        get_user_ids=lambda: ["person-1"],
        load_sleep_sessions=lambda *args, **kwargs: sessions.copy(),
        load_sleep_session_stages=lambda *args, **kwargs: stages.copy(),
    )

    trajectory = sleep.get_sleep_wake_trajectory(
        loader,
        "person-1",
        start_date="2024-01-01",
        end_date="2024-01-04",
        resolution="1h",
    ).xs("person-1")

    assert trajectory.loc["2024-01-01 01:00", "state"] == "sleep"
    assert pd.isna(trajectory.loc["2024-01-01 08:00", "state"])
    assert pd.isna(trajectory.loc["2024-01-02 12:00", "state"])
    assert pd.isna(trajectory.loc["2024-01-03 22:00", "state"])
    assert trajectory.loc["2024-01-03 23:00", "state"] == "sleep"


def test_sleep_regularity_reports_nap_policy_and_comparison_coverage():
    with_nap = sleep.get_sleep_regularity(
        _trajectory_loader(),
        "person-1",
        start_date="2024-01-03",
        end_date="2024-01-04",
        resolution="1h",
        include_naps=True,
    )
    without_nap = sleep.get_sleep_regularity(
        _trajectory_loader(),
        "person-1",
        start_date="2024-01-03",
        end_date="2024-01-04",
        resolution="1h",
        include_naps=False,
    )

    assert with_nap.iloc[0]["SRI"] == pytest.approx(100 - 200 / 24)
    assert with_nap.iloc[0]["sri_coverage"] == 100
    assert without_nap.iloc[0]["SRI"] == 100
    assert sleep.get_sleep_regularity_index(
        _trajectory_loader(),
        "person-1",
        start_date="2024-01-03",
        end_date="2024-01-04",
        resolution="1h",
    ).iloc[0] == pytest.approx(100 - 200 / 24)


def test_sleep_regularity_does_not_fill_missing_session_time_as_wake():
    regularity = sleep.get_sleep_regularity(
        _trajectory_loader(include_nap=False, missing_interval=True),
        "person-1",
        start_date="2024-01-03",
        end_date="2024-01-04",
        resolution="1h",
    )

    assert regularity.iloc[0]["SRI"] == 100
    assert regularity.iloc[0]["sri_coverage"] == pytest.approx(23 / 24 * 100)


def test_sleep_regularity_requires_a_resolution_dividing_24_hours():
    with pytest.raises(ValueError, match="divide evenly"):
        sleep.get_sleep_regularity(
            _trajectory_loader(),
            "person-1",
            start_date="2024-01-03",
            end_date="2024-01-04",
            resolution="7h",
        )
