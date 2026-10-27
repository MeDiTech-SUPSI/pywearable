import numpy as np
import pandas as pd
import pytest

from pywearable import sleep
from pywearable.loader import LifeSnapsParquetLoader, OuraLoader
from tests.helpers import LoaderBase


class _SessionLoader(LoaderBase):
    sleep_sessions_include_naps = True

    def __init__(self, sessions=(), stages=()):
        self.sessions = pd.DataFrame(sessions)
        self.stages = pd.DataFrame(stages)

    def get_user_ids(self):
        return ["person-1", "person-2"]

    def load_sleep_sessions(self, *args, **kwargs):
        return self.sessions.copy()

    def load_sleep_session_stages(self, *args, **kwargs):
        return self.stages.copy()


def _session(identifier, start, end, main=True):
    return {
        "id": identifier,
        "calendar_date": pd.Timestamp(end).normalize(),
        "type": "main_sleep" if main else "nap",
        "main_sleep": main,
        "local_time": pd.Timestamp(start),
        "local_end_time": pd.Timestamp(end),
    }


def _stage(identifier, start, minutes, stage="asleep"):
    return {
        "sleep_summary_id": identifier,
        "local_time": pd.Timestamp(start),
        "duration_in_ms": minutes * 60_000,
        "type": stage,
    }


def _result(loader, start="2024-01-02", end="2024-01-03"):
    return sleep.get_total_sleep_time_24h(
        loader, "person-1", start_date=start, end_date=end
    ).xs("person-1")


def test_main_sleep_and_naps_are_summed_without_entering_nightly_statistics():
    loader = _SessionLoader(
        [
            _session("night", "2024-01-01 23:00", "2024-01-02 07:00"),
            _session("nap", "2024-01-02 13:00", "2024-01-02 14:00", False),
        ],
        [
            _stage("night", "2024-01-01 23:00", 480),
            _stage("nap", "2024-01-02 13:00", 60),
        ],
    )
    result = _result(loader)
    assert result.loc["2024-01-02", "sleep_TST_24h"] == 540
    assert result.loc["2024-01-02", "sleep_TST_24h__coverage"] == 1
    assert "sleep_TST_24h" not in sleep.SLEEP_STATISTICS_DICT
    assert sleep.SLEEP_STATISTIC_TST_24H == "sleep_TST_24h"


def test_sessions_crossing_18_are_assigned_whole_by_their_start():
    loader = _SessionLoader(
        [
            _session("before", "2024-01-02 17:30", "2024-01-02 18:30", False),
            _session("at", "2024-01-02 18:30", "2024-01-02 19:00", False),
        ],
        [
            _stage("before", "2024-01-02 17:30", 60),
            _stage("at", "2024-01-02 18:30", 30),
        ],
    )
    result = _result(loader, end="2024-01-04")
    assert result["sleep_TST_24h"].tolist() == [60, 30]
    assert result["sleep_TST_24h__coverage"].tolist() == [1, 1]


@pytest.mark.parametrize(
    "start,end,day",
    [
        ("2024-01-02 17:59", "2024-01-02 18:01", "2024-01-02"),
        ("2024-01-02 18:00", "2024-01-02 18:02", "2024-01-03"),
    ],
)
def test_exact_sleep_day_boundary(start, end, day):
    result = sleep.get_total_sleep_time_24h(
        _SessionLoader(
            [_session("nap", start, end, False)],
            [_stage("nap", start, 2)],
        ),
        "person-1",
    ).xs("person-1")
    assert result.index.tolist() == [pd.Timestamp(day)]
    assert result.iloc[0]["sleep_TST_24h"] == 2


def test_unknown_and_missing_stages_lower_recorded_session_coverage():
    loader = _SessionLoader(
        [
            _session("night", "2024-01-01 23:00", "2024-01-02 07:00"),
            _session("nap", "2024-01-02 13:00", "2024-01-02 14:00", False),
        ],
        [
            _stage("night", "2024-01-01 23:00", 420),
            _stage("night", "2024-01-02 06:00", 30, "awake"),
            _stage("night", "2024-01-02 06:30", 30, "unmeasurable"),
        ],
    )
    row = _result(loader).iloc[0]
    assert row["sleep_TST_24h"] == 420
    assert row["sleep_TST_24h__coverage"] == pytest.approx(450 / 540)


def test_native_durations_are_clipped_without_resampling_or_double_counting():
    stages = [_stage("nap", "2024-01-02 13:00", 1, "light")]
    loader = _SessionLoader(
        [_session("nap", "2024-01-02 13:00:15", "2024-01-02 13:00:45", False)],
        stages + stages,
    )
    row = _result(loader).iloc[0]
    assert row["sleep_TST_24h"] == 0.5
    assert row["sleep_TST_24h__coverage"] == 1


def test_unknown_or_absent_sessions_do_not_become_zero_sleep_days():
    loader = _SessionLoader(
        [_session("night", "2024-01-01 23:00", "2024-01-02 07:00")],
        [_stage("night", "2024-01-01 23:00", 480, "unmeasurable")],
    )
    result = _result(loader, end="2024-01-04")
    assert result["sleep_TST_24h"].isna().all()
    assert result["sleep_TST_24h__coverage"].eq(0).all()
    assert result.index.tolist() == list(pd.date_range("2024-01-02", periods=2))


def test_observed_wake_can_produce_zero_sleep_without_inferred_daytime_coverage():
    loader = _SessionLoader(
        [_session("nap", "2024-01-02 13:00", "2024-01-02 14:00", False)],
        [_stage("nap", "2024-01-02 13:00", 60, "awake")],
    )
    row = _result(loader).iloc[0]
    assert row["sleep_TST_24h"] == 0
    assert row["sleep_TST_24h__coverage"] == 1


def test_empty_sources_return_requested_missing_days_or_a_typed_empty_frame():
    loader = _SessionLoader()
    result = _result(loader)
    assert np.isnan(result.iloc[0]["sleep_TST_24h"])
    assert result.iloc[0]["sleep_TST_24h__coverage"] == 0
    empty = sleep.get_total_sleep_time_24h(loader)
    assert empty.empty
    assert empty.index.names == ["user_id", "calendar_date"]
    assert empty["sleep_TST_24h"].dtype == float


def test_default_bounds_and_multiple_users_keep_sleep_day_labels():
    loader = _SessionLoader(
        [_session("night", "2024-01-01 23:00", "2024-01-02 07:00")],
        [_stage("night", "2024-01-01 23:00", 480)],
    )
    result = sleep.get_total_sleep_time_24h(loader)
    assert result.index.tolist() == [
        ("person-1", pd.Timestamp("2024-01-02")),
        ("person-2", pd.Timestamp("2024-01-02")),
    ]
    pd.testing.assert_frame_equal(
        _result(loader, "2024-01-02T00:00+02:00", "2024-01-03T00:00+02:00"),
        _result(loader),
    )


@pytest.mark.parametrize(
    "start,end",
    [("2024-01-03", "2024-01-02"), ("2024-01-02", "2024-01-02")],
)
def test_invalid_bounds_are_rejected(start, end):
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        _result(_SessionLoader(), start, end)


def test_nightly_only_loaders_are_not_advertised_as_24_hour_sources():
    loader = _SessionLoader()
    loader.sleep_sessions_include_naps = False
    with pytest.raises(ValueError, match="sleep_sessions_include_naps"):
        _result(loader)
    assert OuraLoader.sleep_sessions_include_naps
    assert LifeSnapsParquetLoader.sleep_sessions_include_naps
    assert not LoaderBase.sleep_sessions_include_naps


def test_uncertain_local_stage_times_are_not_used():
    stage = _stage("night", "2024-01-01 23:00", 480)
    stage["local_time_uncertain"] = True
    loader = _SessionLoader(
        [_session("night", "2024-01-01 23:00", "2024-01-02 07:00")], [stage]
    )
    assert _result(loader).iloc[0].isna().tolist() == [True, False]
    assert _result(loader).iloc[0]["sleep_TST_24h__coverage"] == 0


def test_overlapping_stage_labels_are_rejected_instead_of_inflating_sleep():
    loader = _SessionLoader(
        [_session("night", "2024-01-01 23:00", "2024-01-02 07:00")],
        [
            _stage("night", "2024-01-01 23:00", 480),
            _stage("night", "2024-01-02 00:00", 30, "awake"),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        _result(loader)


def test_overlapping_sessions_are_rejected_instead_of_counted_twice():
    loader = _SessionLoader(
        [
            _session("night", "2024-01-01 23:00", "2024-01-02 07:00"),
            _session("duplicate", "2024-01-02 00:00", "2024-01-02 06:00"),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        _result(loader)
