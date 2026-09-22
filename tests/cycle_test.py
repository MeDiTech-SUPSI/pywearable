import pandas as pd

from pywearable import cycle


class SessionLoader:
    def __init__(self, sessions):
        self.sessions = sessions
        self.calls = 0

    def get_user_ids(self):
        return ["user-1"]

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        self.calls += 1
        result = self.sessions.copy()
        if start_date is not None:
            result = result[result["calendar_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result[result["calendar_date"] < pd.Timestamp(end_date)]
        return result


def _sessions():
    return pd.DataFrame(
        {
            "id": ["night-1", "night-3", "nap"],
            "calendar_date": pd.to_datetime(
                ["2024-01-10", "2024-01-12", "2024-01-10"]
            ),
            "type": ["main_sleep", "main_sleep", "nap"],
            "main_sleep": [True, True, False],
            "local_time": pd.to_datetime(
                ["2024-01-09 23:00", "2024-01-11 23:30", "2024-01-10 14:00"]
            ),
            "local_end_time": pd.to_datetime(
                ["2024-01-10 07:00", "2024-01-12 07:30", "2024-01-10 14:30"]
            ),
            "time_in_bed_seconds": [8 * 3600, 8 * 3600, 30 * 60],
        }
    )


def test_cycle_intervals_do_not_bridge_missing_nights():
    loader = SessionLoader(_sessions())
    intervals = cycle.get_intervals(
        loader,
        cycle.CYCLE_AWAKE,
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-13",
    ).xs("user-1")

    assert not intervals.loc["2024-01-10", "interval_valid"]
    assert not intervals.loc["2024-01-11", "interval_valid"]
    assert not intervals.loc["2024-01-12", "interval_valid"]
    assert loader.calls == 1


def test_midnight_fallback_requires_a_known_cycle_start():
    loader = SessionLoader(_sessions())
    intervals = cycle.get_intervals(
        loader,
        cycle.CYCLE_PHYSIOLOGICAL,
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-13",
        missing_end="midnight",
    ).xs("user-1")

    assert intervals.loc["2024-01-10", "start_time"] == pd.Timestamp(
        "2024-01-09 23:00"
    )
    assert intervals.loc["2024-01-10", "end_time"] == pd.Timestamp(
        "2024-01-11 00:00"
    )
    assert intervals.loc["2024-01-10", "end_source"] == "midnight"
    assert intervals.loc["2024-01-10", "interval_valid"]
    assert not intervals.loc["2024-01-11", "interval_valid"]


def test_night_cycles_use_only_main_sleep_sessions():
    intervals = cycle.get_intervals(
        SessionLoader(_sessions()),
        cycle.CYCLE_NIGHT,
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-11",
    ).xs("user-1")

    assert intervals.loc["2024-01-10", "start_time"] == pd.Timestamp(
        "2024-01-09 23:00"
    )
    assert intervals.loc["2024-01-10", "end_time"] == pd.Timestamp(
        "2024-01-10 07:00"
    )
    assert intervals.loc["2024-01-10", "end_source"] == "session_end"
