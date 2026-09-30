import pandas as pd
import pytest

from pywearable import heart_rate
from tests.helpers import LoaderBase


class HeartRateCycleLoader(LoaderBase):
    heart_rate_time_columns = ("utc_time", "local_time")

    def __init__(self):
        self.session_calls = 0
        self.heart_rate_calls = 0
        self.requested_time_cols = []
        self.sessions = pd.DataFrame(
            {
                "id": ["night-1", "night-2"],
                "calendar_date": pd.to_datetime(["2024-01-10", "2024-01-11"]),
                "type": ["main_sleep", "main_sleep"],
                "main_sleep": [True, True],
                "local_time": pd.to_datetime(
                    ["2024-01-09 23:00", "2024-01-10 23:00"]
                ),
                "local_end_time": pd.to_datetime(
                    ["2024-01-10 07:00", "2024-01-11 07:00"]
                ),
                "time_in_bed_seconds": [8 * 3600, 8 * 3600],
            }
        )
        local = pd.to_datetime(
            [
                "2024-01-10 01:00",
                "2024-01-10 02:00",
                "2024-01-11 01:00",
            ]
        )
        self.hr = pd.DataFrame(
            {
                "local_time": local,
                "utc_time": local - pd.Timedelta(hours=1),
                "timezone": pd.Timedelta(hours=1),
                "heart_rate": [60, 80, 100],
                "timestamp_conflict": False,
            }
        )

    def get_user_ids(self):
        return ["user-1"]

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        self.session_calls += 1
        result = self.sessions.copy()
        if start_date is not None:
            result = result[result["calendar_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result[result["calendar_date"] < pd.Timestamp(end_date)]
        return result

    def load_heart_rate(
        self, user_id, start_date=None, end_date=None, time_col="local_time"
    ):
        self.heart_rate_calls += 1
        self.requested_time_cols.append(time_col)
        result = self.hr.copy()
        if start_date is not None:
            result = result[result[time_col] >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result[result[time_col] < pd.Timestamp(end_date)]
        return result


class MultiUserCycleLoader(LoaderBase):
    heart_rate_time_columns = ("utc_time", "local_time")

    def __init__(self):
        self.heart_rate_requests = {}
        self.sessions = {
            "user-1": self._sessions("2024-01-10"),
            "user-2": self._sessions("2024-03-10"),
        }
        self.heart_rate = {
            "user-1": self._heart_rate("2024-01-10 01:00"),
            "user-2": self._heart_rate("2024-03-10 01:00"),
        }

    @staticmethod
    def _sessions(first_day):
        days = pd.date_range(first_day, periods=2, freq="D")
        return pd.DataFrame(
            {
                "id": [f"night-{day:%Y%m%d}" for day in days],
                "calendar_date": days,
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": days - pd.Timedelta(hours=1),
                "local_end_time": days + pd.Timedelta(hours=7),
                "time_in_bed_seconds": 8 * 3600,
            }
        )

    @staticmethod
    def _heart_rate(timestamp):
        local = pd.DatetimeIndex([timestamp])
        return pd.DataFrame(
            {
                "local_time": local,
                "utc_time": local - pd.Timedelta(hours=1),
                "timezone": pd.Timedelta(hours=1),
                "heart_rate": [70],
                "timestamp_conflict": False,
            }
        )

    def get_user_ids(self):
        return list(self.sessions)

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        return self.sessions[user_id].copy()

    def load_heart_rate(
        self, user_id, start_date=None, end_date=None, time_col="local_time"
    ):
        self.heart_rate_requests[user_id] = (
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
        )
        return self.heart_rate[user_id].copy()


def test_cycle_heart_rate_is_computed_only_for_the_requested_cycle():
    loader = HeartRateCycleLoader()
    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=["heart_rate_mean", "heart_rate_max"],
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-12",
        resolution="cycle_night",
    )

    assert result.loc[("user-1", pd.Timestamp("2024-01-10")), "heart_rate_mean"] == 70
    assert result.loc[("user-1", pd.Timestamp("2024-01-11")), "heart_rate_max"] == 100
    assert loader.session_calls == 1
    assert loader.heart_rate_calls == 1


def test_cycle_to_monthly_uses_equal_cycle_weights_and_expected_coverage():
    result = heart_rate.get_heart_rate_statistics(
        HeartRateCycleLoader(),
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-13",
        resolution=["cycle_night", "monthly"],
        aggregation="mean",
    )

    row = result.iloc[0]
    assert row["heart_rate_mean__cycle_night__monthly_mean"] == 85
    assert row["heart_rate_mean__cycle_night__monthly_mean__coverage"] == pytest.approx(
        2 / 3
    )


def test_scalar_heart_rate_does_not_load_sleep_sessions():
    loader = HeartRateCycleLoader()
    heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-11",
        resolution="1h",
        time_reference="local",
    )
    assert loader.session_calls == 0


def test_default_extraction_returns_coverage_without_filtering():
    loader = HeartRateCycleLoader()

    scalar = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-11",
        resolution="1h",
        time_reference="local",
    )
    hierarchical = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-11",
        resolution=["1h", "daily"],
        aggregation="mean",
        time_reference="local",
    )
    cycle_result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-11",
        resolution="cycle_night",
    )

    assert not scalar.empty
    assert not hierarchical.empty
    assert not cycle_result.empty
    assert "heart_rate_mean__coverage" in scalar
    assert "heart_rate_mean__1h__daily_mean__coverage" in hierarchical
    assert "heart_rate_mean__coverage" in cycle_result


def test_fixed_windows_can_be_aggregated_through_cycles_and_calendar_periods():
    loader = HeartRateCycleLoader()
    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-12",
        resolution=["5min", "cycle_night", "monthly"],
        aggregation="mean",
    )

    column = "heart_rate_mean__5min__cycle_night_mean__monthly_mean"
    assert result.iloc[0][column] == 85
    assert loader.session_calls == 1


def test_per_level_coverage_thresholds_cascade_through_the_hierarchy():
    result = heart_rate.get_heart_rate_statistics(
        HeartRateCycleLoader(),
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-13",
        resolution=["1h", "cycle_night", "monthly"],
        aggregation=[["mean"], ["mean"]],
        min_coverage=[None, 0.2, 0.5],
    )

    feature = "heart_rate_mean__1h__cycle_night_mean__monthly_mean"
    assert pd.isna(result.iloc[0][feature])
    assert result.iloc[0][f"{feature}__coverage"] == pytest.approx(1 / 3)


def test_valid_child_threshold_filters_heart_rate_hierarchy():
    kwargs = {
        "loader": HeartRateCycleLoader(),
        "statistic": "heart_rate_mean",
        "user_id": "user-1",
        "start_date": "2024-01-10",
        "end_date": "2024-01-13",
        "resolution": ["cycle_night", "monthly"],
        "aggregation": "mean",
    }
    accepted = heart_rate.get_heart_rate_statistics(
        min_valid_children=2,
        **kwargs,
    )
    rejected = heart_rate.get_heart_rate_statistics(
        min_valid_children=3,
        **kwargs,
    )

    feature = "heart_rate_mean__cycle_night__monthly_mean"
    assert accepted.iloc[0][feature] == 85
    assert pd.isna(rejected.iloc[0][feature])
    assert rejected.iloc[0][f"{feature}__coverage"] == pytest.approx(2 / 3)


def test_single_resolution_heart_rate_rejects_valid_child_threshold():
    with pytest.raises(ValueError, match="requires a resolution hierarchy"):
        heart_rate.get_heart_rate_statistics(
            HeartRateCycleLoader(),
            statistic="heart_rate_mean",
            user_id="user-1",
            resolution="daily",
            min_valid_children=3,
        )


def test_coverage_threshold_count_matches_every_resolution():
    with pytest.raises(ValueError, match="one value per resolution"):
        heart_rate.get_heart_rate_statistics(
            HeartRateCycleLoader(),
            statistic="heart_rate_mean",
            user_id="user-1",
            resolution=["1h", "daily", "monthly"],
            min_coverage=[0.5, 0.8],
        )


def test_fixed_windows_with_half_their_duration_in_a_cycle_are_included():
    loader = HeartRateCycleLoader()
    loader.sessions = pd.DataFrame(
        {
            "id": ["night-1"],
            "calendar_date": [pd.Timestamp("2024-01-10")],
            "type": ["main_sleep"],
            "main_sleep": [True],
            "local_time": [pd.Timestamp("2024-01-09 23:30")],
            "local_end_time": [pd.Timestamp("2024-01-10 01:10")],
            "time_in_bed_seconds": [100 * 60],
        }
    )
    local = pd.to_datetime(["2024-01-09 23:40", "2024-01-10 00:10"])
    loader.hr = pd.DataFrame(
        {
            "local_time": local,
            "utc_time": local - pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "heart_rate": [60, 80],
            "timestamp_conflict": False,
        }
    )

    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        resolution=["1h", "cycle_night"],
        aggregation="mean",
    )

    feature = "heart_rate_mean__1h__cycle_night_mean"
    assert result.iloc[0][feature] == 70


def test_fixed_windows_can_be_aggregated_to_a_larger_fixed_resolution():
    result = heart_rate.get_heart_rate_statistics(
        HeartRateCycleLoader(),
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10 00:00",
        end_date="2024-01-10 03:00",
        resolution=["5min", "1h"],
        aggregation="mean",
        time_reference="local",
    )

    assert result.index.names == ["user_id", "local_time"]
    assert result.loc[
        ("user-1", pd.Timestamp("2024-01-10 01:00")),
        "heart_rate_mean__5min__1h_mean",
    ] == 60
    assert result.loc[
        ("user-1", pd.Timestamp("2024-01-10 02:00")),
        "heart_rate_mean__5min__1h_mean",
    ] == 80


def test_partial_fixed_windows_feed_calendar_parents_by_physical_overlap():
    loader = HeartRateCycleLoader()
    local = pd.to_datetime(["2024-01-10 10:03", "2024-01-10 10:06"])
    loader.hr = pd.DataFrame(
        {
            "local_time": local,
            "utc_time": local - pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "heart_rate": [10, 100],
            "timestamp_conflict": False,
        }
    )

    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10 10:03",
        end_date="2024-01-10 10:08",
        resolution=["5min", "daily"],
        aggregation="mean",
        time_reference="local",
    )

    feature = "heart_rate_mean__5min__daily_mean"
    assert result.iloc[0][feature] == 55
    assert result.iloc[0][f"{feature}__coverage"] == 1


def test_partial_daily_primary_windows_stop_at_calendar_midnight():
    loader = HeartRateCycleLoader()
    local = pd.to_datetime(["2024-01-10 13:00", "2024-01-11 01:00"])
    loader.hr = pd.DataFrame(
        {
            "local_time": local,
            "utc_time": local - pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "heart_rate": [60, 80],
            "timestamp_conflict": False,
        }
    )

    trajectory = heart_rate._single_resolution_hr_trajectory(
        loader,
        ["user-1"],
        ["heart_rate_mean"],
        "2024-01-10 12:00",
        "2024-01-11 12:00",
        "daily",
        {},
        "local",
        None,
        set(),
    )

    assert trajectory["period_start"].tolist() == pd.to_datetime(
        ["2024-01-10 12:00", "2024-01-11 00:00"]
    ).tolist()
    assert trajectory["period_end"].tolist() == pd.to_datetime(
        ["2024-01-11 00:00", "2024-01-11 12:00"]
    ).tolist()


def test_unbounded_hierarchy_counts_internal_missing_primary_windows():
    loader = HeartRateCycleLoader()
    loader.hr = loader.hr.iloc[:2].copy()
    loader.hr.loc[loader.hr.index[1], "local_time"] = pd.Timestamp(
        "2024-01-10 03:00"
    )
    loader.hr.loc[loader.hr.index[1], "utc_time"] = pd.Timestamp(
        "2024-01-10 02:00"
    )
    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        resolution=["1h", "daily"],
        aggregation="mean",
        time_reference="local",
    )

    coverage = "heart_rate_mean__1h__daily_mean__coverage"
    assert result.index.names == ["user_id", "calendar_date"]
    assert result.iloc[0][coverage] == pytest.approx(2 / 3)


def test_auto_hierarchy_uses_one_local_clock_for_calendar_parents():
    loader = HeartRateCycleLoader()
    heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-11",
        resolution=["1h", "daily"],
        aggregation="mean",
    )

    assert loader.requested_time_cols == ["local_time"]


def test_utc_fixed_windows_cannot_feed_local_sleep_cycles():
    with pytest.raises(ValueError, match="require local time"):
        heart_rate.get_heart_rate_statistics(
            HeartRateCycleLoader(),
            statistic="heart_rate_mean",
            user_id="user-1",
            start_date="2024-01-10",
            end_date="2024-01-11",
            resolution=["1h", "cycle_night"],
            aggregation="mean",
            time_reference="utc",
        )


def test_utc_cannot_be_requested_for_a_primary_sleep_cycle():
    with pytest.raises(ValueError, match="require local time"):
        heart_rate.get_heart_rate_statistics(
            HeartRateCycleLoader(),
            statistic="heart_rate_mean",
            user_id="user-1",
            resolution="cycle_night",
            time_reference="utc",
        )


def test_cycle_bounds_select_labels_without_clipping_physical_intervals():
    result = heart_rate.get_heart_rate_statistics(
        HeartRateCycleLoader(),
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10 12:00",
        end_date="2024-01-11 12:00",
        resolution="cycle_night",
    )

    assert result.index.tolist() == [("user-1", pd.Timestamp("2024-01-10"))]
    assert result.iloc[0]["heart_rate_mean"] == 70

    monthly = heart_rate.get_heart_rate_statistics(
        HeartRateCycleLoader(),
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10 12:00",
        end_date="2024-01-11 12:00",
        resolution=["cycle_night", "monthly"],
        aggregation="mean",
    )
    assert monthly.iloc[0]["heart_rate_mean__cycle_night__monthly_mean"] == 70


def test_cycle_resolution_rejects_unused_secondary_aggregation():
    with pytest.raises(ValueError, match="requires a resolution hierarchy"):
        heart_rate.get_heart_rate_statistics(
            HeartRateCycleLoader(),
            statistic="heart_rate_mean",
            user_id="user-1",
            resolution="cycle_night",
            aggregation="mean",
        )


def test_timedelta_resolution_remains_supported():
    result = heart_rate.get_heart_rate_statistics(
        HeartRateCycleLoader(),
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-11",
        resolution=pd.Timedelta(hours=1),
        time_reference="local",
    )

    assert result.index.names == ["user_id", "local_time"]


def test_cycle_intervals_are_cached_when_the_cycle_is_a_later_level():
    loader = HeartRateCycleLoader()
    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-12",
        resolution=["5min", "1h", "cycle_night", "monthly"],
        aggregation=[["mean"], ["mean"], ["mean"]],
    )

    assert not result.empty
    assert loader.session_calls == 1


def test_cycle_kwargs_forward_the_midnight_missing_end_policy():
    loader = HeartRateCycleLoader()
    default = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_count",
        user_id="user-1",
        start_date="2024-01-11",
        end_date="2024-01-12",
        resolution="cycle_awake",
    )
    fallback = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_count",
        user_id="user-1",
        start_date="2024-01-11",
        end_date="2024-01-12",
        resolution="cycle_awake",
        cycle_kwargs={"missing_end": "midnight"},
    )

    assert pd.isna(default.iloc[0]["heart_rate_count"])
    assert fallback.iloc[0]["heart_rate_count"] == 0


def test_invalid_cycle_can_be_aggregated_without_bounds():
    loader = HeartRateCycleLoader()
    loader.sessions = loader.sessions.iloc[:1].copy()
    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        resolution=["cycle_awake", "monthly"],
        aggregation="mean",
    )

    feature = "heart_rate_mean__cycle_awake__monthly_mean"
    assert len(result) == 1
    assert pd.isna(result.iloc[0][feature])
    assert result.iloc[0][f"{feature}__coverage"] == 0


def test_heart_rate_std_is_available_for_scalar_and_cycle_windows():
    loader = HeartRateCycleLoader()
    scalar = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_std",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-12",
        resolution="daily",
        time_reference="local",
    )
    cycle_result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_std",
        user_id="user-1",
        start_date="2024-01-10",
        end_date="2024-01-12",
        resolution="cycle_night",
    )

    expected = pd.Series([60, 80]).std()
    assert scalar.loc[
        ("user-1", pd.Timestamp("2024-01-10")), "heart_rate_std"
    ] == pytest.approx(expected)
    assert cycle_result.loc[
        ("user-1", pd.Timestamp("2024-01-10")), "heart_rate_std"
    ] == pytest.approx(expected)


def test_fixed_windows_feeding_cycles_use_each_users_own_bounds():
    loader = MultiUserCycleLoader()
    heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="all",
        resolution=["1h", "cycle_night"],
        aggregation="mean",
    )

    assert loader.heart_rate_requests["user-1"] == (
        pd.Timestamp("2024-01-07 23:00"),
        pd.Timestamp("2024-01-13 07:00"),
    )
    assert loader.heart_rate_requests["user-2"] == (
        pd.Timestamp("2024-03-07 23:00"),
        pd.Timestamp("2024-03-13 07:00"),
    )


def test_unbounded_empty_heart_rate_hierarchy_preserves_index_contract():
    loader = HeartRateCycleLoader()
    loader.hr = loader.hr.iloc[0:0]
    result = heart_rate.get_heart_rate_statistics(
        loader,
        statistic="heart_rate_mean",
        user_id="user-1",
        resolution=["1h", "daily"],
        aggregation="mean",
    )

    assert result.empty
    assert result.index.names == ["user_id", "calendar_date"]
