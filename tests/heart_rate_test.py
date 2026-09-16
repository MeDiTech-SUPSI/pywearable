import pandas as pd
import pytest

from pywearable import heart_rate


class HeartRateLoader:
    def __init__(self, samples):
        self.samples = samples
        self.requested_time_cols = []

    def get_user_ids(self):
        return ["person-1"]

    def load_heart_rate(
        self, user_id, start_date=None, end_date=None, time_col="local_time"
    ):
        self.requested_time_cols.append(time_col)
        return self.samples.copy()


class LocalOnlyHeartRateLoader(HeartRateLoader):
    heart_rate_time_columns = ("local_time",)


def _samples():
    utc_time = pd.to_datetime(
        ["2024-01-01 22:30", "2024-01-01 23:30", "2024-01-02 00:30"]
    )
    return pd.DataFrame(
        {
            "utc_time": utc_time,
            "local_time": utc_time + pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "heart_rate": [60, 70, 80],
        }
    )


def test_subdaily_statistics_use_utc_windows_and_keep_timezone():
    loader = HeartRateLoader(_samples())
    result = heart_rate.get_heart_rate_statistics(
        loader,
        user_id="person-1",
        start_date="2024-01-01 22:00",
        end_date="2024-01-02 01:00",
        resolution="1h",
        statistic=["heart_rate_mean", "heart_rate_max"],
    )
    assert loader.requested_time_cols == ["utc_time"]
    assert result.index.names == ["user_id", "utc_time", "timezone"]
    assert result["heart_rate_mean"].dropna().tolist() == [60, 70, 80]


def test_daily_statistics_use_local_calendar_days():
    loader = HeartRateLoader(_samples())
    result = heart_rate.get_heart_rate_statistics(
        loader,
        user_id="person-1",
        start_date="2024-01-01",
        end_date="2024-01-03",
        resolution="daily",
        statistic="heart_rate_mean",
    )
    assert loader.requested_time_cols == ["local_time"]
    assert result.index.names == ["user_id", "local_time"]
    assert result.dropna().to_dict() == {
        ("person-1", pd.Timestamp("2024-01-01")): 60.0,
        ("person-1", pd.Timestamp("2024-01-02")): 75.0,
    }


def test_subdaily_statistics_can_use_local_clock_windows():
    loader = HeartRateLoader(_samples())
    result = heart_rate.get_heart_rate_statistics(
        loader,
        user_id="person-1",
        resolution="1h",
        statistic="heart_rate_mean",
        time_reference="local",
    )
    assert loader.requested_time_cols == ["local_time"]
    assert result.index.names == ["user_id", "local_time"]
    assert result.tolist() == [60, 70, 80]


def test_auto_subdaily_statistics_warn_and_fall_back_to_available_local_time():
    samples = _samples().assign(utc_time=pd.NaT, timezone=pd.NaT)
    loader = LocalOnlyHeartRateLoader(samples)

    with pytest.warns(UserWarning, match="using local-time windows"):
        result = heart_rate.get_heart_rate_statistics(
            loader,
            user_id="person-1",
            resolution="1h",
            statistic="heart_rate_mean",
        )

    assert loader.requested_time_cols == ["local_time"]
    assert result.index.names == ["user_id", "local_time"]
    assert result.tolist() == [60, 70, 80]


def test_explicit_utc_statistics_reject_local_only_loader():
    loader = LocalOnlyHeartRateLoader(_samples())

    with pytest.raises(ValueError, match="does not provide.*utc_time"):
        heart_rate.get_heart_rate_statistics(
            loader,
            user_id="person-1",
            resolution="1h",
            statistic="heart_rate_mean",
            time_reference="utc",
        )

    assert loader.requested_time_cols == []


def test_calendar_statistics_can_use_utc_days():
    loader = HeartRateLoader(_samples())
    result = heart_rate.get_heart_rate_statistics(
        loader,
        user_id="person-1",
        resolution="daily",
        statistic="heart_rate_mean",
        time_reference="utc",
    )
    assert loader.requested_time_cols == ["utc_time"]
    assert result.index.names == ["user_id", "utc_time"]
    assert result.to_dict() == {
        ("person-1", pd.Timestamp("2024-01-01")): 65.0,
        ("person-1", pd.Timestamp("2024-01-02")): 80.0,
    }


def test_invalid_time_reference_is_rejected():
    with pytest.raises(ValueError, match="time_reference"):
        heart_rate.get_heart_rate_statistics(
            HeartRateLoader(_samples()),
            user_id="person-1",
            time_reference="device",
        )


def test_requested_statistics_and_invalid_name():
    loader = HeartRateLoader(_samples())
    result = heart_rate.get_daily_heart_rate_statistics(
        loader,
        user_id="person-1",
        statistic=["heart_rate_count", "heart_rate_min", "heart_rate_p90"],
    )
    assert list(result) == ["heart_rate_count", "heart_rate_min", "heart_rate_p90"]
    with pytest.raises(ValueError, match="statistic must be one among"):
        heart_rate.get_heart_rate_statistics(
            loader, user_id="person-1", statistic="not_a_statistic"
        )


def test_non_daily_defaults_exclude_daily_rest_heart_rate():
    loader = HeartRateLoader(_samples())
    result = heart_rate.get_weekly_heart_rate_statistics(
        loader, user_id="person-1"
    )
    assert "heart_rate_rest" not in result
    with pytest.raises(ValueError, match="only at daily resolution"):
        heart_rate.get_hourly_heart_rate_statistics(
            loader, user_id="person-1", statistic="heart_rate_rest"
        )


@pytest.mark.parametrize(
    ("function", "expected_time_col"),
    [
        (heart_rate.get_minute_heart_rate_statistics, "utc_time"),
        (heart_rate.get_hourly_heart_rate_statistics, "utc_time"),
        (heart_rate.get_daily_heart_rate_statistics, "local_time"),
        (heart_rate.get_weekly_heart_rate_statistics, "local_time"),
        (heart_rate.get_monthly_heart_rate_statistics, "local_time"),
        (heart_rate.get_yearly_heart_rate_statistics, "local_time"),
    ],
)
def test_resolution_convenience_functions_remain_available(
    function, expected_time_col
):
    loader = HeartRateLoader(_samples())
    result = function(
        loader, user_id="person-1", statistic="heart_rate_mean"
    )
    assert isinstance(result, pd.Series)
    assert loader.requested_time_cols == [expected_time_col]


@pytest.mark.parametrize(
    "function",
    [
        heart_rate.get_daily_rest_heart_rate,
        heart_rate.get_daily_max_heart_rate,
        heart_rate.get_daily_min_heart_rate,
        heart_rate.get_daily_mean_heart_rate,
    ],
)
def test_daily_statistic_convenience_functions_remain_available(function):
    result = function(HeartRateLoader(_samples()), user_id="person-1")
    assert isinstance(result, pd.Series)


def test_conflicting_timestamps_are_excluded():
    samples = _samples()
    samples = pd.concat([samples, samples.iloc[[0]].assign(heart_rate=95)])
    samples["timestamp_conflict"] = samples.duplicated("utc_time", keep=False)
    loader = HeartRateLoader(samples)
    with pytest.warns(UserWarning, match="ambiguous"):
        result = heart_rate.get_hourly_heart_rate_statistics(
            loader,
            user_id="person-1",
            statistic="heart_rate_count",
        )
    assert result.sum() == 2


def test_identical_duplicate_timestamps_are_removed_for_every_statistic():
    utc_time = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:30"]
    )
    samples = pd.DataFrame(
        {
            "utc_time": utc_time,
            "local_time": utc_time + pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "heart_rate": [60, 60, 100],
        }
    )

    result = heart_rate.get_hourly_heart_rate_statistics(
        HeartRateLoader(samples),
        user_id="person-1",
        statistic=["heart_rate_count", "heart_rate_mean", "heart_rate_p90"],
    ).iloc[0]

    assert result["heart_rate_count"] == 2
    assert result["heart_rate_mean"] == 80
    assert result["heart_rate_p90"] == pytest.approx(96)


def test_resting_heart_rate_uses_local_days_and_twenty_valid_minutes():
    first_day_local = pd.date_range("2024-01-01 00:00", periods=21, freq="1min")
    second_day_local = pd.date_range("2024-01-02 00:00", periods=21, freq="1min")
    local_time = first_day_local.append(second_day_local)
    samples = pd.DataFrame(
        {
            "utc_time": local_time - pd.Timedelta(hours=1),
            "local_time": local_time,
            "timezone": pd.Timedelta(hours=1),
            "heart_rate": [50] * len(first_day_local) + [70] * len(second_day_local),
        }
    )

    result = heart_rate.get_daily_rest_heart_rate(
        HeartRateLoader(samples), user_id="person-1"
    )

    assert result.to_dict() == {
        ("person-1", pd.Timestamp("2024-01-01")): 50.0,
        ("person-1", pd.Timestamp("2024-01-02")): 70.0,
    }
