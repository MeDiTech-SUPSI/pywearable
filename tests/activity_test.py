import numpy as np
import pandas as pd
import pytest

from pywearable import activity, cycle, schema
from pywearable.loader.garmin import prepare_epoch_activity_samples


class SampleActivityLoader:
    activity_source = "samples"
    activity_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)
    activity_primary_resolution = "1min"

    def __init__(self, samples, heart_rate=None):
        self.samples = samples
        self.heart_rate = (
            schema.empty_heart_rate_frame()
            if heart_rate is None
            else heart_rate
        )

    def get_user_ids(self):
        return ["u1"]

    def load_activity_samples(self, user_id, start_date=None, end_date=None):
        return self.samples.copy()

    def load_heart_rate(
        self, user_id, start_date=None, end_date=None, time_col="local_time"
    ):
        return self.heart_rate.copy()


class DailyActivityLoader:
    activity_source = "daily_summary"
    activity_time_columns = (schema.LOCAL_TIME_COL,)
    activity_primary_resolution = "daily"

    def __init__(self, summary):
        self.summary = summary

    def get_user_ids(self):
        return ["u1"]

    def load_activity_summary(self, user_id, start_date=None, end_date=None):
        return self.summary.copy()


def _minute_samples(start, steps):
    times = pd.date_range(start, periods=len(steps), freq="1min")
    return pd.DataFrame(
        {
            schema.UTC_TIME_COL: times,
            schema.LOCAL_TIME_COL: times,
            schema.TIMEZONE_COL: pd.Timedelta(0),
            schema.DURATION_IN_MS_COL: 60_000,
            schema.ACTIVITY_STEPS_COL: steps,
        }
    )


def test_hourly_coverage_combines_hr_presence_and_positive_steps():
    samples = _minute_samples("2024-01-01", np.zeros(60))
    samples.loc[45, schema.ACTIVITY_STEPS_COL] = 25
    heart_rate_times = pd.date_range("2024-01-01", periods=30, freq="1min")
    heart_rate = pd.DataFrame(
        {
            schema.UTC_TIME_COL: heart_rate_times,
            schema.LOCAL_TIME_COL: heart_rate_times,
            schema.TIMEZONE_COL: pd.Timedelta(0),
            schema.HEART_RATE_COL: 60,
        }
    )
    loader = SampleActivityLoader(samples, heart_rate)

    result = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 01:00",
        resolution="1h",
    )

    assert list(result) == [
        activity.ACTIVITY_STATISTIC_STEPS,
        f"{activity.ACTIVITY_STATISTIC_STEPS}__coverage",
    ]
    row = result.iloc[0]
    assert row[activity.ACTIVITY_STATISTIC_STEPS] == 25
    assert row[f"{activity.ACTIVITY_STATISTIC_STEPS}__coverage"] == pytest.approx(
        31 / 60
    )

    rejected = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 01:00",
        resolution="1h",
        min_coverage=0.6,
    )
    assert pd.isna(rejected.iloc[0][activity.ACTIVITY_STATISTIC_STEPS])


def test_zero_steps_without_heart_rate_do_not_count_as_coverage():
    result = activity.get_activity_statistics(
        SampleActivityLoader(_minute_samples("2024-01-01", np.zeros(60))),
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 01:00",
        resolution="1h",
        min_coverage=0,
    )

    row = result.iloc[0]
    assert row[activity.ACTIVITY_STATISTIC_STEPS] == 0
    assert row[f"{activity.ACTIVITY_STATISTIC_STEPS}__coverage"] == 0


def test_hr_presence_covers_sparse_zero_step_rows():
    samples = _minute_samples("2024-01-01", [0])
    heart_rate_times = pd.date_range("2024-01-01", periods=60, freq="1min")
    heart_rate = pd.DataFrame(
        {
            schema.UTC_TIME_COL: heart_rate_times,
            schema.LOCAL_TIME_COL: heart_rate_times,
            schema.TIMEZONE_COL: pd.Timedelta(0),
            schema.HEART_RATE_COL: 60,
        }
    )
    result = activity.get_activity_statistics(
        SampleActivityLoader(samples, heart_rate),
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 01:00",
        resolution="1h",
        min_coverage=0,
    )

    row = result.iloc[0]
    assert row[activity.ACTIVITY_STATISTIC_STEPS] == 0
    assert row[f"{activity.ACTIVITY_STATISTIC_STEPS}__coverage"] == 1


def test_daily_peak_requires_a_complete_hour_and_returns_earliest_maximum():
    samples = _minute_samples(
        "2024-01-01", np.r_[np.ones(60), np.full(60, 2), np.zeros(60)]
    )
    result = activity.get_daily_activity_statistics(
        SampleActivityLoader(samples),
        statistic=[
            activity.ACTIVITY_STATISTIC_STEPS,
            activity.ACTIVITY_STATISTIC_PEAK_60MIN_STEPS,
            activity.ACTIVITY_STATISTIC_PEAK_60MIN_START,
        ],
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    row = result.iloc[0]
    assert row[activity.ACTIVITY_STATISTIC_STEPS] == 180
    assert row[activity.ACTIVITY_STATISTIC_PEAK_60MIN_STEPS] == 120
    assert row[activity.ACTIVITY_STATISTIC_PEAK_60MIN_START] == pd.Timestamp(
        "2024-01-01 01:00"
    )

    gapped = samples.drop(index=90).reset_index(drop=True)
    result = activity.get_daily_activity_statistics(
        SampleActivityLoader(gapped),
        statistic=activity.ACTIVITY_STATISTIC_PEAK_60MIN_STEPS,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    assert result.iloc[0][activity.ACTIVITY_STATISTIC_PEAK_60MIN_STEPS] < 120


def test_native_daily_zero_is_valid_and_weekly_defaults_to_sum():
    summary = pd.DataFrame(
        {
            schema.CALENDAR_DATE_COL: ["2024-01-01", "2024-01-03"],
            schema.ACTIVITY_STEPS_COL: [0, 300],
        }
    )
    loader = DailyActivityLoader(summary)
    daily = activity.get_daily_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        start_date="2024-01-01",
        end_date="2024-01-04",
        min_coverage=0,
    )
    assert daily.iloc[0][activity.ACTIVITY_STATISTIC_STEPS] == 0
    assert pd.isna(daily.iloc[1][activity.ACTIVITY_STATISTIC_STEPS])
    assert daily.iloc[1][f"{activity.ACTIVITY_STATISTIC_STEPS}__coverage"] == 0

    weekly = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        start_date="2024-01-01",
        end_date="2024-01-08",
        resolution="weekly",
        min_coverage=[None, 0],
    )
    assert weekly.iloc[0]["activity_steps__daily__weekly_sum"] == 300
    assert weekly.iloc[0][
        "activity_steps__daily__weekly_sum__coverage"
    ] == pytest.approx(2 / 7)

    mean = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        start_date="2024-01-01",
        end_date="2024-01-08",
        resolution=["daily", "weekly"],
        aggregation="mean",
    )
    assert mean.iloc[0]["activity_steps__daily__weekly_mean"] == 150


def test_sample_weekly_defaults_sum_steps_and_average_peak_descriptors():
    first = _minute_samples(
        "2024-01-01", np.r_[np.ones(60), np.zeros(60)]
    )
    second = _minute_samples(
        "2024-01-02", np.r_[np.full(60, 2), np.zeros(60)]
    )
    result = activity.get_activity_statistics(
        SampleActivityLoader(pd.concat([first, second], ignore_index=True)),
        start_date="2024-01-01",
        end_date="2024-01-08",
        resolution="weekly",
    )

    row = result.iloc[0]
    assert row["activity_steps__daily__weekly_sum"] == 180
    assert row["activity_peak_60min_steps__daily__weekly_mean"] == 90
    assert row["activity_peak_60min_start__daily__weekly_mean"] == pd.Timestamp(
        "2024-01-01"
    ).time()


def test_daily_summaries_reject_subdaily_and_peak_requests():
    loader = DailyActivityLoader(
        pd.DataFrame(
            {
                schema.CALENDAR_DATE_COL: ["2024-01-01"],
                schema.ACTIVITY_STEPS_COL: [100],
            }
        )
    )
    with pytest.raises(ValueError, match="only remain daily"):
        activity.get_activity_statistics(loader, resolution="1h")
    with pytest.raises(ValueError, match="requires timestamped"):
        activity.get_activity_statistics(
            loader, statistic=activity.ACTIVITY_STATISTIC_PEAK_60MIN_STEPS
        )


def test_fixed_windows_cannot_be_finer_than_or_misalign_native_epochs():
    loader = SampleActivityLoader(_minute_samples("2024-01-01", [1, 2]))
    loader.activity_primary_resolution = "15min"
    with pytest.raises(ValueError, match="integer multiple"):
        activity.get_activity_statistics(loader, resolution="5min")
    with pytest.raises(ValueError, match="integer multiple"):
        activity.get_activity_statistics(loader, resolution="20min")


def test_cycle_windows_exclude_epochs_crossing_their_boundaries(monkeypatch):
    times = pd.date_range("2024-01-01", periods=3, freq="15min")
    samples = pd.DataFrame(
        {
            schema.UTC_TIME_COL: times,
            schema.LOCAL_TIME_COL: times,
            schema.TIMEZONE_COL: pd.Timedelta(0),
            schema.DURATION_IN_MS_COL: 15 * 60_000,
            schema.ACTIVITY_STEPS_COL: [10, 20, 30],
        }
    )
    loader = SampleActivityLoader(samples)
    loader.activity_primary_resolution = "15min"
    intervals = pd.DataFrame(
        {
            "user_id": ["u1"],
            "calendar_date": [pd.Timestamp("2024-01-01")],
            "start_time": [pd.Timestamp("2024-01-01 00:10")],
            "end_time": [pd.Timestamp("2024-01-01 00:40")],
            "interval_valid": [True],
        }
    ).set_index(["user_id", "calendar_date"])
    monkeypatch.setattr(cycle, "get_intervals", lambda *args, **kwargs: intervals)

    result = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        resolution=cycle.CYCLE_NIGHT,
        min_coverage=0,
    )
    assert result.iloc[0][activity.ACTIVITY_STATISTIC_STEPS] == 20
    assert result.iloc[0][f"{activity.ACTIVITY_STATISTIC_STEPS}__coverage"] == 0.5


def test_garmin_epoch_categories_sum_steps_but_count_duration_once():
    frame = pd.DataFrame(
        {
            schema.LOCAL_TIME_COL: [
                pd.Timestamp("2024-01-01 12:00"),
                pd.Timestamp("2024-01-01 12:00"),
                pd.Timestamp("2024-01-01 12:00"),
            ],
            schema.UTC_TIME_COL: [
                pd.Timestamp("2024-01-01 11:00"),
                pd.Timestamp("2024-01-01 11:00"),
                pd.Timestamp("2024-01-01 11:00"),
            ],
            schema.TIMEZONE_COL: pd.Timedelta(hours=1),
            schema.DURATION_IN_MS_COL: 15 * 60_000,
            schema.ACTIVITY_STEPS_COL: [0, 98, 98],
            "intensity": ["SEDENTARY", "ACTIVE", "ACTIVE"],
            "activity_type": ["SEDENTARY", "WALKING", "WALKING"],
        }
    )

    result = prepare_epoch_activity_samples(frame)

    assert len(result) == 1
    assert result.iloc[0][schema.ACTIVITY_STEPS_COL] == 98
    assert result.iloc[0][schema.DURATION_IN_MS_COL] == 15 * 60_000


def test_garmin_epoch_category_conflict_marks_collapsed_steps_missing():
    timestamp = pd.Timestamp("2024-01-01 11:00")
    frame = pd.DataFrame(
        {
            schema.LOCAL_TIME_COL: timestamp + pd.Timedelta(hours=1),
            schema.UTC_TIME_COL: timestamp,
            schema.TIMEZONE_COL: pd.Timedelta(hours=1),
            schema.DURATION_IN_MS_COL: 15 * 60_000,
            schema.ACTIVITY_STEPS_COL: [98, 102],
            "intensity": "ACTIVE",
            "activity_type": "WALKING",
        }
    )

    result = prepare_epoch_activity_samples(frame)

    assert len(result) == 1
    assert pd.isna(result.iloc[0][schema.ACTIVITY_STEPS_COL])
