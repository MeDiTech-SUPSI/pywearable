import datetime
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pywearable import sleep


def _sleep_loader(empty=False):
    summaries = pd.DataFrame(
        {
            "sleep_summary_id": ["n1", "n2", "n3", "n4", "n5"],
            "calendar_date": pd.to_datetime(
                ["2024-01-16", "2024-01-17", "2024-01-20", "2024-01-22", "2024-01-24"]
            ),
            "local_time": pd.to_datetime(
                [
                    "2024-01-15 23:30",
                    "2024-01-17 00:30",
                    "2024-01-20 00:00",
                    "2024-01-21 23:00",
                    "2024-01-24 01:00",
                ]
            ),
            "duration_in_ms": pd.Series([420, 440, np.nan, 460, 480]) * 60_000,
            "awake_duration_in_ms": [0, 0, 0, 0, 0],
            "unmeasurable_sleep_duration_in_ms": [0, 0, 0, 0, 0],
        }
    )
    if empty:
        summaries = summaries.iloc[0:0]

    def load_sleep_summary(user_id, start_date=None, end_date=None, **kwargs):
        result = summaries.copy()
        if start_date is not None:
            result = result[
                result["calendar_date"] >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            result = result[
                result["calendar_date"] < pd.Timestamp(end_date).normalize()
            ]
        return result.reset_index(drop=True)

    return SimpleNamespace(
        get_user_ids=lambda: ["person-1"],
        load_sleep_summary=load_sleep_summary,
        load_sleep_stage=lambda **kwargs: pd.DataFrame(
            columns=["sleep_summary_id", "local_time", "duration_in_ms", "type"]
        ),
    )


def test_weekly_sleep_statistics_keep_partial_periods_and_coverage():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic=["TST", "bedtime"],
        resolution="weekly",
        aggregation={"TST": "mean", "bedtime": "mean"},
        start_date="2024-01-16",
        end_date="2024-01-25",
    )

    assert result.index.get_level_values("calendar_date").tolist() == pd.to_datetime(
        ["2024-01-16", "2024-01-22"]
    ).tolist()
    assert result["period_end"].tolist() == pd.to_datetime(
        ["2024-01-22", "2024-01-25"]
    ).tolist()
    assert result["expected_nights"].tolist() == [6, 3]
    assert result.iloc[0]["TST_mean"] == 430
    assert result.iloc[0]["TST_count"] == 2
    assert result.iloc[0]["TST_coverage"] == pytest.approx(2 / 6 * 100)
    assert result.iloc[0]["bedtime_mean"] == datetime.time(0, 0)


def test_sleep_statistics_parameter_order_matches_domain_statistics_api():
    assert list(inspect.signature(sleep.get_sleep_statistics).parameters)[:6] == [
        "loader",
        "statistic",
        "user_id",
        "start_date",
        "end_date",
        "resolution",
    ]


@pytest.mark.parametrize(
    ("wrapper", "resolution"),
    [
        (sleep.get_daily_sleep_statistics, "daily"),
        (sleep.get_weekly_sleep_statistics, "weekly"),
        (sleep.get_monthly_sleep_statistics, "monthly"),
        (sleep.get_yearly_sleep_statistics, "yearly"),
    ],
)
def test_calendar_sleep_wrappers_match_generic_extraction(wrapper, resolution):
    kwargs = {
        "loader": _sleep_loader(),
        "statistic": "TST",
        "user_id": "person-1",
        "start_date": "2024-01-16",
        "end_date": "2024-01-25",
        "aggregation": ["mean", "p90"],
    }
    expected = sleep.get_sleep_statistics(resolution=resolution, **kwargs)
    result = wrapper(**kwargs)

    pd.testing.assert_frame_equal(result, expected)


def test_sleep_statistics_support_multiple_operations_and_percentiles():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="TST",
        resolution="weekly",
        aggregation=["mean", "median", "min", "max", "p10", "p90"],
        start_date="2024-01-16",
        end_date="2024-01-22",
    )

    assert result.iloc[0]["TST_mean"] == 430
    assert result.iloc[0]["TST_median"] == 430
    assert result.iloc[0]["TST_min"] == 420
    assert result.iloc[0]["TST_max"] == 440
    assert result.iloc[0]["TST_p10"] == 422
    assert result.iloc[0]["TST_p90"] == 438


def test_ordered_sleep_times_use_sleep_day_not_calendar_dates():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="bedtime",
        resolution="weekly",
        aggregation=["min", "median", "max"],
        start_date="2024-01-16",
        end_date="2024-01-18",
    )

    assert result.iloc[0]["bedtime_min"] == datetime.time(23, 30)
    assert result.iloc[0]["bedtime_median"] == datetime.time(0, 0)
    assert result.iloc[0]["bedtime_max"] == datetime.time(0, 30)


def test_temporal_mean_is_circular_and_variability_is_minutes():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="bedtime",
        resolution="weekly",
        aggregation=["mean", "std"],
        start_date="2024-01-16",
        end_date="2024-01-18",
    )

    assert result.iloc[0]["bedtime_mean"] == datetime.time(0, 0)
    assert result.iloc[0]["bedtime_std"] == pytest.approx(30.043, abs=0.01)


def test_opposite_clock_times_have_no_unique_circular_mean():
    values = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00"]))
    calendar_dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-01"]))

    assert sleep._aggregate_temporal_sleep_statistic(
        values, calendar_dates, "mean"
    ) is None
    assert pd.isna(
        sleep._aggregate_temporal_sleep_statistic(values, calendar_dates, "std")
    )


def test_daily_sleep_statistics_preserve_nightly_values_and_missing_days():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="TST",
        resolution="daily",
        start_date="2024-01-16",
        end_date="2024-01-19",
    )

    assert result["TST_mean"].tolist()[:2] == [420, 440]
    assert pd.isna(result.iloc[2]["TST_mean"])
    assert result["TST_count"].tolist() == [1, 1, 0]
    assert result["TST_coverage"].tolist() == [100, 100, 0]


def test_monthly_sleep_statistics_return_both_requested_partial_months():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="TST",
        resolution="monthly",
        start_date="2024-01-15",
        end_date="2024-02-15",
    )

    assert result.index.get_level_values("calendar_date").tolist() == pd.to_datetime(
        ["2024-01-15", "2024-02-01"]
    ).tolist()
    assert result["period_end"].tolist() == pd.to_datetime(
        ["2024-02-01", "2024-02-15"]
    ).tolist()
    assert result["expected_nights"].tolist() == [17, 14]
    assert pd.isna(result.iloc[1]["TST_mean"])
    assert result.iloc[1]["TST_coverage"] == 0


def test_sleep_statistics_reject_noncalendar_aggregation_resolution():
    with pytest.raises(ValueError, match="resolution must be nightly"):
        sleep.get_sleep_statistics(
            _sleep_loader(), statistic="TST", resolution="6h"
        )


def test_aggregated_empty_sleep_statistics_preserve_schema_and_requested_periods():
    result = sleep.get_sleep_statistics(
        _sleep_loader(empty=True),
        user_id="person-1",
        statistic="TST",
        resolution="weekly",
        start_date="2024-01-01",
        end_date="2024-01-08",
    )

    assert len(result) == 1
    assert list(result) == [
        "period_end",
        "expected_nights",
        "TST_mean",
        "TST_count",
        "TST_coverage",
    ]
    assert pd.isna(result.iloc[0]["TST_mean"])
    assert result.iloc[0]["TST_count"] == 0
    assert result.iloc[0]["TST_coverage"] == 0
