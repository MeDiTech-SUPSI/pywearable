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
        statistic=["sleep_TST", "sleep_bedtime"],
        resolution="weekly",
        aggregation="mean",
        start_date="2024-01-16",
        end_date="2024-01-25",
    )

    assert result.index.get_level_values("calendar_date").tolist() == pd.to_datetime(
        ["2024-01-16", "2024-01-22"]
    ).tolist()
    assert result["period_end"].tolist() == pd.to_datetime(
        ["2024-01-22", "2024-01-25"]
    ).tolist()
    assert result.iloc[0]["sleep_TST__nightly__weekly_mean"] == 430
    assert result.iloc[0]["sleep_TST__nightly__weekly_mean__coverage"] == pytest.approx(
        2 / 6
    )
    assert result.iloc[0]["sleep_bedtime__nightly__weekly_mean"] == datetime.time(0, 0)


def test_sleep_statistics_parameter_order_matches_domain_statistics_api():
    assert list(inspect.signature(sleep.get_sleep_statistics).parameters)[:6] == [
        "loader",
        "statistic",
        "user_id",
        "start_date",
        "end_date",
        "resolution",
    ]


def test_explicit_sleep_hierarchy_uses_equal_weighted_levels():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        resolution=["nightly", "weekly", "monthly"],
        aggregation=[["mean"], ["mean"]],
        start_date="2024-01-16",
        end_date="2024-01-24",
    )

    column = "sleep_TST__nightly__weekly_mean__monthly_mean"
    # Weekly means are 430 and 460. The monthly level gives each week one
    # vote (445), rather than pooling the three valid nights (440).
    assert result.iloc[0][column] == 445
    assert result.iloc[0][f"{column}__coverage"] == 1


def test_explicit_sleep_hierarchy_uses_feature_metadata_for_clock_times():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_bedtime",
        resolution=["nightly", "weekly"],
        aggregation=["mean", "min", "max"],
        start_date="2024-01-16",
        end_date="2024-01-22",
    )

    assert result.iloc[0]["sleep_bedtime__nightly__weekly_mean"] == datetime.time(0, 0)
    assert result.iloc[0]["sleep_bedtime__nightly__weekly_min"] == datetime.time(23, 30)
    assert result.iloc[0]["sleep_bedtime__nightly__weekly_max"] == datetime.time(0, 30)


def test_sleep_hierarchy_reindexes_missing_expected_nights_for_coverage():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        resolution=["nightly", "weekly"],
        aggregation="mean",
        start_date="2024-01-16",
        end_date="2024-01-22",
    )

    assert result.iloc[0]["sleep_TST__nightly__weekly_mean__coverage"] == pytest.approx(
        2 / 6
    )


def test_sleep_hierarchy_coverage_threshold_propagates_to_the_next_level():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        resolution=["nightly", "weekly", "monthly"],
        aggregation=[["mean"], ["mean"]],
        min_coverage=[None, 0.5, 0.75],
        start_date="2024-01-16",
        end_date="2024-01-25",
    )

    weekly = "sleep_TST__nightly__weekly_mean"
    monthly = f"{weekly}__monthly_mean"
    # The first partial week has 2/6 valid nights and is rejected; the second
    # has 2/3. One of two weekly children survives, so monthly coverage is 1/2
    # and the final 75% threshold rejects the monthly value.
    assert result.iloc[0][f"{monthly}__coverage"] == 0.5
    assert pd.isna(result.iloc[0][monthly])


def test_scalar_calendar_coverage_threshold_applies_to_calendar_level():
    result = sleep.get_weekly_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        aggregation="mean",
        min_coverage=0.5,
        start_date="2024-01-16",
        end_date="2024-01-22",
    )

    assert result.iloc[0]["sleep_TST__nightly__weekly_mean__coverage"] == pytest.approx(
        2 / 6
    )
    assert pd.isna(result.iloc[0]["sleep_TST__nightly__weekly_mean"])


def test_absolute_valid_night_threshold_is_distinct_from_coverage_ratio():
    kwargs = {
        "loader": _sleep_loader(),
        "user_id": "person-1",
        "statistic": "sleep_TST",
        "aggregation": "mean",
        "start_date": "2024-01-16",
        "end_date": "2024-01-20",
    }
    ratio_only = sleep.get_weekly_sleep_statistics(
        min_coverage=3 / 7,
        **kwargs,
    )
    count_required = sleep.get_weekly_sleep_statistics(
        min_coverage=3 / 7,
        min_valid_children=3,
        **kwargs,
    )

    feature = "sleep_TST__nightly__weekly_mean"
    coverage = f"{feature}__coverage"
    assert ratio_only.iloc[0][feature] == 430
    assert ratio_only.iloc[0][coverage] == 0.5
    assert pd.isna(count_required.iloc[0][feature])
    assert count_required.iloc[0][coverage] == 0.5


def test_valid_child_threshold_propagates_to_the_next_sleep_level():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        resolution=["nightly", "weekly", "monthly"],
        aggregation=[["mean"], ["mean"]],
        min_valid_children=[None, 3, 1],
        start_date="2024-01-16",
        end_date="2024-01-25",
    )

    monthly = "sleep_TST__nightly__weekly_mean__monthly_mean"
    assert pd.isna(result.iloc[0][monthly])
    assert result.iloc[0][f"{monthly}__coverage"] == 0


def test_nightly_sleep_coverage_is_binary_feature_availability():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic=["sleep_TST", "sleep_bedtime"],
        resolution="nightly",
        start_date="2024-01-16",
        end_date="2024-01-21",
    ).xs("person-1")

    assert result.index.tolist() == pd.date_range(
        "2024-01-16", "2024-01-21", inclusive="left"
    ).tolist()
    assert result["sleep_TST__coverage"].tolist() == [1, 1, 0, 0, 0]
    assert result["sleep_bedtime__coverage"].tolist() == [1, 1, 0, 0, 1]


def test_sleep_hierarchy_accepts_primary_availability_threshold():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        resolution=["nightly", "weekly"],
        aggregation="mean",
        min_coverage=[0.7, None],
        start_date="2024-01-16",
        end_date="2024-01-22",
    )

    assert result.iloc[0][
        "sleep_TST__nightly__weekly_mean__coverage"
    ] == pytest.approx(2 / 6)


def test_unbounded_empty_sleep_hierarchy_preserves_index_contract():
    result = sleep.get_sleep_statistics(
        _sleep_loader(empty=True),
        user_id="person-1",
        statistic="sleep_TST",
        resolution=["nightly", "weekly"],
        aggregation="mean",
    )

    assert result.empty
    assert result.index.names == ["user_id", "calendar_date"]


def test_unbounded_sleep_hierarchy_skips_users_without_observed_nights():
    source = _sleep_loader()

    def load_sleep_summary(user_id, **kwargs):
        if user_id == "person-2":
            return source.load_sleep_summary(user_id, **kwargs).iloc[0:0]
        return source.load_sleep_summary(user_id, **kwargs)

    loader = SimpleNamespace(
        get_user_ids=lambda: ["person-1", "person-2"],
        load_sleep_summary=load_sleep_summary,
        load_sleep_stage=source.load_sleep_stage,
    )
    result = sleep.get_sleep_statistics(
        loader,
        user_id="all",
        statistic="sleep_TST",
        resolution=["nightly", "weekly"],
        aggregation="mean",
    )

    assert result.index.get_level_values("user_id").unique().tolist() == [
        "person-1"
    ]


@pytest.mark.parametrize(
    ("wrapper", "resolution"),
    [
        (sleep.get_weekly_sleep_statistics, "weekly"),
        (sleep.get_monthly_sleep_statistics, "monthly"),
        (sleep.get_yearly_sleep_statistics, "yearly"),
    ],
)
def test_calendar_sleep_wrappers_match_generic_extraction(wrapper, resolution):
    kwargs = {
        "loader": _sleep_loader(),
        "statistic": "sleep_TST",
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
        statistic="sleep_TST",
        resolution="weekly",
        aggregation=["mean", "median", "min", "max", "p10", "p90"],
        start_date="2024-01-16",
        end_date="2024-01-22",
    )

    prefix = "sleep_TST__nightly__weekly_"
    assert result.iloc[0][f"{prefix}mean"] == 430
    assert result.iloc[0][f"{prefix}median"] == 430
    assert result.iloc[0][f"{prefix}min"] == 420
    assert result.iloc[0][f"{prefix}max"] == 440
    assert result.iloc[0][f"{prefix}p10"] == 422
    assert result.iloc[0][f"{prefix}p90"] == 438


def test_sleep_statistics_reject_sum_aggregation():
    with pytest.raises(ValueError, match="aggregation operations must be"):
        sleep.get_sleep_statistics(
            _sleep_loader(),
            statistic="sleep_TST",
            user_id="person-1",
            resolution="weekly",
            aggregation="sum",
        )


def test_ordered_sleep_times_use_sleep_day_not_calendar_dates():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_bedtime",
        resolution="weekly",
        aggregation=["min", "median", "max"],
        start_date="2024-01-16",
        end_date="2024-01-18",
    )

    prefix = "sleep_bedtime__nightly__weekly_"
    assert result.iloc[0][f"{prefix}min"] == datetime.time(23, 30)
    assert result.iloc[0][f"{prefix}median"] == datetime.time(0, 0)
    assert result.iloc[0][f"{prefix}max"] == datetime.time(0, 30)


def test_temporal_mean_is_circular_and_variability_is_minutes():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_bedtime",
        resolution="weekly",
        aggregation=["mean", "std"],
        start_date="2024-01-16",
        end_date="2024-01-18",
    )

    assert result.iloc[0]["sleep_bedtime__nightly__weekly_mean"] == datetime.time(0, 0)
    assert result.iloc[0]["sleep_bedtime__nightly__weekly_std"] == pytest.approx(
        30.043, abs=0.01
    )


def test_daily_sleep_statistics_are_the_nightly_trajectory():
    daily = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        resolution="daily",
        start_date="2024-01-16",
        end_date="2024-01-19",
    )
    nightly = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
        resolution="nightly",
        start_date="2024-01-16",
        end_date="2024-01-19",
    )

    pd.testing.assert_frame_equal(daily, nightly)


def test_monthly_sleep_statistics_return_both_requested_partial_months():
    result = sleep.get_sleep_statistics(
        _sleep_loader(),
        user_id="person-1",
        statistic="sleep_TST",
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
    column = "sleep_TST__nightly__monthly_mean"
    assert pd.isna(result.iloc[1][column])
    assert result.iloc[1][f"{column}__coverage"] == 0


def test_sleep_statistics_reject_noncalendar_aggregation_resolution():
    with pytest.raises(ValueError, match="resolution must be nightly"):
        sleep.get_sleep_statistics(
            _sleep_loader(), statistic="sleep_TST", resolution="6h"
        )


def test_sleep_statistics_reject_empty_calendar_date_interval():
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        sleep.get_sleep_statistics(
            _sleep_loader(),
            statistic="sleep_TST",
            start_date="2024-01-16 08:00",
            end_date="2024-01-16 20:00",
        )


def test_aggregated_empty_sleep_statistics_preserve_schema_and_requested_periods():
    result = sleep.get_sleep_statistics(
        _sleep_loader(empty=True),
        user_id="person-1",
        statistic="sleep_TST",
        resolution="weekly",
        start_date="2024-01-01",
        end_date="2024-01-08",
    )

    assert len(result) == 1
    assert list(result) == [
        "period_start",
        "period_end",
        "sleep_TST__nightly__weekly_mean",
        "sleep_TST__nightly__weekly_mean__coverage",
        "sleep_TST__nightly__weekly_std",
        "sleep_TST__nightly__weekly_std__coverage",
        "sleep_TST__nightly__weekly_median",
        "sleep_TST__nightly__weekly_median__coverage",
        "sleep_TST__nightly__weekly_min",
        "sleep_TST__nightly__weekly_min__coverage",
        "sleep_TST__nightly__weekly_max",
        "sleep_TST__nightly__weekly_max__coverage",
        "sleep_TST__nightly__weekly_p10",
        "sleep_TST__nightly__weekly_p10__coverage",
        "sleep_TST__nightly__weekly_p90",
        "sleep_TST__nightly__weekly_p90__coverage",
    ]
    assert pd.isna(result.iloc[0]["sleep_TST__nightly__weekly_mean"])
    assert result.iloc[0]["sleep_TST__nightly__weekly_mean__coverage"] == 0
