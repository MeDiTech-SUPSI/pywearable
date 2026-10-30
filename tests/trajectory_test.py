import datetime
from functools import partial

import numpy as np
import pandas as pd
import pytest

from pywearable import trajectory


def _trajectory(values, *, feature="metric", users=None, start="2024-01-01"):
    values = list(values)
    if users is None:
        users = ["user"] * len(values)
    dates = pd.date_range(start, periods=len(values), freq="D")
    index = pd.MultiIndex.from_arrays(
        [users, dates], names=["user_id", "calendar_date"]
    )
    return pd.DataFrame(
        {
            feature: values,
            f"{feature}__coverage": np.linspace(0.2, 1.0, len(values)),
        },
        index=index,
    )


def test_imputation_uses_only_observed_participant_history_and_keeps_coverage():
    frame = _trajectory([1.0, 3.0, np.nan, 7.0, np.nan])
    coverage = frame["metric__coverage"].copy()

    result = trajectory.impute_trajectory(
        frame, method="ma", window=2, min_periods=2
    )

    assert result.iloc[2]["metric"] == pytest.approx(2)
    assert pd.isna(result.iloc[4]["metric"])
    assert result["metric__is_imputed"].tolist() == [False, False, True, False, False]
    pd.testing.assert_series_equal(result["metric__coverage"], coverage)
    assert pd.isna(frame.iloc[2]["metric"])


def test_trajectory_accepts_an_optional_timezone_index_level():
    frame = _trajectory([1.0, 3.0, np.nan])
    index = pd.MultiIndex.from_arrays(
        [
            frame.index.get_level_values("user_id"),
            frame.index.get_level_values("calendar_date"),
            [pd.Timedelta(hours=1)] * len(frame),
        ],
        names=["user_id", "utc_time", "timezone"],
    )
    frame.index = index

    result = trajectory.impute_trajectory(
        frame, method="ma", window=2, min_periods=2
    )

    assert result.iloc[2]["metric"] == pytest.approx(2)
    assert result.index.equals(index)


def test_imputation_does_not_feed_imputed_values_back_into_later_windows():
    frame = _trajectory([1.0, 3.0, np.nan, np.nan])

    result = trajectory.impute_trajectory(
        frame, method="ma", window=2, min_periods=2
    )

    assert result.iloc[2]["metric"] == pytest.approx(2)
    assert pd.isna(result.iloc[3]["metric"])


def test_repeated_imputation_preserves_existing_fills_without_using_them():
    frame = _trajectory([1.0, 3.0, np.nan, np.nan])
    first = trajectory.impute_trajectory(
        frame, method="ma", window=2, min_periods=2
    )

    second = trajectory.impute_trajectory(
        first, method="ma", window=2, min_periods=2
    )

    assert second.iloc[2]["metric"] == pytest.approx(2)
    assert pd.isna(second.iloc[3]["metric"])
    assert second["metric__is_imputed"].tolist() == [False, False, True, False]


def test_explicit_user_fallback_uses_the_entire_participant_trajectory():
    frame = _trajectory([np.nan, 2.0, np.nan, 8.0])

    result = trajectory.impute_trajectory(
        frame,
        method="ma",
        window=2,
        min_periods=2,
        fallback="user_mean",
    )

    assert result["metric"].tolist() == [5.0, 2.0, 5.0, 8.0]
    assert result["metric__is_imputed"].tolist() == [True, False, True, False]


def test_participants_are_isolated_unless_population_imputation_is_requested():
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
    )
    index = pd.MultiIndex.from_arrays(
        [["a", "a", "b", "b"], dates], names=["user_id", "calendar_date"]
    )
    frame = pd.DataFrame(
        {
            "metric": [1.0, np.nan, 100.0, np.nan],
            "metric__coverage": [1.0, 0.0, 1.0, 0.0],
        },
        index=index,
    )

    personal = trajectory.impute_trajectory(frame, method="user_mean")
    population = trajectory.impute_trajectory(frame, method="population_mean")

    assert personal.loc[("a", pd.Timestamp("2024-01-02")), "metric"] == 1
    assert personal.loc[("b", pd.Timestamp("2024-01-02")), "metric"] == 100
    assert population.loc[("a", pd.Timestamp("2024-01-02")), "metric"] == 50.5
    assert population.loc[("b", pd.Timestamp("2024-01-02")), "metric"] == 50.5


def test_user_and_population_median_imputation():
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ]
    )
    index = pd.MultiIndex.from_arrays(
        [["a", "a", "a", "b", "b", "b"], dates],
        names=["user_id", "calendar_date"],
    )
    frame = pd.DataFrame(
        {
            "metric": [1.0, np.nan, 9.0, 100.0, np.nan, 200.0],
            "metric__coverage": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        },
        index=index,
    )

    personal = trajectory.impute_trajectory(frame, method="user_median")
    population = trajectory.impute_trajectory(frame, method="population_median")

    assert personal.loc[("a", pd.Timestamp("2024-01-02")), "metric"] == 5
    assert personal.loc[("b", pd.Timestamp("2024-01-02")), "metric"] == 150
    assert population.loc[("a", pd.Timestamp("2024-01-02")), "metric"] == 54.5
    assert population.loc[("b", pd.Timestamp("2024-01-02")), "metric"] == 54.5


def test_rolling_median_is_historical_and_nonrecursive():
    frame = _trajectory([1.0, 3.0, np.nan, np.nan])

    result = trajectory.impute_trajectory(
        frame, method="rolling_median", window=2, min_periods=2
    )

    assert result.iloc[2]["metric"] == pytest.approx(2)
    assert pd.isna(result.iloc[3]["metric"])


def test_cumulative_and_exponential_imputation_are_historical():
    frame = _trajectory([1.0, 3.0, np.nan, 5.0, np.nan])

    cumulative = trajectory.impute_trajectory(
        frame, method="cumulative_mean", min_periods=2
    )
    exponential = trajectory.impute_trajectory(
        frame, method="ema", window=2, min_periods=2
    )

    assert cumulative.iloc[2]["metric"] == pytest.approx(2)
    assert cumulative.iloc[4]["metric"] == pytest.approx(3)
    assert exponential.iloc[2]["metric"] == pytest.approx(7 / 3)
    assert exponential.iloc[4]["metric"] == pytest.approx(37 / 9)


def test_clock_imputation_uses_a_circular_mean_and_restores_the_target_date():
    frame = _trajectory(
        [
            pd.Timestamp("2023-12-31 23:30"),
            pd.Timestamp("2024-01-02 00:30"),
            pd.NaT,
        ],
        feature="sleep_bedtime",
    )

    result = trajectory.impute_trajectory(
        frame,
        method="ma",
        window=2,
        min_periods=2,
    )

    assert result.iloc[2]["sleep_bedtime"] == pd.Timestamp("2024-01-03 00:00")
    assert result.iloc[2]["sleep_bedtime__is_imputed"]


def test_clock_rolling_median_keeps_times_around_midnight_adjacent():
    frame = _trajectory(
        [
            pd.Timestamp("2023-12-31 23:00"),
            pd.Timestamp("2024-01-02 01:00"),
            pd.NaT,
        ],
        feature="sleep_bedtime",
    )

    result = trajectory.impute_trajectory(
        frame,
        method="rolling_median",
        window=2,
        min_periods=2,
    )

    assert result.iloc[2]["sleep_bedtime"] == pd.Timestamp("2024-01-03 00:00")


@pytest.mark.parametrize(
    "feature", ["sleep_wakeup", "sleep_getup", "sleep_midpoint"]
)
def test_wake_date_clock_imputation_keeps_afternoon_values_on_target_date(feature):
    frame = _trajectory(
        [
            pd.Timestamp("2024-01-01 13:00"),
            pd.Timestamp("2024-01-02 14:00"),
            pd.NaT,
        ],
        feature=feature,
    )

    result = trajectory.impute_trajectory(
        frame,
        method="ma",
        window=2,
        min_periods=2,
    )

    assert result.iloc[2][feature] == pd.Timestamp("2024-01-03 13:30")


@pytest.mark.parametrize("feature", ["sleep_bedtime", "sleep_sleep_onset"])
def test_previous_evening_clock_imputation_uses_the_previous_date(feature):
    frame = _trajectory(
        [
            pd.Timestamp("2023-12-31 20:00"),
            pd.Timestamp("2024-01-01 22:00"),
            pd.NaT,
        ],
        feature=feature,
    )

    result = trajectory.impute_trajectory(
        frame,
        method="ma",
        window=2,
        min_periods=2,
    )

    assert result.iloc[2][feature] == pd.Timestamp("2024-01-02 21:00")


def test_elapsed_time_window_uses_timestamps_instead_of_row_count():
    frame = _trajectory([1.0, 3.0, 9.0, np.nan])
    frame.index = pd.MultiIndex.from_arrays(
        [
            ["user"] * 4,
            pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-10", "2024-01-11"]
            ),
        ],
        names=["user_id", "calendar_date"],
    )

    result = trajectory.impute_trajectory(
        frame,
        method="ma",
        window="3D",
        min_periods=1,
    )

    assert result.iloc[3]["metric"] == 9


def test_historical_imputation_orders_each_participant_without_reordering_output():
    frame = _trajectory([np.nan, 1.0, 3.0])
    original_index = pd.MultiIndex.from_arrays(
        [
            ["user"] * 3,
            pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
        ],
        names=["user_id", "calendar_date"],
    )
    frame.index = original_index

    result = trajectory.impute_trajectory(
        frame,
        method="ma",
        window=2,
        min_periods=2,
    )

    assert result.index.equals(original_index)
    assert result.iloc[0]["metric"] == pytest.approx(2)


def test_trend_methods_are_independent_and_exclude_current_observation():
    frame = _trajectory([1.0, 2.0, 3.0, 4.0])

    result = trajectory.compute_trajectory_trend(
        frame,
        baseline_method="ma",
        baseline_window=2,
        min_periods_baseline=2,
        normal_range_method="cumulative_mean",
        min_periods_normal_range=2,
    )

    row = result.iloc[2]
    assert row["metric__trend_baseline"] == pytest.approx(1.5)
    assert row["metric__trend_baseline_support"] == 2
    assert row["metric__trend_normal_center"] == pytest.approx(1.5)
    assert row["metric__trend_normal_std"] == pytest.approx(np.sqrt(0.5))
    assert row["metric__trend_normal_support"] == 2
    assert row["metric__trend_deviation"] == pytest.approx(1.5)


def test_trend_ignores_imputed_observations_by_default():
    frame = _trajectory([1.0, 100.0, 3.0])
    frame["metric__is_imputed"] = [False, True, False]

    observed = trajectory.compute_trajectory_trend(
        frame,
        baseline_method="cumulative_mean",
        min_periods_baseline=2,
        normal_range_method="cumulative_mean",
        min_periods_normal_range=2,
    )
    including_imputed = trajectory.compute_trajectory_trend(
        frame,
        baseline_method="cumulative_mean",
        min_periods_baseline=2,
        normal_range_method="cumulative_mean",
        min_periods_normal_range=2,
        include_imputed=True,
    )

    assert pd.isna(observed.iloc[2]["metric__trend_baseline"])
    assert observed.iloc[2]["metric__trend_baseline_support"] == 1
    assert including_imputed.iloc[2]["metric__trend_baseline"] == pytest.approx(50.5)


def test_clock_trend_uses_circular_baselines_and_minute_deviations():
    frame = _trajectory(
        [
            pd.Timestamp("2023-12-31 23:30"),
            pd.Timestamp("2024-01-02 00:30"),
            pd.Timestamp("2024-01-03 00:15"),
        ],
        feature="sleep_bedtime",
    )

    result = trajectory.compute_trajectory_trend(
        frame,
        baseline_method="ma",
        baseline_window=2,
        min_periods_baseline=2,
        normal_range_method="ma",
        normal_range_window=2,
        min_periods_normal_range=2,
    )

    assert result.iloc[2]["sleep_bedtime__trend_baseline"] == pd.Timestamp(
        "2024-01-03 00:00"
    )
    assert result.iloc[2]["sleep_bedtime__trend_deviation"] == pytest.approx(15)
    assert result.iloc[2]["sleep_bedtime__trend_normal_std"] == pytest.approx(
        30.043003, rel=1e-5
    )


def test_trajectory_analysis_requires_explicit_coverage_columns():
    frame = _trajectory([1.0, np.nan]).drop(columns="metric__coverage")

    with pytest.raises(ValueError, match="missing coverage columns"):
        trajectory.impute_trajectory(frame, features="metric")


def test_weekday_aggregation_is_participant_specific_and_reports_support():
    frame = _trajectory(np.arange(14, dtype=float))

    result = trajectory.aggregate_trajectory_by_weekday(frame, "metric")

    monday = result.loc[("user", "Monday")]
    assert monday["metric"] == pytest.approx(3.5)
    assert monday["metric__support"] == 2
    assert monday["metric__coverage"] == pytest.approx(
        frame["metric__coverage"].iloc[[0, 7]].mean()
    )
    assert list(result.xs("user").index) == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]


def test_weekday_aggregation_uses_circular_clock_means():
    frame = _trajectory(
        [pd.Timestamp("2023-12-31 23:30"), pd.Timestamp("2024-01-08 00:30")],
        feature="sleep_bedtime",
    )
    frame.index = pd.MultiIndex.from_arrays(
        [
            ["user", "user"],
            pd.to_datetime(["2024-01-01", "2024-01-08"]),
        ],
        names=["user_id", "calendar_date"],
    )

    result = trajectory.aggregate_trajectory_by_weekday(frame, "sleep_bedtime")

    assert result.loc[("user", "Monday"), "sleep_bedtime"] == datetime.time(0, 0)


def test_weekday_custom_aggregation_preserves_context_support_and_input():
    frame = _trajectory([7.0, np.nan, 1.0, 20.0, 10.0])
    frame.index = pd.MultiIndex.from_arrays(
        [
            ["a", "a", "a", "b", "b"],
            pd.to_datetime([
                "2024-01-15", "2024-01-08", "2024-01-01",
                "2024-01-08", "2024-01-01",
            ]),
        ],
        names=["user_id", "calendar_date"],
    )
    original = frame.copy(deep=True)
    received = []

    def slope_per_day(values):
        assert values.name == "metric"
        assert values.index.names == ["user_id", "calendar_date"]
        assert values.index.is_monotonic_increasing
        received.append(values.copy())
        observed = values.dropna()
        dates = observed.index.get_level_values("calendar_date")
        elapsed = (dates[-1] - dates[0]) / pd.Timedelta(days=1)
        result = (observed.iloc[-1] - observed.iloc[0]) / elapsed
        values.iloc[0] = -1000.0
        return result

    result = trajectory.aggregate_trajectory_by_weekday(
        frame, "metric", aggregation=slope_per_day
    )

    assert result.loc[("a", "Monday"), "metric"] == pytest.approx(6 / 14)
    assert result.loc[("b", "Monday"), "metric"] == pytest.approx(10 / 7)
    assert [len(values) for values in received] == [3, 2]
    assert [values.isna().sum() for values in received] == [1, 0]
    for user in ("a", "b"):
        monday = result.loc[(user, "Monday")]
        assert monday["metric__support"] == 2
        assert monday["metric__coverage"] == pytest.approx(
            frame.xs(user)["metric__coverage"].mean()
        )
        assert pd.isna(result.loc[(user, "Tuesday"), "metric"])
    pd.testing.assert_frame_equal(frame, original)


def test_weekday_custom_aggregation_respects_imputation_option():
    frame = _trajectory([0.0, 100.0, 8.0])
    frame.index = pd.MultiIndex.from_product(
        [["user"], pd.date_range("2024-01-01", periods=3, freq="7D")],
        names=["user_id", "calendar_date"],
    )
    frame["metric__is_imputed"] = [False, True, False]

    def interquartile_range(values):
        return values.quantile(0.75) - values.quantile(0.25)

    for include_imputed, expected, support in ((False, 4.0, 2), (True, 50.0, 3)):
        result = trajectory.aggregate_trajectory_by_weekday(
            frame, "metric", aggregation=interquartile_range,
            include_imputed=include_imputed,
        )
        monday = result.loc[("user", "Monday")]
        assert monday["metric"] == expected
        assert monday["metric__support"] == support
        assert monday["metric__coverage"] == pytest.approx(
            frame["metric__coverage"].mean()
        )


def test_weekday_custom_aggregation_preserves_clock_values():
    frame = _trajectory(
        [pd.Timestamp("2023-12-31 23:30"), pd.Timestamp("2024-01-08 00:30")],
        feature="sleep_bedtime",
    )
    frame.index = pd.MultiIndex.from_product(
        [["user"], pd.date_range("2024-01-01", periods=2, freq="7D")],
        names=["user_id", "calendar_date"],
    )

    def latest_bedtime(values):
        return max(
            values.dropna(),
            key=lambda value: ((value.hour - 12) % 24, value.minute),
        )

    result = trajectory.aggregate_trajectory_by_weekday(
        frame, "sleep_bedtime", aggregation=latest_bedtime
    )

    assert result.loc[("user", "Monday"), "sleep_bedtime"] == datetime.time(0, 30)


@pytest.mark.parametrize("returned, feature, message", [
    ([1.0, 2.0], "metric", "one scalar"),
    ("invalid", "metric", "numeric scalar"),
    (3.0, "sleep_bedtime", "clock time"),
])
def test_weekday_custom_aggregation_validates_return_type(returned, feature, message):
    value = datetime.time(23, 30) if feature == "sleep_bedtime" else 1.0
    frame = _trajectory([value], feature=feature)

    def invalid_result(values):
        return returned

    with pytest.raises(TypeError, match=message):
        trajectory.aggregate_trajectory_by_weekday(
            frame, feature, aggregation=invalid_result
        )


@pytest.mark.parametrize("aggregation", [
    lambda values: values.mean(), partial(np.mean), "iqr", ["mean"],
])
def test_weekday_aggregation_rejects_unnamed_or_unsupported_operations(aggregation):
    with pytest.raises(ValueError):
        trajectory.aggregate_trajectory_by_weekday(
            _trajectory([1.0]), "metric", aggregation=aggregation
        )


def test_weekday_custom_aggregation_skips_empty_and_all_missing_series():
    def unexpected_call(values):
        raise AssertionError("Empty or all-missing series must not call the reducer")

    result = trajectory.aggregate_trajectory_by_weekday(
        _trajectory([np.nan]), "metric", aggregation=unexpected_call
    )

    assert result["metric"].isna().all()
    assert result["metric__support"].eq(0).all()


def test_weekday_custom_aggregation_can_return_missing():
    frame = _trajectory([1.0])

    def unavailable(values):
        return np.nan

    result = trajectory.aggregate_trajectory_by_weekday(
        frame, "metric", aggregation=unavailable
    )

    monday = result.loc[("user", "Monday")]
    assert pd.isna(monday["metric"])
    assert monday["metric__support"] == 1
    assert monday["metric__coverage"] == frame["metric__coverage"].iloc[0]


def test_weekday_custom_aggregation_propagates_callback_errors():
    def failing_reducer(values):
        raise RuntimeError("Custom calculation failed")

    with pytest.raises(RuntimeError, match="Custom calculation failed"):
        trajectory.aggregate_trajectory_by_weekday(
            _trajectory([1.0]), "metric", aggregation=failing_reducer
        )
