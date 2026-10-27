"""Custom reducers retain temporal context and shared hierarchy qualification."""

import datetime
from functools import partial

import numpy as np
import pandas as pd
import pytest

from pywearable import hierarchy


def slope_per_day(values):
    """Fit change per elapsed day, retaining gaps in the time axis."""
    observed = values.dropna()
    if len(observed) < 2:
        return np.nan
    dates = observed.index.get_level_values("window_start")
    elapsed = (dates - dates[0]) / pd.Timedelta(days=1)
    return float(np.polyfit(elapsed, observed.to_numpy(dtype=float), 1)[0])


def median_observed(values):
    """Summarize the preceding level without pooling its original samples."""
    return values.median()


def _children():
    days = pd.date_range("2026-04-06", periods=14)
    index = pd.MultiIndex.from_product(
        [["user-1"], days], names=["user_id", "window_start"]
    )
    return pd.DataFrame(
        {
            "period_start": days,
            "period_end": days + pd.Timedelta(days=1),
            "sleep_TST__nightly": [
                420, np.nan, np.nan, 450, np.nan, np.nan, 480,
                500, 520, 540, 560, 580, 600, 620,
            ],
        },
        index=index,
    )


def _parents(resolution):
    return hierarchy.calendar_windows(
        ["user-1"], resolution,
        {"user-1": (pd.Timestamp("2026-04-06"), pd.Timestamp("2026-04-20"))},
    )


def test_callable_normalization_and_stable_names():
    assert hierarchy.resolve_feature_operations(
        ["a", "b"], ["MEAN", slope_per_day, slope_per_day, "mean"]
    ) == {"a": ["mean", slope_per_day], "b": ["mean", slope_per_day]}
    assert hierarchy.aggregation_name(slope_per_day) == "slope_per_day"
    assert hierarchy.normalize_aggregation_levels(slope_per_day, 2) == [
        slope_per_day, slope_per_day
    ]
    levels = [[slope_per_day], [median_observed]]
    assert hierarchy.normalize_aggregation_levels(levels, 2) == levels
    assert hierarchy.warn_if_wide(["a", "b"], [["mean", slope_per_day], "mean"]) == 4
    with pytest.raises(ValueError, match="nest operation lists"):
        hierarchy.normalize_aggregation_levels([slope_per_day, median_observed], 2)


@pytest.mark.parametrize("operation", [lambda values: 1, partial(np.mean)])
def test_unnamed_callables_are_rejected(operation):
    with pytest.raises(ValueError, match="named callables"):
        hierarchy.resolve_feature_operations(["value"], operation)


def test_duplicate_names_cannot_overwrite_an_output():
    def mean(values):
        return values.mean()

    with pytest.raises(ValueError, match="unique within a level"):
        hierarchy.resolve_feature_operations(["value"], ["mean", mean])
    other = partial(np.mean)
    other.__name__ = "mean"
    with pytest.raises(ValueError, match="unique within a level"):
        hierarchy.resolve_feature_operations(["value"], [mean, other])


def test_callback_gets_sorted_named_series_with_gaps_and_cannot_mutate_input():
    values = _children()["sleep_TST__nightly"].iloc[[3, 1, 0]]
    original = values.copy()

    def inspect_series(series):
        assert series.name == values.name
        assert series.index.names == ["user_id", "window_start"]
        assert series.index.is_monotonic_increasing
        assert len(series) == 3 and series.isna().sum() == 1
        series.iloc[0] = -1000
        return 12.0

    assert hierarchy.aggregate_values(values, inspect_series) == 12
    pd.testing.assert_series_equal(values, original)


@pytest.mark.parametrize("result", [1, 1.5, np.int64(2), np.float32(2.5)])
def test_numeric_scalar_types_are_supported(result):
    def numeric_result(values):
        return result

    assert hierarchy.aggregate_values(pd.Series([1.0]), numeric_result) == result


def test_same_function_name_at_different_levels_keeps_distinct_implementations():
    def weekly_reducer(values):
        return values.mean()

    def monthly_reducer(values):
        return values.min()

    weekly_reducer.__name__ = monthly_reducer.__name__ = "summary"
    weekly, _ = hierarchy.aggregate_level(
        _children(), _parents("weekly"), "weekly", weekly_reducer,
    )
    monthly, _ = hierarchy.aggregate_level(
        weekly, _parents("monthly"), "monthly", monthly_reducer,
    )
    assert monthly["sleep_TST__nightly__weekly_summary__monthly_summary"].iloc[0] == 450


@pytest.mark.parametrize("threshold", [None, 0.4, 0.5])
def test_successive_callbacks_with_and_without_coverage(threshold):
    weekly, _ = hierarchy.aggregate_level(
        _children(), _parents("weekly"), "weekly",
        ["mean", slope_per_day], min_coverage=threshold,
    )
    feature = "sleep_TST__nightly__weekly_slope_per_day"
    np.testing.assert_allclose(weekly[feature + "__coverage"], [3 / 7, 1])
    if threshold == 0.5:
        assert pd.isna(weekly[feature].iloc[0])
    else:
        assert weekly[feature].iloc[0] == pytest.approx(10)
    assert weekly[feature].iloc[1] == pytest.approx(20)

    monthly, _ = hierarchy.aggregate_level(
        weekly, _parents("monthly"), "monthly", median_observed,
    )
    output = feature + "__monthly_median_observed"
    assert monthly[output].iloc[0] == pytest.approx(20 if threshold == 0.5 else 15)
    assert monthly[output + "__coverage"].iloc[0] == (0.5 if threshold == 0.5 else 1)
    assert "sleep_TST__nightly__weekly_mean__monthly_median_observed" in monthly


def test_count_requirement_rejects_callback_output_and_propagates_missingness():
    weekly, _ = hierarchy.aggregate_level(
        _children(), _parents("weekly"), "weekly", slope_per_day,
        min_valid_children=4,
    )
    monthly, _ = hierarchy.aggregate_level(
        weekly, _parents("monthly"), "monthly", median_observed,
        min_coverage=0.75, min_valid_children=2,
    )
    feature = "sleep_TST__nightly__weekly_slope_per_day__monthly_median_observed"
    assert pd.isna(monthly[feature].iloc[0])
    assert monthly[feature + "__coverage"].iloc[0] == 0.5


def test_custom_aggregation_keeps_participants_separate():
    first = _children()
    second = first.rename(index={"user-1": "user-2"})
    second["sleep_TST__nightly"] *= 2
    parents = _parents("weekly")
    result, _ = hierarchy.aggregate_level(
        pd.concat([first, second]),
        pd.concat([parents, parents.rename(index={"user-1": "user-2"})]),
        "weekly", slope_per_day,
    )
    feature = "sleep_TST__nightly__weekly_slope_per_day"
    np.testing.assert_allclose(result.xs("user-1")[feature], [10, 20])
    np.testing.assert_allclose(result.xs("user-2")[feature], [20, 40])


def test_all_missing_windows_do_not_invoke_the_callback():
    def must_not_run(values):
        raise AssertionError("No observations are available")

    values = _children()["sleep_TST__nightly"] * np.nan
    assert pd.isna(hierarchy.aggregate_values(values, must_not_run))
    assert pd.isna(hierarchy.aggregate_values(values.iloc[:0], must_not_run))
    assert hierarchy.aggregate_values(values, must_not_run, temporal=True) is None


@pytest.mark.parametrize("result", [[1, 2], pd.Series([1]), np.array(1), "12", datetime.time(23)])
def test_numeric_callbacks_must_return_numeric_scalars(result):
    def invalid_result(values):
        return result

    with pytest.raises(TypeError, match="scalar"):
        hierarchy.aggregate_values(pd.Series([1.0]), invalid_result)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT])
def test_custom_callbacks_can_report_missing(missing):
    def unavailable(values):
        return missing

    assert pd.isna(hierarchy.aggregate_values(pd.Series([1]), unavailable))
    assert hierarchy.aggregate_values(
        pd.Series([datetime.time(23)]), unavailable, temporal=True
    ) is None


def test_clock_type_is_preserved_through_custom_and_builtin_levels():
    def latest_bedtime(values):
        observed = values.dropna()
        return max(observed, key=lambda value: (value.hour - 12) % 24 * 60 + value.minute)

    children = _children().iloc[:2].rename(columns={"sleep_TST__nightly": "bedtime"})
    children["bedtime"] = [datetime.time(23, 50), datetime.time(0, 10)]
    weekly, temporal = hierarchy.aggregate_level(
        children, _parents("weekly").iloc[:1], "weekly", latest_bedtime,
        temporal_features={"bedtime"},
    )
    feature = "bedtime__weekly_latest_bedtime"
    assert weekly[feature].iloc[0] == datetime.time(0, 10)
    assert temporal == {feature}
    monthly, temporal = hierarchy.aggregate_level(
        weekly, _parents("monthly"), "monthly", "mean", temporal_features=temporal,
    )
    assert monthly[feature + "__monthly_mean"].iloc[0] == datetime.time(0, 10)
    assert temporal == {feature + "__monthly_mean"}


@pytest.mark.parametrize("result", [12, 12.0, "23:00"])
def test_clock_callbacks_cannot_silently_return_another_type(result):
    def wrong_type(values):
        return result

    with pytest.raises(TypeError, match="clock time"):
        hierarchy.aggregate_values(
            pd.Series([datetime.time(23)]), wrong_type, temporal=True
        )


@pytest.mark.parametrize("result", [
    datetime.time(23), datetime.datetime(2026, 4, 6, 23),
    pd.Timestamp("2026-04-06 23:00"), np.datetime64("2026-04-06T23:00"),
])
def test_datetime_callback_results_keep_clock_semantics(result):
    def selected_bedtime(values):
        return result

    assert hierarchy.aggregate_values(
        pd.Series([pd.Timestamp("2026-04-06 23:00")]), selected_bedtime,
        temporal=True,
    ) == datetime.time(23)


def test_callback_errors_are_not_silently_converted_to_missing():
    def broken_calculation(values):
        raise ValueError("calculation failed")

    with pytest.raises(ValueError, match="calculation failed"):
        hierarchy.aggregate_values(pd.Series([1.0]), broken_calculation)
