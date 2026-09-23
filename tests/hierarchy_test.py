import datetime

import numpy as np
import pandas as pd
import pytest

from pywearable import hierarchy


def _trajectory(values, starts, ends, feature="value"):
    index = pd.MultiIndex.from_arrays(
        [["user-1"] * len(starts), pd.to_datetime(starts)],
        names=["user_id", "window_start"],
    )
    return pd.DataFrame(
        {
            "period_start": pd.to_datetime(starts),
            "period_end": pd.to_datetime(ends),
            feature: values,
        },
        index=index,
    )


def test_resolution_chains_must_increase():
    with pytest.raises(ValueError, match="smaller to larger"):
        hierarchy.normalize_resolution_chain(["monthly", "daily"])
    with pytest.raises(ValueError, match="smaller to larger"):
        hierarchy.normalize_resolution_chain(["cycle_night", "daily"])


def test_single_item_resolution_sequence_is_not_a_hierarchy():
    assert hierarchy.normalize_resolution_chain(["1h"]) == (["1h"], False)


def test_calendar_aggregation_equal_weights_children_and_reports_coverage():
    trajectory = _trajectory(
        [10.0, 20.0, np.nan],
        ["2024-01-01", "2024-01-08", "2024-01-15"],
        ["2024-01-08", "2024-01-15", "2024-01-22"],
    )
    windows = hierarchy.calendar_windows(
        ["user-1"], "monthly", {"user-1": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"))}
    )
    result, _ = hierarchy.aggregate_level(
        trajectory,
        windows,
        target_resolution="monthly",
        aggregation=["mean", "p90"],
    )

    row = result.iloc[0]
    assert row["value__monthly_mean"] == 15
    assert row["value__monthly_p90"] == 19
    assert row["value__monthly_mean__coverage"] == pytest.approx(2 / 3)


def test_physical_membership_requires_complete_child_windows():
    trajectory = _trajectory(
        [1.0, 2.0],
        ["2024-01-01 22:55", "2024-01-01 23:00"],
        ["2024-01-01 23:00", "2024-01-01 23:05"],
    )
    parent = pd.DataFrame(
        {
            "start_time": [pd.Timestamp("2024-01-01 22:58")],
            "end_time": [pd.Timestamp("2024-01-01 23:05")],
        },
        index=pd.MultiIndex.from_tuples(
            [("user-1", pd.Timestamp("2024-01-02"))],
            names=["user_id", "window_start"],
        ),
    )
    result, _ = hierarchy.aggregate_level(
        trajectory,
        parent,
        target_resolution="cycle_night",
        aggregation="mean",
        membership="physical",
    )
    assert result.iloc[0]["value__cycle_night_mean"] == 2


def test_majority_overlap_membership_is_separate_from_feature_validity():
    trajectory = _trajectory(
        [1.0, np.nan, 9.0],
        [
            "2024-01-01 22:00",
            "2024-01-01 23:00",
            "2024-01-02 00:00",
        ],
        [
            "2024-01-01 23:00",
            "2024-01-02 00:00",
            "2024-01-02 01:00",
        ],
    )
    parent = pd.DataFrame(
        {
            "start_time": [pd.Timestamp("2024-01-01 22:30")],
            "end_time": [pd.Timestamp("2024-01-02 00:00")],
        },
        index=pd.MultiIndex.from_tuples(
            [("user-1", pd.Timestamp("2024-01-02"))],
            names=["user_id", "window_start"],
        ),
    )

    result, _ = hierarchy.aggregate_level(
        trajectory,
        parent,
        target_resolution="cycle_night",
        aggregation="mean",
        membership="majority_overlap",
    )

    row = result.iloc[0]
    assert row["value__cycle_night_mean"] == 1
    assert row["value__cycle_night_mean__coverage"] == 0.5


def test_majority_overlap_tie_uses_half_open_midpoint_assignment():
    trajectory = _trajectory(
        [10.0],
        ["2024-01-01 23:00"],
        ["2024-01-02 00:00"],
    )
    parent = pd.DataFrame(
        {
            "start_time": pd.to_datetime(
                ["2024-01-01 22:30", "2024-01-01 23:30"]
            ),
            "end_time": pd.to_datetime(
                ["2024-01-01 23:30", "2024-01-02 00:30"]
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("user-1", pd.Timestamp("2024-01-01")),
                ("user-1", pd.Timestamp("2024-01-02")),
            ],
            names=["user_id", "window_start"],
        ),
    )

    result, _ = hierarchy.aggregate_level(
        trajectory,
        parent,
        target_resolution="cycle_physiological",
        aggregation="mean",
        membership="majority_overlap",
    )

    feature = "value__cycle_physiological_mean"
    assert pd.isna(result.loc[("user-1", pd.Timestamp("2024-01-01")), feature])
    assert result.loc[("user-1", pd.Timestamp("2024-01-02")), feature] == 10


def test_membership_must_be_supported():
    trajectory = _trajectory(
        [1.0],
        ["2024-01-01 23:00"],
        ["2024-01-01 23:05"],
    )
    windows = hierarchy.fixed_windows(
        ["user-1"],
        "1h",
        {"user-1": (pd.Timestamp("2024-01-01 23:00"), pd.Timestamp("2024-01-02"))},
    )

    with pytest.raises(ValueError, match="membership"):
        hierarchy.aggregate_level(
            trajectory,
            windows,
            target_resolution="1h",
            aggregation="mean",
            membership="physcial",
        )


def test_hierarchy_aggregation_operations_apply_to_every_feature():
    operations = hierarchy.resolve_feature_operations(
        ["feature_a", "feature_b"], ["mean", "p90"]
    )
    assert operations == {
        "feature_a": ["mean", "p90"],
        "feature_b": ["mean", "p90"],
    }
    with pytest.raises(ValueError, match="strings or lists"):
        hierarchy.resolve_feature_operations(
            ["feature_a", "feature_b"], {"feature_a": "mean"}
        )


def test_temporal_mean_and_ordered_statistics_cross_midnight():
    values = pd.Series([datetime.time(23, 50), datetime.time(0, 10)])
    assert hierarchy.aggregate_values(
        values, "mean", temporal=True
    ) == datetime.time(0, 0)
    assert hierarchy.aggregate_values(
        values, "min", temporal=True
    ) == datetime.time(23, 50)
    assert hierarchy.aggregate_values(
        values, "max", temporal=True
    ) == datetime.time(0, 10)


def test_large_hierarchy_warns_before_execution():
    with pytest.warns(UserWarning, match="1,470"):
        hierarchy.warn_if_wide(
            [f"feature_{index}" for index in range(30)],
            [list(hierarchy.STANDARD_AGGREGATIONS)] * 2,
        )
