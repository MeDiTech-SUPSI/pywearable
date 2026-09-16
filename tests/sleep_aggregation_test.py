import pandas as pd
import pytest

from pywearable import sleep


def _nightly_features():
    index = pd.MultiIndex.from_product(
        [
            ["person-1"],
            pd.to_datetime(
                ["2024-01-16", "2024-01-17", "2024-01-20", "2024-01-22", "2024-01-24"]
            ),
        ],
        names=["user_id", "calendar_date"],
    )
    return pd.DataFrame(
        {
            "TST": [420, 440, None, 460, 480],
            "bedtime": pd.to_datetime(
                [
                    "2024-01-15 23:30",
                    "2024-01-17 00:30",
                    None,
                    "2024-01-21 23:00",
                    "2024-01-24 01:00",
                ]
            ),
        },
        index=index,
    )


def test_weekly_sleep_aggregation_keeps_partial_edge_periods_and_coverage():
    result = sleep.aggregate_sleep_statistics(
        _nightly_features(),
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
    assert result.iloc[0]["TST"] == 430
    assert result.iloc[0]["TST_count"] == 2
    assert result.iloc[0]["TST_coverage"] == pytest.approx(2 / 6 * 100)
    # 23:30 on the preceding date and 00:30 average to midnight, not noon.
    assert result.iloc[0]["bedtime"] == pd.Timestamp("2024-01-16 00:00")


def test_monthly_sleep_aggregation_returns_both_requested_partial_months():
    result = sleep.aggregate_sleep_statistics(
        _nightly_features()[["TST"]],
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
    assert pd.isna(result.iloc[1]["TST"])
    assert result.iloc[1]["TST_coverage"] == 0


def test_sleep_aggregation_supports_temporal_variability_in_minutes():
    result = sleep.aggregate_sleep_statistics(
        _nightly_features()[["bedtime"]],
        resolution="weekly",
        aggregation="std",
        start_date="2024-01-16",
        end_date="2024-01-22",
    )

    assert result.iloc[0]["bedtime"] == pytest.approx(42.42640687)


def test_sleep_aggregation_rejects_non_calendar_resolution():
    with pytest.raises(ValueError, match="calendar based"):
        sleep.aggregate_sleep_statistics(_nightly_features(), resolution="6h")
