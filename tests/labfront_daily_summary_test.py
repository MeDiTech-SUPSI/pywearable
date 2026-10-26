"""Daily Garmin revisions must not depend on export ordering."""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from pywearable.loader import LabfrontLoader


def _summary(duration=86_400_000, stress=21_120_000, average=20, qualifier="BALANCED"):
    return {
        "calendarDate": pd.Timestamp("2024-04-12"),
        "isoDate": pd.Timestamp("2024-04-12"),
        "unixTimestampInMs": 1712872800000,
        "dailySummaryId": "summary-1",
        "durationInMs": duration,
        "stressDurationInMs": stress,
        "averageStressInStressLevel": average,
        "maxStressInStressLevel": 80,
        "stressQualifier": qualifier,
    }


def _loader(monkeypatch, rows):
    loader = LabfrontLoader.__new__(LabfrontLoader)
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(loader, "get_data_from_datetime", lambda **kwargs: frame.copy())
    return loader


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("context", [False, True])
def test_longest_daily_snapshot_wins_independently_of_order(monkeypatch, order, context):
    partial = _summary(58_800_000, 16_980_000, average=60, qualifier="STRESSFUL")
    full = _summary()
    snapshots = [partial, full, full.copy()]
    rows = [snapshots[index] for index in order]
    if context:
        previous = _summary()
        previous["calendarDate"] = previous["isoDate"] = pd.Timestamp("2024-04-11")
        previous["unixTimestampInMs"] -= 86_400_000
        rows = [previous, *rows]
    loader = _loader(monkeypatch, rows)
    for feature, expected in [
        ("garmin_daily_stress_duration_seconds", 21_120),
        ("garmin_daily_average_stress", 20),
        ("garmin_daily_stress_qualifier", "BALANCED"),
    ]:
        result = loader.load_provider_feature(feature, "person").set_index("calendar_date")
        assert result.loc[pd.Timestamp("2024-04-12"), feature] == expected
    pd.testing.assert_frame_equal(loader.load_daily_summary("person"), pd.DataFrame(rows))


def test_identical_measurements_with_different_summary_ids_are_not_conflicts(monkeypatch):
    first, second = _summary(), _summary()
    second["dailySummaryId"] = "summary-2"
    loader = _loader(monkeypatch, [first, second])
    with warnings.catch_warnings(record=True) as caught:
        result = loader.load_provider_feature("garmin_daily_stress_duration_seconds", "person")
    assert not caught
    assert result["garmin_daily_stress_duration_seconds"].tolist() == [21_120]


def test_equal_coverage_conflicts_only_remove_disputed_values(monkeypatch):
    first, second = _summary(), _summary(stress=22_000_000, qualifier="STRESSFUL")
    for rows in ([first, second], [second, first]):
        loader = _loader(monkeypatch, rows)
        for feature in ["garmin_daily_stress_duration_seconds", "garmin_daily_stress_qualifier"]:
            with pytest.warns(UserWarning, match="Ambiguous Garmin daily-summary"):
                result = loader.load_provider_feature(feature, "person")
            assert result[feature].isna().all()
        with pytest.warns(UserWarning, match="Ambiguous Garmin daily-summary"):
            result = loader.load_provider_feature("garmin_daily_max_stress", "person")
        assert result["garmin_daily_max_stress"].tolist() == [80]


@pytest.mark.parametrize("hours", [23, 25])
def test_non_24_hour_daily_periods_are_not_rejected(monkeypatch, hours):
    full_duration = hours * 3_600_000
    loader = _loader(monkeypatch, [
        _summary(full_duration - 3_600_000, stress=10_000_000),
        _summary(full_duration),
    ])
    result = loader.load_provider_feature("garmin_daily_stress_duration_seconds", "person")
    assert result["garmin_daily_stress_duration_seconds"].tolist() == [21_120]


@pytest.mark.parametrize("duration", [None, 0, -1, np.inf])
def test_invalid_duration_does_not_outrank_known_coverage(monkeypatch, duration):
    loader = _loader(monkeypatch, [_summary(duration, stress=10_000_000), _summary()])
    result = loader.load_provider_feature("garmin_daily_stress_duration_seconds", "person")
    assert result["garmin_daily_stress_duration_seconds"].tolist() == [21_120]


def test_conflicting_snapshots_without_coverage_are_not_selected_arbitrarily(monkeypatch):
    rows = [_summary(stress=10_000_000), _summary()]
    for row in rows:
        row.pop("durationInMs")
    loader = _loader(monkeypatch, rows)
    with pytest.warns(UserWarning, match="Ambiguous Garmin daily-summary"):
        result = loader.load_provider_feature("garmin_daily_stress_duration_seconds", "person")
    assert result["garmin_daily_stress_duration_seconds"].isna().all()


def test_distinct_starting_periods_are_not_compared_by_duration(monkeypatch):
    first, second = _summary(3_600_000), _summary()
    second["isoDate"] += pd.Timedelta(hours=1)
    second["unixTimestampInMs"] += 3_600_000
    loader = _loader(monkeypatch, [first, second])
    with pytest.warns(UserWarning, match="Ambiguous Garmin daily-summary"):
        result = loader.load_provider_feature("garmin_daily_stress_duration_seconds", "person")
    assert result["garmin_daily_stress_duration_seconds"].isna().all()


def test_local_start_is_used_when_physical_start_is_unavailable(monkeypatch):
    rows = [_summary(58_800_000, stress=16_980_000), _summary()]
    for row in rows:
        row.pop("unixTimestampInMs")
    loader = _loader(monkeypatch, rows)
    result = loader.load_provider_feature("garmin_daily_stress_duration_seconds", "person")
    assert result["garmin_daily_stress_duration_seconds"].tolist() == [21_120]
