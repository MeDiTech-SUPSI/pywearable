import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pywearable import hrv, schema


class RawBBILoader:
    hrv_source = "bbi"
    hrv_time_columns = ("utc_time", "local_time")

    def __init__(self, sample_count=600, stage_type="asleep"):
        timestamps = pd.date_range("2024-01-01 23:00", periods=sample_count, freq="s")
        self.bbi = pd.DataFrame(
            {
                "utc_time": timestamps - pd.Timedelta(hours=1),
                "local_time": timestamps,
                "timezone": pd.Timedelta(hours=1),
                "bbi": np.resize([900.0, 1100.0], sample_count),
                "source": "test_bbi",
            }
        )
        self.stage_type = stage_type

    def get_user_ids(self):
        return ["user"]

    def load_bbi(self, user_id, start_date=None, end_date=None):
        return self.bbi.copy()

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "id": ["night"],
                "calendar_date": [pd.Timestamp("2024-01-02")],
                "type": ["main_sleep"],
                "main_sleep": [True],
                "local_time": [pd.Timestamp("2024-01-01 23:00")],
                "local_end_time": [pd.Timestamp("2024-01-01 23:10")],
                "time_in_bed_seconds": [600.0],
            }
        )

    def load_sleep_stage(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "sleep_summary_id": ["night"],
                "local_time": [pd.Timestamp("2024-01-01 23:00")],
                "duration_in_ms": [600_000],
                "type": [self.stage_type],
            }
        )


class ProviderHRVLoader(RawBBILoader):
    hrv_source = "provider_samples"
    hrv_time_columns = ("local_time",)

    def __init__(self, with_coverage=True):
        self.with_coverage = with_coverage
        self.bbi_called = False

    def load_bbi(self, *args, **kwargs):
        self.bbi_called = True
        raise AssertionError("provider loaders must not fall back to BBI")

    def load_hrv_samples(self, user_id, start_date=None, end_date=None):
        frame = pd.DataFrame(
            {
                "utc_time": [pd.NaT, pd.NaT],
                "local_time": pd.to_datetime(
                    ["2024-01-01 23:00", "2024-01-01 23:05"]
                ),
                "timezone": [pd.NaT, pd.NaT],
                "hrv_rmssd": [40.0, 44.0],
            }
        )
        if self.with_coverage:
            frame["hrv_coverage"] = [0.8, 0.6]
        return frame


class InternalWakeGapLoader(RawBBILoader):
    def __init__(self):
        super().__init__(sample_count=300)
        self.bbi["bbi"] = np.concatenate(
            [
                np.full(120, 900.0),
                np.full(60, 1000.0),
                np.full(120, 1100.0),
            ]
        )

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "id": ["night"],
                "calendar_date": [pd.Timestamp("2024-01-02")],
                "type": ["main_sleep"],
                "main_sleep": [True],
                "local_time": [pd.Timestamp("2024-01-01 23:00")],
                "local_end_time": [pd.Timestamp("2024-01-01 23:05")],
                "time_in_bed_seconds": [300.0],
            }
        )

    def load_sleep_stage(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "sleep_summary_id": ["night", "night", "night"],
                "local_time": pd.to_datetime(
                    [
                        "2024-01-01 23:00",
                        "2024-01-01 23:02",
                        "2024-01-01 23:03",
                    ]
                ),
                "duration_in_ms": [120_000, 60_000, 120_000],
                "type": ["light", "awake", "light"],
            }
        )


def test_time_domain_definitions_and_insufficient_sdnn():
    values = np.array([1000.0, 1030.0, 970.0, 1030.0])
    assert hrv.compute_pnn20(values) == 75.0
    assert hrv.compute_pnn50(values) == 50.0
    assert np.isnan(hrv.compute_sdnn(values))


def test_frequency_statistics_share_one_spectrum(monkeypatch):
    calls = 0
    original = hrv._power_spectrum

    def counted(values):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(hrv, "_power_spectrum", counted)
    hrv._window_statistics(
        np.resize([900.0, 1100.0], 300),
        ["hrv_vlf_power", "hrv_lf_power", "hrv_hf_power", "hrv_lfhf_ratio"],
        "pywearable",
    )
    assert calls == 1


def test_neurokit_frequency_uses_absolute_power_and_four_hz(monkeypatch):
    frequency_kwargs = {}

    def hrv_frequency(intervals, **kwargs):
        frequency_kwargs.update(kwargs)
        return pd.DataFrame(
            {
                "HRV_VLF": [1.0],
                "HRV_LF": [2.0],
                "HRV_HF": [4.0],
                "HRV_LFHF": [0.5],
            }
        )

    fake_neurokit = SimpleNamespace(hrv_frequency=hrv_frequency)
    monkeypatch.setitem(sys.modules, "neurokit2", fake_neurokit)

    result = hrv._neurokit_window_statistics(
        np.resize([900.0, 1100.0], 300),
        ["hrv_vlf_power", "hrv_lf_power", "hrv_hf_power", "hrv_lfhf_ratio"],
    )

    assert frequency_kwargs["normalize"] is False
    assert frequency_kwargs["interpolation_rate"] == 4
    assert result["hrv_lfhf_ratio"] == 0.5


def test_neurokit_rejects_frequency_and_avoids_cross_gap_differences(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("NeuroKit must not receive a discontinuous BBI sequence")

    fake_neurokit = SimpleNamespace(
        hrv_time=unexpected_call,
        hrv_frequency=unexpected_call,
    )
    monkeypatch.setitem(sys.modules, "neurokit2", fake_neurokit)
    values = np.concatenate([np.full(60, 900.0), np.full(60, 1100.0)])
    segments = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])

    result = hrv._window_statistics(
        values,
        ["hrv_rmssd", "hrv_lf_power"],
        "neurokit",
        segment_ids=segments,
    )

    assert result["hrv_rmssd"] == 0
    assert np.isnan(result["hrv_lf_power"])


def test_raw_bbi_primary_coverage_filters_five_minute_feature():
    loader = RawBBILoader(sample_count=150)
    result = hrv.get_hrv_statistics(
        loader,
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-01 23:00",
        end_date="2024-01-01 23:05",
        resolution="5min",
        time_reference="local",
        min_coverage=0.7,
    )
    assert result.iloc[0]["hrv_rmssd__coverage"] == pytest.approx(0.5)
    assert np.isnan(result.iloc[0]["hrv_rmssd"])


def test_raw_bbi_night_hierarchy_uses_equal_five_minute_windows():
    result = hrv.get_hrv_statistics(
        RawBBILoader(),
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution=["5min", "cycle_night"],
        aggregation="mean",
        min_coverage=[0.7, None],
    )
    feature = "hrv_rmssd__5min__cycle_night_mean"
    assert result.loc[("user", pd.Timestamp("2024-01-02")), feature] == pytest.approx(200)
    assert result.loc[
        ("user", pd.Timestamp("2024-01-02")), f"{feature}__coverage"
    ] == 1


def test_raw_bbi_wake_filtering_is_limited_to_night_cycles():
    loader = RawBBILoader(stage_type="awake")
    direct = hrv.get_hrv_statistics(
        loader,
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-01 23:00",
        end_date="2024-01-01 23:10",
        resolution="5min",
        time_reference="local",
        min_coverage=0.7,
    )
    nightly = hrv.get_hrv_statistics(
        loader,
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution=["5min", "cycle_night"],
        aggregation="mean",
        min_coverage=[0.7, None],
    )

    assert direct["hrv_rmssd"].notna().all()
    assert nightly["hrv_rmssd__5min__cycle_night_mean"].isna().all()


def test_internal_wake_gap_is_not_treated_as_contiguous_bbi():
    result = hrv.get_hrv_statistics(
        InternalWakeGapLoader(),
        statistic=["hrv_rmssd", "hrv_pnn50", "hrv_lf_power"],
        user_id="user",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution=["5min", "cycle_night"],
        aggregation="mean",
        min_coverage=[0.7, None],
    )
    row = result.loc[("user", pd.Timestamp("2024-01-02"))]

    assert row["hrv_rmssd__5min__cycle_night_mean"] == 0
    assert row["hrv_pnn50__5min__cycle_night_mean"] == 0
    assert np.isnan(row["hrv_lf_power__5min__cycle_night_mean"])


def test_provider_source_does_not_fall_back_and_unsupported_features_are_missing():
    loader = ProviderHRVLoader()
    result = hrv.get_hrv_statistics(
        loader,
        statistic=["hrv_rmssd", "hrv_sdnn"],
        user_id="user",
        start_date="2024-01-01 23:00",
        end_date="2024-01-01 23:10",
        resolution="5min",
        time_reference="local",
    )
    assert list(result["hrv_rmssd"]) == [40.0, 44.0]
    assert result["hrv_sdnn"].isna().all()
    assert not loader.bbi_called


def test_unknown_provider_coverage_warns_and_rejects_only_when_requested():
    loader = ProviderHRVLoader(with_coverage=False)
    baseline = hrv.get_hrv_statistics(
        loader,
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-01 23:00",
        end_date="2024-01-01 23:10",
        resolution="5min",
        time_reference="local",
    )
    assert baseline["hrv_rmssd"].notna().all()
    assert baseline["hrv_rmssd__coverage"].isna().all()
    with pytest.warns(UserWarning, match="Primary HRV coverage could not be determined"):
        strict = hrv.get_hrv_statistics(
            loader,
            statistic="hrv_rmssd",
            user_id="user",
            start_date="2024-01-01 23:00",
            end_date="2024-01-01 23:10",
            resolution="5min",
            time_reference="local",
            min_coverage=0.7,
        )
    assert strict["hrv_rmssd"].isna().all()
    assert strict["hrv_rmssd__coverage"].isna().all()


def test_explicit_utc_is_rejected_for_sleep_defined_cycles():
    with pytest.raises(ValueError, match="require local time"):
        hrv.get_hrv_statistics(
            RawBBILoader(),
            statistic="hrv_rmssd",
            user_id="user",
            resolution=["5min", "cycle_night"],
            time_reference="utc",
            aggregation="mean",
        )


def test_short_resolution_helpers_use_the_general_api(monkeypatch):
    calls = []
    expected = object()

    def fake_get_hrv_statistics(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(hrv, "get_hrv_statistics", fake_get_hrv_statistics)
    loader = RawBBILoader()

    assert hrv.get_five_minute_hrv_statistics(loader, min_coverage=0.6) is expected
    assert calls[-1]["resolution"] == "5min"
    assert calls[-1]["min_coverage"] == 0.6

    assert hrv.get_night_hrv_statistics(loader) is expected
    assert calls[-1]["resolution"] == ["5min", "cycle_night"]
    assert calls[-1]["aggregation"] == "mean"
    assert calls[-1]["min_coverage"] == [0.7, None]

    assert hrv.get_daily_hrv_statistics(loader, aggregation="median") is expected
    assert calls[-1]["resolution"] == "daily"
    assert calls[-1]["aggregation"] == "median"


def test_named_night_wrapper_rejects_non_mean_aggregation():
    with pytest.raises(ValueError, match="only support aggregation='mean'"):
        hrv.get_night_hrv_rmssd(RawBBILoader(), aggregation="median")


def test_named_night_wrapper_uses_the_general_api():
    result = hrv.get_night_hrv_rmssd(
        RawBBILoader(),
        user_id="user",
        start_date="2024-01-02",
        end_date="2024-01-03",
    )

    assert result.name == "hrv_rmssd"
    assert result.loc[("user", pd.Timestamp("2024-01-02"))] == pytest.approx(200)
