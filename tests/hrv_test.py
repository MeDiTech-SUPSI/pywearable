import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pywearable import hrv, schema
from tests.helpers import LoaderBase


class RawBBILoader(LoaderBase):
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


class SubperiodProviderHRVLoader(ProviderHRVLoader):
    def load_hrv_samples(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "utc_time": [pd.NaT] * 6,
                "local_time": pd.to_datetime(
                    [
                        "2024-01-01 23:00",
                        "2024-01-01 23:05",
                        "2024-01-01 23:10",
                        "2024-01-02 23:00",
                        "2024-01-02 23:05",
                        "2024-01-02 23:10",
                    ]
                ),
                "timezone": [pd.NaT] * 6,
                "hrv_rmssd": [1.0, 100.0, 1.0, 2.0, 2.0, 2.0],
                "hrv_coverage": [1.0] * 6,
            }
        )

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "id": ["night-1", "night-2"],
                "calendar_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "type": ["main_sleep", "main_sleep"],
                "main_sleep": [True, True],
                "local_time": pd.to_datetime(
                    ["2024-01-01 23:00", "2024-01-02 23:00"]
                ),
                "local_end_time": pd.to_datetime(
                    ["2024-01-01 23:15", "2024-01-02 23:15"]
                ),
                "time_in_bed_seconds": [900.0, 900.0],
            }
        )


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


class LoaderFilteredGapLoader(RawBBILoader):
    """Return BBIs after a provider-quality filter removed one source row."""

    def __init__(self):
        super().__init__(sample_count=300)
        self.bbi["bbi"] = np.concatenate(
            [np.full(150, 900.0), np.array([1000.0]), np.full(149, 1100.0)]
        )

    def load_bbi(self, user_id, start_date=None, end_date=None):
        return self.bbi.drop(index=150).reset_index(drop=True)


def test_time_domain_definitions_and_insufficient_sdnn():
    values = np.array([1000.0, 1030.0, 970.0, 1030.0])
    assert hrv.compute_pnn20(values) == 100.0
    assert hrv.compute_pnn50(values) == pytest.approx(2 / 3 * 100)
    assert np.isnan(hrv.compute_sdnn(values))


def test_builtin_backend_exposes_expanded_time_domain_features():
    values = np.resize([900.0, 1100.0], 300)
    requested = [
        "hrv_min_nn",
        "hrv_mean_nn",
        "hrv_max_nn",
        "hrv_sdnn_rmssd_ratio",
        "hrv_cv_nn",
        "hrv_cv_sd",
        "hrv_iqr_nn",
    ]

    result = hrv._window_statistics(values, requested, "pywearable")

    assert result["hrv_min_nn"] == 900
    assert result["hrv_mean_nn"] == 1000
    assert result["hrv_max_nn"] == 1100
    assert result["hrv_sdnn_rmssd_ratio"] == pytest.approx(
        hrv.compute_sdnn(values) / hrv.compute_rmssd(values)
    )
    assert result["hrv_cv_nn"] == pytest.approx(hrv.compute_sdnn(values) / 1000)
    assert result["hrv_cv_sd"] == pytest.approx(hrv.compute_rmssd(values) / 1000)
    assert result["hrv_iqr_nn"] == 200


def test_backend_default_registries_are_distinct():
    assert set(hrv.HRV_STATISTICS) < set(hrv.NEUROKIT_HRV_STATISTICS)
    assert "hrv_sdsd" in hrv.NEUROKIT_HRV_STATISTICS
    assert "hrv_total_power" in hrv.NEUROKIT_HRV_STATISTICS
    assert "hrv_sdsd" not in hrv.HRV_STATISTICS


def test_frequency_statistics_share_one_spectrum(monkeypatch):
    calls = 0
    original = hrv._power_spectrum

    def counted(values, timestamps=None):
        nonlocal calls
        calls += 1
        return original(values, timestamps)

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


def test_neurokit_returns_requested_time_and_frequency_features_only(monkeypatch):
    calls = []

    def hrv_time(intervals, **kwargs):
        calls.append("time")
        return pd.DataFrame(
            {
                "HRV_MeanNN": [1000.0],
                "HRV_SDSD": [12.0],
                "HRV_MadNN": [20.0],
            }
        )

    def hrv_frequency(intervals, **kwargs):
        calls.append("frequency")
        return pd.DataFrame({"HRV_TP": [123.0], "HRV_VHF": [4.0]})

    fake_neurokit = SimpleNamespace(
        hrv_time=hrv_time,
        hrv_frequency=hrv_frequency,
        hrv_nonlinear=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("nonlinear HRV must not be called")
        ),
    )
    monkeypatch.setitem(sys.modules, "neurokit2", fake_neurokit)

    requested = ["hrv_sdsd", "hrv_mad_nn", "hrv_total_power"]
    result = hrv._neurokit_window_statistics(
        np.resize([900.0, 1100.0], 300), requested
    )

    assert result == {
        "hrv_sdsd": 12.0,
        "hrv_mad_nn": 20.0,
        "hrv_total_power": 123.0,
    }
    assert calls == ["time", "frequency"]


def test_neurokit_receives_observed_times_for_discontinuous_windows(monkeypatch):
    received = []

    def hrv_time(intervals, **kwargs):
        received.append(("time", intervals["RRI_Time"].copy()))
        return pd.DataFrame({"HRV_RMSSD": [12.0]})

    def hrv_frequency(intervals, **kwargs):
        received.append(("frequency", intervals["RRI_Time"].copy()))
        return pd.DataFrame({"HRV_LF": [34.0]})

    fake_neurokit = SimpleNamespace(hrv_time=hrv_time, hrv_frequency=hrv_frequency)
    monkeypatch.setitem(sys.modules, "neurokit2", fake_neurokit)
    values = np.concatenate([np.full(60, 900.0), np.full(60, 1100.0)])
    segments = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])
    timestamps = pd.date_range("2024-01-01", periods=120, freq="s").to_series()
    timestamps.iloc[60:] += pd.Timedelta(seconds=3)

    result = hrv._window_statistics(
        values,
        ["hrv_rmssd", "hrv_lf_power"],
        "neurokit",
        segment_ids=segments,
        timestamps=timestamps,
    )

    assert result == {"hrv_rmssd": 12.0, "hrv_lf_power": 34.0}
    assert [name for name, _ in received] == ["time", "frequency"]
    for _, interval_times in received:
        assert interval_times[0] == 0
        assert interval_times[60] - interval_times[59] == 4


def test_builtin_frequency_uses_observed_times_across_a_gap(monkeypatch):
    observed_times = None
    original = hrv.interpolate.interp1d

    def capture_times(times, values, **kwargs):
        nonlocal observed_times
        observed_times = np.asarray(times)
        return original(times, values, **kwargs)

    monkeypatch.setattr(hrv.interpolate, "interp1d", capture_times)
    values = np.resize([900.0, 1100.0], 300)
    timestamps = pd.date_range("2024-01-01", periods=300, freq="s").to_series()
    timestamps.iloc[150:] += pd.Timedelta(seconds=3)

    result = hrv._window_statistics(
        values,
        ["hrv_lf_power"],
        "pywearable",
        segment_ids=np.concatenate(
            [np.zeros(150, dtype=int), np.ones(150, dtype=int)]
        ),
        timestamps=timestamps,
    )

    assert np.isfinite(result["hrv_lf_power"])
    assert observed_times[150] - observed_times[149] == 4


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


@pytest.mark.parametrize("injection", ["duplicates", "infinite"])
def test_invalid_or_repeated_bbi_cannot_inflate_coverage(injection):
    loader = RawBBILoader(sample_count=150)
    extra = loader.bbi.copy()
    if injection == "infinite":
        extra = extra.iloc[:1].copy()
        extra["bbi"] = np.inf
    loader.bbi = pd.concat([loader.bbi, extra], ignore_index=True)
    result = hrv.get_hrv_statistics(
        loader, statistic="hrv_rmssd", user_id="user", resolution="5min",
        start_date="2024-01-01 23:00", end_date="2024-01-01 23:05",
        time_reference="local", min_coverage=0.7,
    )
    assert result.iloc[0]["hrv_rmssd__coverage"] == pytest.approx(0.5)
    assert pd.isna(result.iloc[0]["hrv_rmssd"])


def test_primary_window_step_builds_overlapping_windows():
    loader = RawBBILoader()
    result = hrv.get_hrv_statistics(
        loader,
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-01 23:00",
        end_date="2024-01-01 23:10",
        resolution="5min",
        window_step="1min",
        time_reference="local",
    )

    assert len(result) == 10
    assert result.index.get_level_values("local_time")[1] == pd.Timestamp(
        "2024-01-01 23:01"
    )


def test_overlapping_primary_windows_aggregate_to_night():
    result = hrv.get_hrv_statistics(
        RawBBILoader(),
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution=["5min", "nightly"],
        window_step="1min",
        aggregation="mean",
        min_coverage=[0.7, None],
    )

    feature = "hrv_rmssd__5min__cycle_night_mean"
    assert result.loc[("user", pd.Timestamp("2024-01-02")), feature] == pytest.approx(
        200
    )
    assert result.loc[
        ("user", pd.Timestamp("2024-01-02")), f"{feature}__coverage"
    ] == 1


def test_provider_hrv_rejects_non_native_window_step():
    with pytest.raises(ValueError, match="provider-computed HRV"):
        hrv.get_hrv_statistics(
            ProviderHRVLoader(),
            statistic="hrv_rmssd",
            user_id="user",
            resolution="5min",
            window_step="1min",
        )


@pytest.mark.parametrize("window_step", ["0min", "6min"])
def test_window_step_must_fit_inside_primary_window(window_step):
    with pytest.raises(ValueError, match="window_step"):
        hrv.get_hrv_statistics(
            RawBBILoader(),
            statistic="hrv_rmssd",
            resolution="5min",
            window_step=window_step,
        )


def test_fixed_hrv_output_retains_both_clocks_and_each_window_offset():
    loader = RawBBILoader()
    loader.bbi.loc[:299, "timezone"] = pd.Timedelta(hours=1)
    loader.bbi.loc[300:, "timezone"] = pd.Timedelta(hours=2)
    loader.bbi["local_time"] = loader.bbi["utc_time"] + loader.bbi["timezone"]

    utc = hrv.get_hrv_statistics(
        loader,
        statistic="hrv_rmssd",
        user_id="user",
        resolution="5min",
    )

    assert utc.index.names == ["user_id", "utc_time"]
    assert utc["timezone"].tolist() == [
        pd.Timedelta(hours=1),
        pd.Timedelta(hours=2),
    ]
    assert utc["local_time"].tolist() == [
        pd.Timestamp("2024-01-01 23:00"),
        pd.Timestamp("2024-01-02 00:05"),
    ]

    local = hrv.get_hrv_statistics(
        RawBBILoader(),
        statistic="hrv_rmssd",
        user_id="user",
        resolution="5min",
        time_reference="local",
    )
    assert local.index.names == ["user_id", "local_time"]
    assert local["utc_time"].tolist() == [
        pd.Timestamp("2024-01-01 22:00"),
        pd.Timestamp("2024-01-01 22:05"),
    ]
    assert local["timezone"].eq(pd.Timedelta(hours=1)).all()


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


def test_nightly_and_cycle_night_are_complete_hrv_aliases():
    common = {
        "loader": RawBBILoader(),
        "statistic": "hrv_rmssd",
        "user_id": "user",
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "aggregation": "mean",
    }
    nightly = hrv.get_hrv_statistics(resolution="nightly", **common)
    cycle_night = hrv.get_hrv_statistics(resolution="cycle_night", **common)
    explicit_nightly = hrv.get_hrv_statistics(
        resolution=["5min", "nightly"], **common
    )
    explicit_cycle = hrv.get_hrv_statistics(
        resolution=["5min", "cycle_night"], **common
    )

    pd.testing.assert_frame_equal(nightly, cycle_night)
    pd.testing.assert_frame_equal(nightly, explicit_nightly)
    pd.testing.assert_frame_equal(nightly, explicit_cycle)


def _subperiod_hrv_results():
    loader = SubperiodProviderHRVLoader()
    common = {
        "loader": loader,
        "statistic": "hrv_rmssd",
        "user_id": "user",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
    }
    direct = hrv.get_hrv_statistics(resolution="monthly", **common)
    by_night = hrv.get_hrv_statistics(
        resolution=["5min", "cycle_night", "monthly"],
        aggregation=[["median", "max"], ["max"]],
        **common,
    )
    return direct.iloc[0], by_night.iloc[0]


def test_direct_max_differs_from_maximum_nightly_median():
    direct, by_night = _subperiod_hrv_results()

    assert direct["hrv_rmssd__5min__monthly_max"] == 100
    assert by_night[
        "hrv_rmssd__5min__cycle_night_median__monthly_max"
    ] == 2


def test_maximum_nightly_maximum_equals_direct_maximum():
    direct, by_night = _subperiod_hrv_results()

    assert by_night[
        "hrv_rmssd__5min__cycle_night_max__monthly_max"
    ] == direct["hrv_rmssd__5min__monthly_max"]


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


def test_sleep_stage_filtering_can_be_disabled():
    result = hrv.get_hrv_statistics(
        RawBBILoader(stage_type="awake"),
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution="nightly",
        aggregation="mean",
        sleep_stages_to_remove=None,
    )

    feature = "hrv_rmssd__5min__cycle_night_mean"
    assert result.loc[("user", pd.Timestamp("2024-01-02")), feature] == pytest.approx(
        200
    )


def test_sleep_stage_filtering_accepts_a_custom_list():
    result = hrv.get_hrv_statistics(
        RawBBILoader(stage_type="light"),
        statistic="hrv_rmssd",
        user_id="user",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution="nightly",
        aggregation="mean",
        sleep_stages_to_remove=["light"],
    )

    feature = "hrv_rmssd__5min__cycle_night_mean"
    assert np.isnan(result.loc[("user", pd.Timestamp("2024-01-02")), feature])


def test_loader_filtered_timestamp_gap_preserves_time_axis_and_successive_differences():
    result = hrv.get_hrv_statistics(
        LoaderFilteredGapLoader(),
        statistic=["hrv_rmssd", "hrv_pnn50", "hrv_lf_power"],
        user_id="user",
        start_date="2024-01-01 23:00",
        end_date="2024-01-01 23:05",
        resolution="5min",
        time_reference="local",
        min_coverage=0.7,
    )

    assert result.iloc[0]["hrv_rmssd"] == 0
    assert result.iloc[0]["hrv_pnn50"] == 0
    assert np.isfinite(result.iloc[0]["hrv_lf_power"])


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
