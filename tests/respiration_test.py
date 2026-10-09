import datetime

import pandas as pd
import pytest

import pywearable.cycle
import pywearable.loader
import pywearable.respiration
import pywearable.schema
from pywearable.loader.labfront import constants as labfront_constants
from pywearable.loader.labfront.loader import LabfrontLoader
from pywearable.loader.lifesnaps import LifeSnapsParquetLoader

class SampleRespirationLoader(pywearable.loader.BaseLoader):
    def __init__(self):
        times = pd.date_range("2024-01-01", periods=10, freq="1min")
        common = {
            "utc_time": times,
            "local_time": times,
            "timezone": pd.Timedelta(0),
        }
        self.respiration = pd.DataFrame(
            {**common, "respiratory_rate": range(10, 20)}
        )
        self.spo2 = pd.DataFrame(
            {**common, "spo2": [89, 89, 95, 96, 97, 98, 97, 96, 95, 94]}
        )

    def get_user_ids(self):
        return ["u1"]

    def load_respiration(self, user_id, start_date=None, end_date=None):
        return self.respiration.copy()

    def load_pulse_ox(self, user_id, start_date=None, end_date=None):
        return self.spo2.copy()

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "id": ["night-1", "night-2"],
                "calendar_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "type": ["main_sleep", "main_sleep"],
                "main_sleep": [True, True],
                "local_time": pd.to_datetime(
                    ["2024-01-01 00:00", "2024-01-02 00:00"]
                ),
                "local_end_time": pd.to_datetime(
                    ["2024-01-01 00:05", "2024-01-02 00:10"]
                ),
                "time_in_bed_seconds": [300, 600],
            }
        )

    def load_sleep_summary(self, *args, **kwargs):
        return pd.DataFrame()

    def load_sleep_stage(self, *args, **kwargs):
        return pd.DataFrame()

    def load_heart_rate(self, *args, **kwargs):
        return pd.DataFrame()


class SummaryRespirationLoader(SampleRespirationLoader):
    respiratory_rate_source = "provider_summary"
    spo2_source = "provider_summary"

    def load_respiration_summary(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "calendar_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "respiratory_rate_mean": [14.0, 16.0],
                "respiratory_rate_std": [1.0, 2.0],
            }
        )

    def load_spo2_summary(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "calendar_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "spo2_mean": [97.0, 95.0],
            }
        )


def test_raw_five_minute_statistics_coverage_and_threshold_duration():
    loader = SampleRespirationLoader()
    result = pywearable.respiration.get_five_minute_respiration_statistics(
        loader,
        statistic=[
            "respiratory_rate_mean",
            "respiratory_rate_count",
            "spo2_p20",
            "spo2_p30",
            "spo2_below_threshold_fraction",
            "spo2_below_threshold_duration_seconds",
        ],
        user_id="u1",
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:10",
        spo2_threshold=90,
    )
    first = result.loc[("u1", pd.Timestamp("2024-01-01 00:00"))]
    assert first["local_time"] == pd.Timestamp("2024-01-01 00:00")
    assert first["timezone"] == pd.Timedelta(0)
    assert first["respiratory_rate_mean"] == 12
    assert first["respiratory_rate_count"] == 5
    assert first["respiratory_rate_mean__coverage"] == 1
    assert first["spo2_below_threshold_fraction"] == pytest.approx(0.4)
    assert first["spo2_below_threshold_duration_seconds"] == 120
    assert first["spo2_p20"] == 89
    assert first["spo2_p30"] == pytest.approx(90.2)


def test_raw_hierarchy_aggregates_equal_child_windows():
    result = pywearable.respiration.get_respiration_statistics(
        SampleRespirationLoader(),
        statistic="respiratory_rate_mean",
        user_id="u1",
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:10",
        resolution=["5min", "daily"],
        aggregation="mean",
    )
    column = "respiratory_rate_mean__5min__daily_mean"
    assert result.loc[("u1", pd.Timestamp("2024-01-01")), column] == 14.5


def test_raw_five_minute_to_day_uses_local_calendar():
    loader = SampleRespirationLoader()
    local_times = pd.date_range("2024-01-02 00:00", periods=5, freq="1min")
    loader.respiration = pd.DataFrame(
        {
            "local_time": local_times,
            "utc_time": local_times - pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "respiratory_rate": [14] * 5,
        }
    )
    result = pywearable.respiration.get_respiration_statistics(
        loader,
        statistic="respiratory_rate_mean",
        user_id="u1",
        start_date="2024-01-02",
        end_date="2024-01-03",
        resolution=["5min", "daily"],
        aggregation="mean",
    )
    column = "respiratory_rate_mean__5min__daily_mean"
    assert result.loc[("u1", pd.Timestamp("2024-01-02")), column] == 14


def test_raw_five_minute_to_night_filters_ordinary_samples(monkeypatch):
    loader = SampleRespirationLoader()
    local_times = pd.date_range("2024-01-01 23:50", periods=20, freq="1min")
    samples = pd.DataFrame(
        {
            "local_time": local_times,
            "utc_time": local_times - pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "respiratory_rate": [100] * 5 + [10] * 5 + [20] * 5 + [100] * 5,
        }
    )
    monkeypatch.setattr(loader, "load_respiration", lambda **kwargs: samples.copy())
    monkeypatch.setattr(
        loader,
        "load_sleep_sessions",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "id": ["night-1"],
                "calendar_date": [pd.Timestamp("2024-01-01")],
                "type": ["main_sleep"],
                "main_sleep": [True],
                "local_time": [pd.Timestamp("2024-01-01 23:55")],
                "local_end_time": [pd.Timestamp("2024-01-02 00:05")],
                "time_in_bed_seconds": [600],
            }
        ),
    )
    result = pywearable.respiration.get_respiration_statistics(
        loader,
        statistic="respiratory_rate_mean",
        user_id="u1",
        start_date="2024-01-01",
        end_date="2024-01-02",
        resolution=["5min", pywearable.cycle.CYCLE_NIGHT],
        aggregation="mean",
    )
    column = "respiratory_rate_mean__5min__cycle_night_mean"
    assert result.loc[("u1", pd.Timestamp("2024-01-01")), column] == 15


def test_raw_night_and_awake_shortcuts_use_the_intended_cycles():
    loader = SampleRespirationLoader()
    nightly = pywearable.respiration.get_night_respiration_statistics(
        loader,
        statistic="respiratory_rate_mean",
        user_id="u1",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    awake = pywearable.respiration.get_awake_respiration_statistics(
        loader,
        statistic="respiratory_rate_mean",
        user_id="u1",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    assert nightly.loc[
        ("u1", pd.Timestamp("2024-01-01")), "respiratory_rate_mean"
    ] == 12
    assert awake.loc[
        ("u1", pd.Timestamp("2024-01-01")), "respiratory_rate_mean"
    ] == 17


def test_combined_raw_night_features_share_cycle_lookup(monkeypatch):
    original = pywearable.cycle.get_intervals
    calls = []

    def count_calls(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pywearable.cycle, "get_intervals", count_calls)
    result = pywearable.respiration.get_night_respiration_statistics(
        SampleRespirationLoader(),
        statistic=["respiratory_rate_mean", "spo2_mean"],
        user_id="u1",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    assert len(calls) == 1
    assert result.loc[("u1", pd.Timestamp("2024-01-01")), "spo2_mean"] == 93.2


def test_provider_summaries_remain_at_native_or_larger_grain():
    loader = SummaryRespirationLoader()
    nightly = pywearable.respiration.get_night_respiration_statistics(
        loader,
        statistic=["respiratory_rate_mean", "respiratory_rate_std", "spo2_mean"],
        user_id="u1",
    )
    assert nightly.loc[("u1", pd.Timestamp("2024-01-01")), "spo2_mean"] == 97
    assert nightly.loc[
        ("u1", pd.Timestamp("2024-01-02")), "respiratory_rate_std"
    ] == 2

    with pytest.raises(ValueError, match="native nightly grain"):
        pywearable.respiration.get_five_minute_respiration_statistics(
            loader, statistic="respiratory_rate_mean", user_id="u1"
        )
    with pytest.raises(ValueError, match="native nightly grain"):
        pywearable.respiration.get_respiration_statistics(
            loader,
            statistic="respiratory_rate_mean",
            user_id="u1",
            resolution=["5min", pywearable.cycle.CYCLE_NIGHT],
            aggregation="mean",
        )
    with pytest.raises(ValueError, match="native nightly grain"):
        pywearable.respiration.get_respiration_statistics(
            loader,
            statistic="respiratory_rate_mean",
            user_id="u1",
            resolution="daily",
        )

    monthly = pywearable.respiration.get_respiration_statistics(
        loader,
        statistic="respiratory_rate_mean",
        user_id="u1",
        resolution="monthly",
        aggregation="mean",
    )
    column = "respiratory_rate_mean__cycle_night__monthly_mean"
    assert monthly.iloc[0][column] == 15


def test_provider_month_uses_night_label_not_previous_evening(monkeypatch):
    loader = SummaryRespirationLoader()
    monkeypatch.setattr(
        loader,
        "load_respiration_summary",
        lambda **kwargs: pd.DataFrame(
            {
                "calendar_date": [pd.Timestamp("2024-02-01")],
                "local_time": [pd.Timestamp("2024-01-31 23:00")],
                "local_end_time": [pd.Timestamp("2024-02-01 07:00")],
                "respiratory_rate_mean": [14.0],
            }
        ),
    )
    monthly = pywearable.respiration.get_respiration_statistics(
        loader,
        statistic="respiratory_rate_mean",
        user_id="u1",
        resolution="monthly",
        aggregation="mean",
    )
    assert monthly.index.get_level_values("calendar_date").tolist() == [
        pd.Timestamp("2024-02-01")
    ]


def test_provider_primary_coverage_is_not_invented():
    with pytest.warns(UserWarning, match="coverage could not be determined"):
        result = pywearable.respiration.get_night_respiration_statistics(
            SummaryRespirationLoader(),
            statistic="spo2_mean",
            user_id="u1",
            min_coverage=0.5,
        )
    assert result["spo2_mean"].isna().all()
    assert result["spo2_mean__coverage"].isna().all()


@pytest.mark.parametrize("family", ["respiratory_rate", "spo2"])
def test_provider_summary_hierarchy_counts_missing_nights(monkeypatch, family):
    loader = SummaryRespirationLoader()
    statistic = f"{family}_mean"
    method = "load_respiration_summary" if family == "respiratory_rate" else "load_spo2_summary"
    summary = pd.DataFrame({
        "calendar_date": pd.to_datetime(["2024-01-01", "2024-01-07"]),
        statistic: [15.0, 16.0] if family == "respiratory_rate" else [96.0, 97.0],
        f"{family}_coverage": [1.0, 1.0],
    })
    monkeypatch.setattr(loader, method, lambda **kwargs: summary.copy())
    options = dict(loader=loader, statistic=statistic, user_id="u1",
                   start_date="2024-01-01", end_date="2024-01-15")
    nightly = pywearable.respiration.get_respiration_statistics(**options)
    assert len(nightly) == 14
    missing = nightly[statistic].isna()
    assert missing.sum() == 12
    assert nightly.loc[missing, f"{statistic}__coverage"].eq(0).all()
    weekly = pywearable.respiration.get_respiration_statistics(
        **options, resolution=["nightly", "weekly"], aggregation="mean",
        min_coverage=[None, 0.7],
    )
    feature = f"{statistic}__cycle_night__weekly_mean"
    assert len(weekly) == 2
    assert weekly[feature].isna().all()
    assert weekly[f"{feature}__coverage"].tolist() == pytest.approx([2 / 7, 0])


def test_empty_provider_summary_preserves_requested_nights(monkeypatch):
    loader = SummaryRespirationLoader()
    monkeypatch.setattr(loader, "load_respiration_summary", lambda **kwargs: pd.DataFrame())
    result = pywearable.respiration.get_respiration_statistics(
        loader, statistic="respiratory_rate_mean", start_date="2024-01-01",
        end_date="2024-01-08", resolution=["nightly", "weekly"], aggregation="mean",
    )
    feature = "respiratory_rate_mean__cycle_night__weekly_mean"
    assert len(result) == 1
    assert pd.isna(result.iloc[0][feature])
    assert result.iloc[0][f"{feature}__coverage"] == 0


def test_schema_requires_canonical_columns_and_drops_invalid_values():
    frame = pd.DataFrame(
        {
            "local_time": pd.to_datetime(
                ["2024-01-01", "2024-01-01 00:01"], format="mixed"
            ),
            "respiratory_rate": [0, 14],
        }
    )
    normalized = pywearable.schema.prepare_respiration_samples(frame)
    assert normalized["respiratory_rate"].tolist() == [14]
    assert normalized["local_time"].tolist() == [pd.Timestamp("2024-01-01 00:01")]
    with pytest.raises(ValueError, match="respiratory_rate"):
        pywearable.schema.prepare_respiration_samples(
            pd.DataFrame(
                {
                    "isoDate": [pd.Timestamp("2024-01-01")],
                    "breathsPerMinute": [14],
                }
            )
        )


def test_summary_schema_preserves_grain_and_rejects_invalid_values():
    respiratory = pywearable.schema.prepare_respiration_summary(
        pd.DataFrame(
            {
                "calendar_date": ["2024-01-01", "bad"],
                "respiratory_rate_mean": [-2, 14],
                "respiratory_rate_std": [-1, 1],
            }
        )
    )
    assert len(respiratory) == 1
    assert pd.isna(respiratory.loc[0, "respiratory_rate_mean"])
    assert pd.isna(respiratory.loc[0, "respiratory_rate_std"])

    spo2 = pywearable.schema.prepare_spo2_summary(
        pd.DataFrame(
            {
                "calendar_date": ["2024-01-01"],
                "spo2_mean": [101],
                "spo2_lower_bound": [0],
                "spo2_upper_bound": [99],
            }
        )
    )
    assert pd.isna(spo2.loc[0, "spo2_mean"])
    assert pd.isna(spo2.loc[0, "spo2_lower_bound"])
    assert spo2.loc[0, "spo2_upper_bound"] == 99


def test_lifesnaps_summaries_use_local_sleep_end_and_expose_spo2(
    tmp_path, monkeypatch
):
    user_dir = tmp_path / "u1"
    user_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 07:00",
                    "2024-01-02 07:00",
                    "2024-01-02 02:00",
                ]
            ),
            "full_sleep_breathing_rate": [14.2, 14.2, 99.0],
            "full_sleep_standard_deviation": [1.1, 1.1, 9.0],
        }
    ).to_parquet(user_dir / "respiratory_rate_summary.parquet")
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "average_value": [96.5, 96.5],
            "lower_bound": [94.0, 94.0],
            "upper_bound": [98.0, 98.0],
        }
    ).to_parquet(user_dir / "daily_spo2.parquet")
    loader = LifeSnapsParquetLoader(tmp_path)
    monkeypatch.setattr(
        loader,
        "load_sleep_summary",
        lambda **kwargs: pd.DataFrame(
            {
                "sleep_summary_id": ["night-1"],
                "calendar_date": pd.to_datetime(["2024-01-01"]),
                "local_time": pd.to_datetime(["2024-01-01 23:00"]),
                "local_end_time": pd.to_datetime(["2024-01-02 07:00"]),
            }
        ),
    )

    respiration = loader.load_respiration_summary("u1")
    spo2 = loader.load_spo2_summary("u1")
    assert respiration.loc[0, "calendar_date"] == pd.Timestamp("2024-01-01")
    assert respiration.loc[0, "respiratory_rate_mean"] == 14.2
    assert respiration.loc[0, "respiratory_rate_std"] == 1.1
    assert respiration.loc[0, "sleep_summary_id"] == "night-1"
    assert len(respiration) == 1
    assert spo2.loc[0, "spo2_mean"] == 96.5
    assert spo2.loc[0, "spo2_lower_bound"] == 94.0
    assert spo2.loc[0, "spo2_upper_bound"] == 98.0


def test_lifesnaps_respiration_without_matched_sleep_is_not_a_night(tmp_path):
    user_dir = tmp_path / "u1"
    user_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02 07:00")],
            "full_sleep_breathing_rate": [14.2],
        }
    ).to_parquet(user_dir / "respiratory_rate_summary.parquet")
    loader = LifeSnapsParquetLoader(tmp_path)
    assert loader.load_respiration_summary("u1").empty


def test_labfront_connect_samples_do_not_load_sleep_streams(monkeypatch):
    loader = LabfrontLoader.__new__(LabfrontLoader)
    pulse_ox = pd.DataFrame(
        {
            "isoDate": pd.to_datetime(["2024-01-01 12:00"]),
            "unixTimestampInMs": [1_704_110_400_000],
            "timezoneOffsetInMs": [0],
            "spo2": [97],
        }
    )
    respiration = pulse_ox.drop(columns="spo2").assign(breathsPerMinute=14)
    requested_metrics = []

    def load_data(*args, metric, **kwargs):
        requested_metrics.append(metric)
        if metric == labfront_constants.GARMIN_CONNECT_DAILY_PULSE_OX_FOLDER:
            return pulse_ox.copy()
        if metric == labfront_constants.GARMIN_CONNECT_RESPIRATION_FOLDER:
            return respiration.copy()
        raise AssertionError(f"unexpected stream: {metric}")

    monkeypatch.setattr(loader, "get_data_from_datetime", load_data)
    pulse_result = loader.load_garmin_connect_pulse_ox("u1")
    respiration_result = loader.load_garmin_connect_respiration("u1")

    assert pulse_result["spo2"].tolist() == [97]
    assert respiration_result["breathsPerMinute"].tolist() == [14]
    assert requested_metrics == [
        labfront_constants.GARMIN_CONNECT_DAILY_PULSE_OX_FOLDER,
        labfront_constants.GARMIN_CONNECT_RESPIRATION_FOLDER,
    ]


def test_labfront_loader_translates_provider_respiration_columns(monkeypatch):
    loader = LabfrontLoader.__new__(LabfrontLoader)
    provider = pd.DataFrame(
        {
            "isoDate": [pd.Timestamp("2024-01-01 12:00")],
            "unixTimestampInMs": [1_704_110_400_000],
            "timezoneOffsetInMs": [0],
            "breathsPerMinute": [14],
        }
    )
    monkeypatch.setattr(
        loader,
        "load_garmin_connect_respiration",
        lambda **kwargs: provider.copy(),
    )
    result = loader.load_respiration("u1")
    assert result.loc[0, "respiratory_rate"] == 14
    assert result.loc[0, "local_time"] == pd.Timestamp("2024-01-01 12:00")
    assert result.loc[0, "utc_time"] == pd.Timestamp("2024-01-01 12:00")
