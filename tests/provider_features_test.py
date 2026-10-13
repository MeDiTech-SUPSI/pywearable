import numpy as np
import pandas as pd
import pytest

from pywearable import provider_features, schema
from pywearable.loader import LabfrontLoader
from pywearable.loader.base import BaseLoader
from pywearable.loader.labfront import constants as labfront_constants
from pywearable.loader.lifesnaps import LifeSnapsParquetLoader
from pywearable.loader.tesserae.tesserae_parquet import TesseraeParquetLoader


class ProviderFeatureLoader(BaseLoader):
    provider_features = frozenset(
        {
            provider_features.GARMIN_STRESS_LEVEL,
            provider_features.GARMIN_BODY_BATTERY,
            provider_features.FITBIT_STRESS_MANAGEMENT_SCORE,
        }
    )
    provider_feature_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)

    def __init__(self):
        times = pd.date_range("2024-01-01", periods=10, freq="1min")
        self.samples = {
            provider_features.GARMIN_STRESS_LEVEL: pd.DataFrame(
                {
                    schema.UTC_TIME_COL: times,
                    schema.LOCAL_TIME_COL: times,
                    schema.TIMEZONE_COL: pd.Timedelta(0),
                    provider_features.GARMIN_STRESS_LEVEL: [
                        10,
                        20,
                        np.nan,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                    ],
                }
            ),
            provider_features.GARMIN_BODY_BATTERY: pd.DataFrame(
                {
                    schema.UTC_TIME_COL: times,
                    schema.LOCAL_TIME_COL: times,
                    schema.TIMEZONE_COL: pd.Timedelta(0),
                    provider_features.GARMIN_BODY_BATTERY: np.arange(10, 20),
                }
            ),
        }
        self.daily = pd.DataFrame(
            {
                schema.CALENDAR_DATE_COL: pd.to_datetime(
                    ["2024-01-01", "2024-01-03"]
                ),
                provider_features.FITBIT_STRESS_MANAGEMENT_SCORE: [70.0, 80.0],
            }
        )

    def get_user_ids(self):
        return ["person"]

    def load_heart_rate(self, *args, **kwargs):
        return pd.DataFrame()

    def load_sleep_summary(self, *args, **kwargs):
        return pd.DataFrame()

    def load_sleep_stage(self, *args, **kwargs):
        return pd.DataFrame()

    def load_provider_feature(
        self, feature, user_id, start_date=None, end_date=None, **kwargs
    ):
        if feature in self.samples:
            return self.samples[feature].copy()
        return self.daily.copy()


class CustomProviderFeatureLoader(ProviderFeatureLoader):
    provider_features = frozenset(
        {"example_sleep_health_score", "example_daily_strain_score"}
    )
    provider_feature_definitions = {
        "example_sleep_health_score": {"grain": "daily", "numeric": True},
        "example_daily_strain_score": {"grain": "daily", "numeric": True},
    }

    def __init__(self):
        super().__init__()
        self.daily = pd.DataFrame(
            {
                schema.CALENDAR_DATE_COL: pd.to_datetime(
                    ["2024-01-01", "2024-01-02"]
                ),
                "example_sleep_health_score": [82.0, 76.0],
                "example_daily_strain_score": [61.0, 73.0],
            }
        )

    def load_provider_feature(
        self, feature, user_id, start_date=None, end_date=None, **kwargs
    ):
        return self.daily[[schema.CALENDAR_DATE_COL, feature]].copy()


def test_loader_declared_daily_provider_features_use_shared_pipeline():
    loader = CustomProviderFeatureLoader()

    assert provider_features.get_available_provider_features(loader) == [
        "example_daily_strain_score",
        "example_sleep_health_score",
    ]
    result = provider_features.get_provider_features(
        loader,
        feature=["example_sleep_health_score", "example_daily_strain_score"],
        user_id="person",
        start_date="2024-01-01",
        end_date="2024-01-03",
    )

    assert result["example_sleep_health_score"].tolist() == [82.0, 76.0]
    assert result["example_daily_strain_score"].tolist() == [61.0, 73.0]
    assert result["example_sleep_health_score__coverage"].eq(1.0).all()


def test_loader_declared_provider_feature_definitions_are_validated():
    loader = CustomProviderFeatureLoader()
    loader.provider_feature_definitions = {
        "example_sleep_health_score": {"grain": "nightly"}
    }

    with pytest.raises(ValueError, match="grain 'samples' or 'daily'"):
        provider_features.get_available_provider_features(loader)


@pytest.mark.parametrize(
    "source,folder",
    [
        (
            labfront_constants.DATA_SOURCE_GARMIN_SDK,
            labfront_constants.GARMIN_DEVICE_STRESS_FOLDER,
        ),
        (
            labfront_constants.DATA_SOURCE_GARMIN_HEALTH_API,
            labfront_constants.GARMIN_CONNECT_STRESS_FOLDER,
        ),
    ],
)
def test_labfront_stress_routes_to_the_selected_stress_folder(
    monkeypatch, source, folder,
):
    loader = LabfrontLoader.__new__(LabfrontLoader)
    requested = {}

    def load(**kwargs):
        requested.update(kwargs)
        return pd.DataFrame({"stressLevel": [25]})

    monkeypatch.setattr(loader, "get_data_from_datetime", load)
    result = loader.load_stress("person", "2024-01-01", "2024-01-02", source=source)
    assert requested["metric"] == folder
    assert requested["start_date"] == "2024-01-01"
    assert requested["end_date"] == "2024-01-02"
    assert result.stressLevel.iloc[0] == 25


def test_provider_sample_duplicate_conflicts_are_not_selected_arbitrarily():
    timestamp = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame(
        {
            schema.UTC_TIME_COL: [timestamp, timestamp],
            schema.LOCAL_TIME_COL: [timestamp, timestamp],
            provider_features.GARMIN_STRESS_LEVEL: [20, 40],
        }
    )

    result = schema.prepare_provider_feature_samples(
        frame, provider_features.GARMIN_STRESS_LEVEL
    )

    assert len(result) == 1
    assert pd.isna(result.loc[0, provider_features.GARMIN_STRESS_LEVEL])


def test_sample_features_support_statistics_coverage_and_multiple_streams():
    loader = ProviderFeatureLoader()

    result = provider_features.get_provider_features(
        loader,
        feature=[
            provider_features.GARMIN_STRESS_LEVEL,
            provider_features.GARMIN_BODY_BATTERY,
        ],
        statistic=["mean", "max"],
        user_id="person",
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:10",
        resolution="5min",
        min_coverage=0.8,
    )

    first = result.loc[("person", pd.Timestamp("2024-01-01 00:00"))]
    assert first[f"{provider_features.GARMIN_STRESS_LEVEL}_mean"] == 25
    assert first[f"{provider_features.GARMIN_STRESS_LEVEL}_max"] == 40
    assert first[f"{provider_features.GARMIN_BODY_BATTERY}_mean"] == 12
    assert first[f"{provider_features.GARMIN_STRESS_LEVEL}_mean__coverage"] == 0.8
    assert first[f"{provider_features.GARMIN_BODY_BATTERY}_mean__coverage"] == 1


def test_single_sample_feature_returns_coverage_without_a_threshold():
    result = provider_features.get_provider_features(
        ProviderFeatureLoader(),
        feature=provider_features.GARMIN_STRESS_LEVEL,
        statistic="mean",
        user_id="person",
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:10",
        resolution="5min",
    )

    feature = f"{provider_features.GARMIN_STRESS_LEVEL}_mean"
    assert list(result) == [schema.LOCAL_TIME_COL, schema.TIMEZONE_COL,
                            feature, f"{feature}__coverage"]
    assert result.iloc[0][f"{feature}__coverage"] == 0.8


@pytest.mark.parametrize("time_reference", ["utc", "local"])
@pytest.mark.parametrize("resolution", ["5min", ["5min", "10min"]])
def test_fixed_provider_features_keep_both_clocks_and_offsets(time_reference, resolution):
    loader = ProviderFeatureLoader()
    for samples in loader.samples.values():
        samples[schema.TIMEZONE_COL] = pd.to_timedelta([1] * 5 + [2] * 5, unit="h")
        samples[schema.LOCAL_TIME_COL] = samples[schema.UTC_TIME_COL] + samples[schema.TIMEZONE_COL]
    result = provider_features.get_provider_features(
        loader, feature=[provider_features.GARMIN_STRESS_LEVEL, provider_features.GARMIN_BODY_BATTERY],
        statistic="mean", resolution=resolution, time_reference=time_reference,
        **({"aggregation": "mean"} if isinstance(resolution, list) else {}),
    )
    assert not result.empty
    selected = schema.UTC_TIME_COL if time_reference == "utc" else schema.LOCAL_TIME_COL
    alternative = schema.LOCAL_TIME_COL if time_reference == "utc" else schema.UTC_TIME_COL
    assert result.index.names == [schema.USER_COL, selected]
    observed = result[result[schema.TIMEZONE_COL].notna()]
    times = pd.Series(observed.index.get_level_values(selected), index=observed.index)
    expected = (times + observed[schema.TIMEZONE_COL] if selected == schema.UTC_TIME_COL
                else times - observed[schema.TIMEZONE_COL])
    pd.testing.assert_series_equal(observed[alternative], expected, check_names=False)
    if time_reference == "local":
        assert result[schema.TIMEZONE_COL].isna().any()
    if resolution == "5min":
        assert set(observed[schema.TIMEZONE_COL]) == {pd.Timedelta(hours=1), pd.Timedelta(hours=2)}


@pytest.mark.parametrize("resolution", ["5min", ["5min", "10min"]])
def test_empty_fixed_provider_features_keep_time_metadata(resolution):
    loader = ProviderFeatureLoader()
    loader.samples[provider_features.GARMIN_STRESS_LEVEL] = loader.samples[provider_features.GARMIN_STRESS_LEVEL].iloc[0:0]
    result = provider_features.get_provider_features(
        loader, feature=provider_features.GARMIN_STRESS_LEVEL, statistic="mean",
        resolution=resolution, **({"aggregation": "mean"} if isinstance(resolution, list) else {}),
    )
    assert result.empty
    assert result.index.names == [schema.USER_COL, schema.UTC_TIME_COL]
    assert schema.LOCAL_TIME_COL in result
    assert pd.api.types.is_timedelta64_dtype(result[schema.TIMEZONE_COL])


def test_provider_features_do_not_invent_missing_source_offsets():
    loader = ProviderFeatureLoader()
    samples = loader.samples[provider_features.GARMIN_STRESS_LEVEL]
    samples[schema.LOCAL_TIME_COL] = pd.NaT
    samples[schema.TIMEZONE_COL] = pd.Series(pd.NaT, index=samples.index, dtype="timedelta64[ns]")
    result = provider_features.get_provider_features(
        loader, feature=provider_features.GARMIN_STRESS_LEVEL, statistic="mean", resolution="5min",
    )
    assert result.garmin_stress_level_mean.notna().all()
    assert result[schema.LOCAL_TIME_COL].isna().all()
    assert result[schema.TIMEZONE_COL].isna().all()


def test_sample_hierarchy_uses_equal_primary_window_weights():
    loader = ProviderFeatureLoader()

    result = provider_features.get_provider_features(
        loader,
        feature=provider_features.GARMIN_STRESS_LEVEL,
        statistic="mean",
        user_id="person",
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:10",
        resolution=["5min", "daily"],
        aggregation="mean",
    )

    column = "garmin_stress_level_mean__5min__daily_mean"
    assert result.loc[("person", pd.Timestamp("2024-01-01")), column] == 47.5


def test_native_daily_features_are_not_expanded_and_missing_dates_count():
    loader = ProviderFeatureLoader()

    with pytest.raises(ValueError, match="native daily"):
        provider_features.get_provider_features(
            loader,
            feature=provider_features.FITBIT_STRESS_MANAGEMENT_SCORE,
            resolution="5min",
        )

    result = provider_features.get_provider_features(
        loader,
        feature=provider_features.FITBIT_STRESS_MANAGEMENT_SCORE,
        user_id="person",
        start_date="2024-01-01",
        end_date="2024-01-08",
        resolution="weekly",
        aggregation="mean",
        min_coverage=[None, 0.5],
    )
    column = "fitbit_stress_management_score__daily__weekly_mean"
    row = result.loc[("person", pd.Timestamp("2024-01-01"))]
    assert pd.isna(row[column])
    assert row[f"{column}__coverage"] == pytest.approx(2 / 7)


def test_mixed_native_grains_are_rejected():
    loader = ProviderFeatureLoader()
    with pytest.raises(ValueError, match="same native grain"):
        provider_features.get_provider_features(
            loader,
            feature=[
                provider_features.GARMIN_STRESS_LEVEL,
                provider_features.FITBIT_STRESS_MANAGEMENT_SCORE,
            ],
        )


def test_body_battery_sleep_features_use_cycle_boundaries(monkeypatch):
    loader = ProviderFeatureLoader()
    loader.provider_features = loader.provider_features | {
        provider_features.GARMIN_BODY_BATTERY_BEDTIME,
        provider_features.GARMIN_BODY_BATTERY_GETUP,
        provider_features.GARMIN_BODY_BATTERY_SLEEP_GAIN,
        provider_features.GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION,
    }
    times = pd.to_datetime(
        ["2024-01-01 21:58", "2024-01-01 22:00", "2024-01-02 06:00"]
    )
    loader.samples[provider_features.GARMIN_BODY_BATTERY] = pd.DataFrame(
        {
            schema.UTC_TIME_COL: times,
            schema.LOCAL_TIME_COL: times,
            schema.TIMEZONE_COL: pd.Timedelta(0),
            provider_features.GARMIN_BODY_BATTERY: [20.0, 22.0, 70.0],
        }
    )
    intervals = pd.DataFrame(
        {
            schema.USER_COL: ["person"],
            schema.CALENDAR_DATE_COL: [pd.Timestamp("2024-01-01")],
            "start_time": [pd.Timestamp("2024-01-01 22:00")],
            "end_time": [pd.Timestamp("2024-01-02 06:00")],
            "interval_valid": [True],
        }
    ).set_index([schema.USER_COL, schema.CALENDAR_DATE_COL])
    monkeypatch.setattr(
        provider_features.cycle, "get_intervals", lambda *args, **kwargs: intervals
    )

    result = provider_features.get_provider_features(
        loader,
        feature=[
            provider_features.GARMIN_BODY_BATTERY_BEDTIME,
            provider_features.GARMIN_BODY_BATTERY_GETUP,
            provider_features.GARMIN_BODY_BATTERY_SLEEP_GAIN,
            provider_features.GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION,
        ],
        user_id="person",
        start_date="2024-01-01",
        end_date="2024-01-02",
        resolution="cycle_night",
    )

    row = result.loc[("person", pd.Timestamp("2024-01-01"))]
    assert row[provider_features.GARMIN_BODY_BATTERY_BEDTIME] == 22
    assert row[provider_features.GARMIN_BODY_BATTERY_GETUP] == 70
    assert row[provider_features.GARMIN_BODY_BATTERY_SLEEP_GAIN] == 48
    assert row[
        provider_features.GARMIN_BODY_BATTERY_SLEEP_RECOVERY_FRACTION
    ] == pytest.approx(48 / 78)


@pytest.mark.parametrize("include_offset", [True, False])
def test_labfront_translates_stress_sentinels_without_losing_body_battery(
    monkeypatch, include_offset,
):
    loader = LabfrontLoader.__new__(LabfrontLoader)
    raw = pd.DataFrame(
        {
            "isoDate": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:03", "2024-01-01 00:06"]
            ),
            "unixTimestampInMs": [1704063600000, 1704063780000, 1704063960000],
            "timezoneOffsetInMs": [3600000, 3600000, 3600000],
            "stressLevel": [-1, -2, 25],
            "bodyBattery": [50, 49, 48],
        }
    )
    if not include_offset:
        raw = raw.drop(columns="timezoneOffsetInMs")
    monkeypatch.setattr(loader, "load_stress", lambda *args, **kwargs: raw.copy())

    stress = loader.load_provider_feature(
        provider_features.GARMIN_STRESS_LEVEL, "person"
    )
    battery = loader.load_provider_feature(
        provider_features.GARMIN_BODY_BATTERY, "person"
    )

    assert stress[provider_features.GARMIN_STRESS_LEVEL].isna().iloc[0]
    assert stress[provider_features.GARMIN_STRESS_LEVEL].isna().iloc[1]
    assert stress[provider_features.GARMIN_STRESS_LEVEL].iloc[2] == 25
    assert battery[provider_features.GARMIN_BODY_BATTERY].tolist() == [50, 49, 48]
    assert stress[schema.UTC_TIME_COL].iloc[0] == pd.Timestamp("2023-12-31 23:00")
    assert stress[schema.TIMEZONE_COL].eq(pd.Timedelta(hours=1)).all()


def test_lifesnaps_drops_duplicates_and_requires_completed_fitbit_scores(tmp_path):
    user = "person"
    folder = tmp_path / user
    folder.mkdir()
    rows = pd.DataFrame(
        {
            "DATE": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "UPDATED_AT": ["a", "a", "b", "c", "d"],
            "STRESS_SCORE": [80, 80, 0, 0, 0],
            "SLEEP_POINTS": [25, 25, 0, 0, 0],
            "MAX_SLEEP_POINTS": [30, 30, 30, 30, 30],
            "RESPONSIVENESS_POINTS": [25, 25, 0, 0, 0],
            "MAX_RESPONSIVENESS_POINTS": [30, 30, 30, 30, 30],
            "EXERTION_POINTS": [30, 30, 0, 0, 0],
            "MAX_EXERTION_POINTS": [40, 40, 40, 40, 40],
            "STATUS": ["READY", "READY", "NO_DATA", "IN_PROGRESS", "READY"],
            "CALCULATION_FAILED": [False, False, False, False, True],
        }
    )
    rows.to_parquet(folder / "stress_score.parquet")
    loader = LifeSnapsParquetLoader(tmp_path)

    result = loader.load_provider_feature(
        provider_features.FITBIT_STRESS_MANAGEMENT_SCORE, user
    )

    assert len(result) == 4
    assert result[provider_features.FITBIT_STRESS_MANAGEMENT_SCORE].iloc[0] == 80
    assert (
        result[provider_features.FITBIT_STRESS_MANAGEMENT_SCORE]
        .iloc[1:]
        .isna()
        .all()
    )


def test_tesserae_provider_feature_supports_unbounded_loading(tmp_path):
    user = "person"
    folder = tmp_path / user / "GarminHealthAPI"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                ["2024-01-01 00:00Z", "2024-01-01 00:03Z"]
            ),
            "tz_source": ["export", "export"],
            "local_time": pd.to_datetime(
                ["2024-01-01 01:00", "2024-01-01 01:03"]
            ),
            "hr": [-2, 35],
        }
    ).to_parquet(folder / "Stress.parquet")
    loader = TesseraeParquetLoader(tmp_path)

    result = loader.load_provider_feature(
        provider_features.GARMIN_STRESS_LEVEL, user
    )

    assert result[provider_features.GARMIN_STRESS_LEVEL].isna().iloc[0]
    assert result[provider_features.GARMIN_STRESS_LEVEL].iloc[1] == 35
