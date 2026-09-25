import numpy as np
import pandas as pd
import pytest

from pywearable import sleep
from pywearable.loader import LifeSnapsParquetLoader
from pywearable.loader.tesserae import TesseraeCSVLoader, TesseraeParquetLoader


def test_tesserae_csv_sleep_uses_optional_bounds_and_true_utc_milliseconds(
    tmp_path, monkeypatch
):
    sleep_folder = tmp_path / "person-1" / "GarminHealthAPI"
    sleep_folder.mkdir(parents=True)
    sleep_file = sleep_folder / "Sleep.csv"
    sleep_file.touch()
    raw = pd.DataFrame(
        {
            "ParticipantID": ["person-1"],
            "local_time": pd.to_datetime(["2024-01-01 23:00"]),
            "Timestamp": ["2024-01-01T23:00:00+01:00"],
            "sleep_levels_map": ["{}"],
            "awake_seconds": [600],
            "rem_sleep_seconds": [600],
            "deep_sleep_seconds": [1200],
            "unmeasurable_sleep_seconds": [0],
            "light_sleep_seconds": [1800],
            "duration_seconds": [3600],
            "validation": ["ENHANCED_FINAL"],
            "tz_source": ["test"],
        }
    )
    monkeypatch.setattr(
        "pywearable.loader.tesserae.tesserae_csv.glob.glob",
        lambda _: [str(sleep_file)],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: raw.copy())

    loader = TesseraeCSVLoader(tmp_path)
    result = loader.load_sleep_summary("person-1")

    assert result.loc[0, "utc_time"] == pd.Timestamp("2024-01-01 22:00")
    assert result.loc[0, "timezone_offset_in_ms"] == 3_600_000
    assert result.loc[0, "unix_timestamp_in_ms"] == 1_704_146_400_000
    assert loader._convert_timezone_offset("2024-01-01T23:00:00+0100") == 3_600_000


def test_tesserae_csv_empty_stage_supports_omitted_bounds(tmp_path):
    (tmp_path / "person-1").mkdir()

    result = TesseraeCSVLoader(tmp_path).load_sleep_stage("person-1")

    assert {"sleep_summary_id", "local_time", "duration_in_ms", "type"} <= set(
        result
    )


def test_tesserae_sleep_summary_coerces_null_numeric_values(tmp_path, monkeypatch):
    sleep_folder = tmp_path / "person-1" / "GarminHealthAPI"
    sleep_folder.mkdir(parents=True)
    (sleep_folder / "Sleep.parquet").touch()
    raw = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-01T22:00:00Z"], utc=True),
            "local_time": pd.to_datetime(["2024-01-01 23:00"]),
            "duration_seconds": ["3600"],
            "rem_sleep_seconds": ["NULL"],
            "deep_sleep_seconds": ["1200"],
            "awake_seconds": ["0"],
            "light_sleep_seconds": ["1800"],
            "unmeasurable_sleep_seconds": ["600"],
            "validation": ["AUTO_FINAL"],
            "sleep_levels_map": ["{}"],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())

    result = TesseraeParquetLoader(tmp_path).load_sleep_summary("person-1")

    assert result.loc[0, "duration_in_ms"] == 3_600_000
    assert result.loc[0, "rem_sleep_duration_in_ms"] == 0
    assert result.loc[0, "unmeasurable_sleep_duration_in_ms"] == 600_000


def test_tesserae_normalizes_legacy_duration_that_includes_awake_time(
    tmp_path, monkeypatch
):
    sleep_folder = tmp_path / "person-1" / "GarminHealthAPI"
    sleep_folder.mkdir(parents=True)
    (sleep_folder / "Sleep.parquet").touch()
    raw = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-01T22:00:00Z"], utc=True),
            "local_time": pd.to_datetime(["2024-01-01 23:00"]),
            "duration_seconds": [3600],
            "rem_sleep_seconds": [0],
            "deep_sleep_seconds": [1200],
            "awake_seconds": [600],
            "light_sleep_seconds": [1800],
            "unmeasurable_sleep_seconds": [0],
            "validation": ["AUTO_TENTATIVE"],
            "sleep_levels_map": ["{}"],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())

    result = TesseraeParquetLoader(tmp_path).load_sleep_summary("person-1")

    assert result.loc[0, "duration_in_ms"] == 3_000_000
    assert result.loc[0, "awake_duration_in_ms"] == 600_000
    assert result.loc[0, "local_end_time"] == pd.Timestamp("2024-01-02 00:00")


def test_tesserae_sleep_uses_calendar_day_and_drops_empty_records(
    tmp_path, monkeypatch
):
    sleep_folder = tmp_path / "person-1" / "GarminHealthAPI"
    sleep_folder.mkdir(parents=True)
    (sleep_folder / "Sleep.parquet").touch()
    raw = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                ["2024-01-01T22:00:00Z", "2024-01-02T22:00:00Z"], utc=True
            ),
            "local_time": pd.to_datetime(
                ["2024-01-01 23:00", "2024-01-02 23:00"]
            ),
            "duration_seconds": [3600, 0],
            "rem_sleep_seconds": [600, 0],
            "deep_sleep_seconds": [1200, 0],
            "awake_seconds": [600, 0],
            "light_sleep_seconds": [1800, 0],
            "unmeasurable_sleep_seconds": [0, 0],
            "validation": ["AUTO_FINAL", "ENHANCED_TENTATIVE"],
            "sleep_levels_map": ["{}", "{}"],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())

    result = TesseraeParquetLoader(tmp_path).load_sleep_summary(
        "person-1", start_date="2024-01-02", end_date="2024-01-03"
    )

    assert len(result) == 1
    assert result.loc[0, "local_time"] == pd.Timestamp("2024-01-01 23:00")
    assert result.loc[0, "calendar_date"] == pd.Timestamp("2024-01-02")


def test_tesserae_empty_sleep_stage_uses_shared_contract(tmp_path):
    (tmp_path / "person-1").mkdir()

    result = TesseraeParquetLoader(tmp_path).load_sleep_stage("person-1")

    assert {
        "sleep_summary_id",
        "utc_time",
        "local_time",
        "duration_in_ms",
        "type",
    } <= set(result.columns)


def test_lifesnaps_sleep_uses_main_sessions_and_normalizes_fitbit_stages(
    tmp_path, monkeypatch
):
    user_folder = tmp_path / "person-1"
    user_folder.mkdir()
    (user_folder / "sleep.parquet").touch()
    (user_folder / "sleep_stages.parquet").touch()
    (user_folder / "sleep_short_stages.parquet").touch()

    summaries = pd.DataFrame(
        [
            {
                "logId": "staged-night",
                "dateOfSleep": "2024-01-02",
                "startTime": "2024-01-01 22:00",
                "endTime": "2024-01-01 23:15",
                "minutesAsleep": 60,
                "minutesAwake": 15,
                "timeInBed": 75,
                "type": "stages",
                "mainSleep": True,
                "deep_minutes": 20,
                "light_minutes": 30,
                "rem_minutes": 10,
            },
            {
                "logId": "nap",
                "dateOfSleep": "2024-01-02",
                "startTime": "2024-01-02 12:00",
                "endTime": "2024-01-02 12:30",
                "minutesAsleep": 25,
                "minutesAwake": 5,
                "timeInBed": 30,
                "type": "classic",
                "mainSleep": False,
            },
            {
                "logId": "classic-night",
                "dateOfSleep": "2024-01-03",
                "startTime": "2024-01-02 22:00",
                "endTime": "2024-01-02 23:10",
                "minutesAsleep": 60,
                "minutesAwake": 10,
                "timeInBed": 70,
                "type": "classic",
                "mainSleep": True,
            },
        ]
    )
    stages = pd.DataFrame(
        {
            "logId": [
                "staged-night",
                "staged-night",
                "staged-night",
                "staged-night",
                "staged-night",
                "nap",
                "nap",
                "classic-night",
                "classic-night",
                "classic-night",
            ],
            "dateTime": pd.to_datetime(
                [
                    "2024-01-01 22:00",
                    "2024-01-01 22:05",
                    "2024-01-01 22:35",
                    "2024-01-01 22:55",
                    "2024-01-01 23:10",
                    "2024-01-02 12:00",
                    "2024-01-02 12:25",
                    "2024-01-02 22:00",
                    "2024-01-02 22:05",
                    "2024-01-02 23:05",
                ]
            ),
            "seconds": [300, 1800, 1200, 900, 300, 1500, 300, 300, 3600, 300],
            "level": [
                "wake",
                "light",
                "deep",
                "rem",
                "wake",
                "asleep",
                "wake",
                "restless",
                "asleep",
                "wake",
            ],
        }
    )
    short_stages = pd.DataFrame(
        {
            "logId": ["staged-night"],
            "dateTime": pd.to_datetime(["2024-01-01 23:00"]),
            "seconds": [300],
            "level": ["wake"],
        }
    )

    def read_parquet(path):
        if path.name == "sleep.parquet":
            return summaries.copy()
        if path.name == "sleep_short_stages.parquet":
            return short_stages.copy()
        return stages.copy()

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    loader = LifeSnapsParquetLoader(tmp_path)

    sessions = loader.load_sleep_sessions("person-1")
    assert sessions["id"].tolist() == ["staged-night", "nap", "classic-night"]
    assert sessions.set_index("id")["main_sleep"].to_dict() == {
        "staged-night": True,
        "nap": False,
        "classic-night": True,
    }

    loaded_summaries = loader.load_sleep_summary("person-1")
    assert loaded_summaries["sleep_summary_id"].tolist() == [
        "staged-night",
        "classic-night",
    ]
    assert loaded_summaries["duration_in_ms"].tolist() == [
        60 * 60_000,
        60 * 60_000,
    ]
    assert loaded_summaries["awake_duration_in_ms"].tolist() == [
        15 * 60_000,
        10 * 60_000,
    ]
    assert loaded_summaries["local_end_time"].tolist() == pd.to_datetime(
        ["2024-01-01 23:15", "2024-01-02 23:10"]
    ).tolist()
    assert loaded_summaries["utc_time"].isna().all()
    assert loaded_summaries["timezone_offset_in_ms"].isna().all()

    loaded_stages = loader.load_sleep_stage("person-1")
    assert "nap" not in set(loaded_stages["sleep_summary_id"])
    all_session_stages = loader.load_sleep_session_stages("person-1")
    assert "nap" in set(all_session_stages["sleep_summary_id"])
    assert set(loaded_stages["type"]) == {
        "awake",
        "asleep",
        "light",
        "deep",
        "rem",
    }
    assert loaded_stages["utc_time"].isna().all()
    assert loaded_stages["timezone_offset_in_ms"].isna().all()

    features = sleep.get_sleep_statistics(loader, user_id="person-1")
    np.testing.assert_allclose(features["sleep_TIB"], [75, 70])
    np.testing.assert_allclose(features["sleep_TST"], [60, 60])
    np.testing.assert_allclose(features["sleep_SOL"], [5, 5])
    np.testing.assert_allclose(features["sleep_WASO"], [5, 0])
    assert features["sleep_count_awake"].tolist() == [3, 2]
    assert features.loc[("person-1", pd.Timestamp("2024-01-02")), "sleep_count_REM"] == 2
    assert pd.isna(
        features.loc[("person-1", pd.Timestamp("2024-01-03")), "sleep_count_REM"]
    )
    assert pd.isna(features.loc[("person-1", pd.Timestamp("2024-01-03")), "sleep_%_REM"])


def test_lifesnaps_marks_unknown_intervals_unmeasurable(tmp_path, monkeypatch):
    user_folder = tmp_path / "person-1"
    user_folder.mkdir()
    (user_folder / "sleep.parquet").touch()
    (user_folder / "sleep_stages.parquet").touch()

    summaries = pd.DataFrame(
        [
            {
                "logId": "night",
                "dateOfSleep": "2024-01-02",
                "startTime": "2024-01-01 22:00",
                "endTime": "2024-01-01 23:20",
                "minutesAsleep": 60,
                "minutesAwake": 10,
                "timeInBed": 80,
                "type": "stages",
                "mainSleep": True,
                "deep_minutes": 20,
                "light_minutes": 30,
                "rem_minutes": 10,
            }
        ]
    )
    stages = pd.DataFrame(
        {
            "logId": ["night", "night", "night", "night", "night"],
            "dateTime": pd.to_datetime(
                [
                    "2024-01-01 22:00",
                    "2024-01-01 22:10",
                    "2024-01-01 22:40",
                    "2024-01-01 23:00",
                    "2024-01-01 23:10",
                ]
            ),
            "seconds": [600, 1800, 1200, 600, 600],
            "level": ["wake", "light", "deep", "rem", "unknown"],
        }
    )

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: summaries.copy()
        if path.name == "sleep.parquet"
        else stages.copy(),
    )
    loader = LifeSnapsParquetLoader(tmp_path)

    summary = loader.load_sleep_summary("person-1")
    loaded_stages = loader.load_sleep_stage("person-1")
    features = sleep.get_sleep_statistics(loader, user_id="person-1")

    assert summary.loc[0, "duration_in_ms"] == 70 * 60_000
    assert summary.loc[0, "awake_duration_in_ms"] == 10 * 60_000
    assert summary.loc[0, "unmeasurable_sleep_duration_in_ms"] == 10 * 60_000
    assert loaded_stages.iloc[-1]["type"] == "unmeasurable"
    assert features.iloc[0]["sleep_TIB"] == 80
    assert features.iloc[0]["sleep_TST"] == 60
    assert features.iloc[0]["sleep_unmeasurable"] == 10


def test_lifesnaps_drops_only_incoherent_stages(tmp_path, monkeypatch):
    user_folder = tmp_path / "person-1"
    user_folder.mkdir()
    (user_folder / "sleep.parquet").touch()
    (user_folder / "sleep_stages.parquet").touch()

    summaries = pd.DataFrame(
        [
            {
                "logId": "conflicting-night",
                "dateOfSleep": "2024-01-02",
                "startTime": "2024-01-01 22:00",
                "endTime": "2024-01-01 23:10",
                "minutesAsleep": 60,
                "minutesAwake": 10,
                "timeInBed": 70,
                "type": "classic",
                "mainSleep": True,
            }
        ]
    )
    stages = pd.DataFrame(
        {
            "logId": ["conflicting-night", "conflicting-night"],
            "dateTime": pd.to_datetime(
                ["2024-01-01 22:00", "2024-01-01 22:30"]
            ),
            "seconds": [1800, 2400],
            "level": ["asleep", "awake"],
        }
    )
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: summaries.copy()
        if path.name == "sleep.parquet"
        else stages.copy(),
    )
    loader = LifeSnapsParquetLoader(tmp_path)

    with pytest.warns(UserWarning, match="stage durations conflict"):
        loaded_stages = loader.load_sleep_stage("person-1")
    with pytest.warns(UserWarning, match="stage durations conflict"):
        features = sleep.get_sleep_statistics(loader, user_id="person-1")

    assert loaded_stages.empty
    assert features.iloc[0]["sleep_TST"] == 60
    assert pd.isna(features.iloc[0]["sleep_SPT"])
