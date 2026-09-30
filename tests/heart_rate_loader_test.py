import pandas as pd
import pytest

from pywearable.loader import LifeSnapsParquetLoader
from pywearable.loader.sphyncs import SphyncsDataLoader
from pywearable.loader.tesserae import TesseraeParquetLoader


REQUIRED_COLUMNS = {"utc_time", "local_time", "timezone", "heart_rate"}


def test_sphyncs_heart_rate_uses_local_only_shared_contract(tmp_path, monkeypatch):
    user_folder = tmp_path / "person-1"
    user_folder.mkdir()
    (user_folder / "biometrics.parquet").touch()
    raw = pd.DataFrame(
        {
            "dateTime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:30"]
            ),
            "heart": [60, 70],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())
    loader = SphyncsDataLoader(tmp_path)

    result = loader.load_heart_rate(
        "person-1", "2024-01-01 00:15", "2024-01-01 01:00"
    )

    assert REQUIRED_COLUMNS <= set(result)
    assert result["heart_rate"].tolist() == [70]
    assert result["utc_time"].isna().all()
    with pytest.raises(ValueError, match="does not provide UTC"):
        loader.load_heart_rate("person-1", time_col="utc_time")


def test_lifesnaps_heart_rate_uses_shared_contract_and_selected_clock(
    tmp_path, monkeypatch
):
    user_folder = tmp_path / "person-1"
    user_folder.mkdir()
    (user_folder / "heart_rate.parquet").touch()
    raw = pd.DataFrame(
        {
            "dateTime": ["2024-01-01T23:30:00", "2024-01-02T00:30:00"],
            "bpm": [60, 70],
            "confidence": [3, 3],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())

    result = LifeSnapsParquetLoader(tmp_path).load_heart_rate(
        "person-1",
        start_date="2024-01-02",
        end_date="2024-01-03",
    )
    assert REQUIRED_COLUMNS <= set(result)
    assert result["heart_rate"].tolist() == [70]
    assert result["utc_time"].isna().all()
    assert result["timezone"].isna().all()

    with pytest.raises(ValueError, match="does not provide UTC"):
        LifeSnapsParquetLoader(tmp_path).load_heart_rate(
            "person-1", time_col="utc_time"
        )


def test_tesserae_heart_rate_keeps_latest_duplicate_and_filters_selected_clock(
    tmp_path, monkeypatch
):
    user_folder = tmp_path / "person-1" / "GarminHealthAPI"
    user_folder.mkdir(parents=True)
    (user_folder / "HR.parquet").touch()
    raw = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2024-01-01T23:30:00Z",
                    "2024-01-01T23:30:00Z",
                    "2024-01-02T00:30:00Z",
                ],
                utc=True,
            ),
            "local_time": pd.to_datetime(
                ["2024-01-02 00:30", "2024-01-02 00:30", "2024-01-02 01:30"]
            ),
            "hr": [60, 62, 70],
            "tz_source": ["data", "data", "data"],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())

    result = TesseraeParquetLoader(tmp_path).load_heart_rate(
        "person-1",
        start_date="2024-01-02 00:00",
        end_date="2024-01-02 01:00",
    )
    assert REQUIRED_COLUMNS <= set(result)
    assert result["heart_rate"].tolist() == [62]
    assert result.loc[0, "timezone"] == pd.Timedelta(hours=1)
    assert result.loc[0, "timezone_offset"] == pd.Timedelta(hours=1)
