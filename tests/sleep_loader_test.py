import pandas as pd

from pywearable.loader.tesserae import TesseraeParquetLoader


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
            "light_sleep_seconds": ["2400"],
            "validation": ["AUTO_FINAL"],
            "sleep_levels_map": ["{}"],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())

    result = TesseraeParquetLoader(tmp_path).load_sleep_summary("person-1")

    assert result.loc[0, "duration_in_ms"] == 3_600_000
    assert result.loc[0, "rem_sleep_duration_in_ms"] == 0


def test_tesserae_empty_sleep_stage_uses_shared_contract(tmp_path):
    (tmp_path / "person-1").mkdir()

    result = TesseraeParquetLoader(tmp_path).load_sleep_stage(
        "person-1", start_date=None, end_date=None
    )

    assert {
        "sleep_summary_id",
        "utc_time",
        "local_time",
        "duration_in_ms",
        "type",
    } <= set(result.columns)
