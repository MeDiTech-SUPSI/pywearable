import datetime
from pathlib import Path

import pandas as pd
import pytest

import pywearable.loader
from pywearable.loader.labfront import constants as lf_constants


@pytest.fixture
def loader(scope="session"):
    return pywearable.loader.LabfrontLoader(Path(__file__).parent / "sample_data")


def test_get_user_ids(loader):
    user_ids = loader.get_user_ids()
    assert type(user_ids) == list
    assert user_ids == ["user-01_673"]


def test_get_full_ids(loader):
    full_ids = loader.get_full_ids()
    assert type(full_ids) == list
    assert full_ids == ["user-01_673"]


def test_get_full_id(loader):
    full_id = loader.get_full_id("user-01")
    assert type(full_id) == str
    assert full_id == "user-01_673"


def test_get_user_id(loader):
    user_id = loader.get_user_id("user-01_673")
    assert type(user_id) == str
    assert user_id == "user-01"


def test_get_labfront_ids(loader: pywearable.loader.LabfrontLoader):
    labfront_id = loader.get_labfront_ids()
    assert type(labfront_id) == list
    assert labfront_id == [673]


def test_load_heart_rate_invalid_user_id(loader: pywearable.loader.LabfrontLoader):
    with pytest.raises(ValueError):
        loader.load_garmin_connect_heart_rate("user-02")


@pytest.mark.parametrize("start_date", [None])
def test_load_heart_rate_is_df(loader: pywearable.loader.LabfrontLoader, start_date):
    df = loader.load_garmin_connect_heart_rate("user-01")
    assert isinstance(df, pd.DataFrame)


def test_load_heart_rate_is_empty(loader: pywearable.loader.LabfrontLoader):
    df = loader.load_garmin_connect_heart_rate(
        "user-01",
        start_date=datetime.datetime(2030, 1, 1),
        end_date=datetime.datetime(2030, 1, 2),
    )
    assert len(df) == 0


def test_load_heart_rate_uses_shared_contract(loader: pywearable.loader.LabfrontLoader):
    df = loader.load_heart_rate(
        "user-01",
        start_date="2023-01-03 12:23:00",
        end_date="2023-01-03 12:24:00",
    )
    assert {"utc_time", "local_time", "timezone", "heart_rate"} <= set(df)
    assert (df["local_time"] >= pd.Timestamp("2023-01-03 12:23:00")).all()
    assert (df["local_time"] < pd.Timestamp("2023-01-03 12:24:00")).all()
    assert (df["timezone"] == pd.Timedelta(hours=8)).all()


def test_load_sdk_heart_rate_removes_invalid_sentinel_values(
    loader: pywearable.loader.LabfrontLoader,
):
    df = loader.load_heart_rate(
        "user-01", source=lf_constants.DATA_SOURCE_GARMIN_SDK
    )
    assert (df["heart_rate"] > 0).all()
    assert "timezone_name" in df
