import datetime
import os

import pandas as pd
import pytest

import pywearable.constants
import pywearable.loader
import pywearable.respiration

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MOCK_VALUES_DIR = os.path.join(CURRENT_DIR, "mock_values")
PULSE_OX_USER_ID = "01"


@pytest.fixture
def pulse_ox():
    """
    pytest.fixture that returns pulse ox data in the
    format required by pywearable.
    """
    pulse_ox_file = os.path.join(MOCK_VALUES_DIR, "pulse_ox.csv")
    pulse_ox = pd.read_csv(pulse_ox_file)
    pulse_ox[pywearable.constants.ISODATE_COL] = pd.to_datetime(
        pulse_ox[pywearable.constants.UNIXTIMESTAMP_IN_MS_COL]
        + pulse_ox[pywearable.constants.TIMEZONEOFFSET_IN_MS_COL],
        unit="ms",
        utc=True,
    ).dt.tz_localize(None)
    pulse_ox[pywearable.constants.CALENDAR_DATE_COL] = pd.to_datetime(
        pulse_ox[pywearable.constants.CALENDAR_DATE_COL], format="%Y-%m-%d"
    ).dt.date
    return pulse_ox


@pytest.fixture()
def base_loader(monkeypatch, pulse_ox):
    """
    Fixture that returns a BaseLoader with mock values
    for the load_sleep_summary and load_sleep_stage
    functions. The value returned by these functions
    are specified by the sleep_summary and
    sleep_stages fixtures.
    """
    monkeypatch.setattr(pywearable.loader.BaseLoader, "__abstractmethods__", frozenset())
    loader = pywearable.loader.BaseLoader()
    monkeypatch.setattr(
        loader, "load_sleep_pulse_ox", lambda **kwargs: pulse_ox.copy()
    )
    monkeypatch.setattr(loader, "get_user_ids", lambda: [PULSE_OX_USER_ID])
    return loader


def test_mean_rest_pulse_ox(base_loader, pulse_ox):
    mean_rest_pulse_ox = pywearable.respiration.get_mean_rest_pulse_ox(base_loader)
    metric = pywearable.respiration._RESPIRATION_METRIC_MEAN_PULSE_OX
    expected = pulse_ox.groupby(pywearable.constants.CALENDAR_DATE_COL)[
        pywearable.constants.SPO2_SPO2_COL
    ].mean()
    expected.index = pd.to_datetime(expected.index)
    expected.name = metric

    assert isinstance(mean_rest_pulse_ox, pd.DataFrame)
    assert mean_rest_pulse_ox.index.names == [
        pywearable.constants.USER_COL,
        pywearable.constants.CALENDAR_DATE_COL,
    ]
    pd.testing.assert_series_equal(
        mean_rest_pulse_ox.xs(
            PULSE_OX_USER_ID, level=pywearable.constants.USER_COL
        )[metric],
        expected,
        check_freq=False,
    )
