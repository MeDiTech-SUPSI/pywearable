"""
This conftest.py exposes a fixture, called ``base_loader``
that returns mock values for all the functions defined
in ``pywearable.loader.BaseLoader``. The mock values are not 
linked one with respect to the other (e.g., heart-rate data 
are not related to average values in daily-sleep-summary) so
if checks of computations are required, then it is necessary
to use other types of data different than the ones
defined by this fixture.
"""

from pathlib import Path

import pandas as pd
import pytest

from pywearable.loader import BaseLoader
from pywearable.loader.labfront import constants as lf_constants

BASE_PATH = Path(__file__).parent / "mock_values"

MOCK_FILES_DICT = {
    lf_constants.METRIC_BBI: BASE_PATH / "bbi.csv",
    lf_constants.METRIC_DAILY_SUMMARY: BASE_PATH / "daily_summary.csv",
    lf_constants.METRIC_HEART_RATE: BASE_PATH / "heart_rate.csv",
    lf_constants.METRIC_SLEEP_RESPIRATION: BASE_PATH / "sleep_respiration.csv",
    lf_constants.METRIC_SLEEP_STAGE: BASE_PATH / "sleep_stage.csv",
    lf_constants.METRIC_SLEEP_SUMMARY: BASE_PATH / "sleep_summary.csv",
}


def load_sleep_stage() -> pd.DataFrame:
    sleep_stage = pd.read_csv(
        MOCK_FILES_DICT[lf_constants.METRIC_SLEEP_STAGE],
        parse_dates=[lf_constants.ISODATE_COL],
    )
    sleep_stage[lf_constants.ISODATE_COL] = sleep_stage[
        lf_constants.ISODATE_COL
    ].dt.tz_localize(None)
    return sleep_stage


def load_sleep_respiration() -> pd.DataFrame:
    sleep_respiration = pd.read_csv(
        MOCK_FILES_DICT[lf_constants.METRIC_SLEEP_RESPIRATION],
        parse_dates=[lf_constants.ISODATE_COL],
    )
    sleep_respiration[lf_constants.ISODATE_COL] = sleep_respiration[
        lf_constants.ISODATE_COL
    ].dt.tz_localize(None)
    return sleep_respiration


def load_heart_rate() -> pd.DataFrame:
    heart_rate = pd.read_csv(
        MOCK_FILES_DICT[lf_constants.METRIC_HEART_RATE],
        parse_dates=[lf_constants.ISODATE_COL],
    )
    heart_rate[lf_constants.ISODATE_COL] = heart_rate[
        lf_constants.ISODATE_COL
    ].dt.tz_localize(None)
    return heart_rate


@pytest.fixture(scope="session")
def base_loader(session_mocker):
    loader = BaseLoader()
    # load_daily_summary
    session_mocker.patch(
        "pywearable.loader.BaseLoader.load_daily_summary",
        return_value=pd.read_csv(
            MOCK_FILES_DICT[lf_constants.METRIC_DAILY_SUMMARY],
            parse_dates=[
                lf_constants.CALENDAR_DATE_COL,
                lf_constants.ISODATE_COL,
            ],
        ),
    )
    # load_heart_rate
    session_mocker.patch(
        "pywearable.loader.BaseLoader.load_heart_rate", return_value=load_heart_rate()
    )
    session_mocker.patch(
        "pywearable.loader.BaseLoader.get_user_ids", return_value=["user-01"]
    )
    # load_bbi
    bbi = pd.read_csv(
        MOCK_FILES_DICT[lf_constants.METRIC_BBI],
        parse_dates=[lf_constants.ISODATE_COL],
    )
    bbi[lf_constants.ISODATE_COL] = bbi[
        lf_constants.ISODATE_COL
    ].dt.tz_localize(None)
    session_mocker.patch(
        "pywearable.loader.BaseLoader.load_bbi",
        return_value=bbi,
    )

    # load_sleep_summary
    sleep_summary = pd.read_csv(
        MOCK_FILES_DICT[lf_constants.METRIC_SLEEP_SUMMARY],
        parse_dates=[
            lf_constants.ISODATE_COL,
            lf_constants.CALENDAR_DATE_COL,
        ],
    )
    sleep_summary[lf_constants.ISODATE_COL] = sleep_summary[
        lf_constants.ISODATE_COL
    ].dt.tz_localize(None)
    sleep_summary[lf_constants.CALENDAR_DATE_COL] = sleep_summary[
        lf_constants.CALENDAR_DATE_COL
    ].dt.tz_localize(None)
    session_mocker.patch(
        "pywearable.loader.BaseLoader.load_sleep_summary", return_value=sleep_summary
    )
    session_mocker.patch(
        "pywearable.loader.BaseLoader.load_sleep_stage", return_value=load_sleep_stage()
    )
    # get_user_ids
    session_mocker.patch(
        "pywearable.loader.BaseLoader.get_user_ids",
        return_value=["user-01"],
    )
    return loader
