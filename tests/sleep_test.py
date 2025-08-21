import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import numpy.testing
import pandas as pd
import pandas.testing
import pytest

import pywearable.constants
import pywearable.loader
import pywearable.sleep

# TODO tests to be added:
# - users with no data
# - add filtering function as argument to computation
# - start and end data with no date for user


@pytest.fixture
def sleep_summary():
    """
    pytest.fixture that returns sleep summary in the
    format required by pywearable.
    """
    sleep_summary = pd.read_csv(
        StringIO(
            """sleepSummaryId,calendarDate,timezoneOffsetInMs,unixTimestampInMs,isoDate,durationInMs,unmeasurableSleepDurationInMs,deepSleepDurationInMs,lightSleepDurationInMs,remSleepDurationInMs,awakeDurationInMs,overallSleepScore
x4c64722-64595538-6630,2023-05-09,7200000,1683576120000,2023-05-08T22:02:00.000+02:00,26160000,0,3900000,15480000,6780000,780000,88
x4c64722-645ab9b4-55c8,2023-05-10,7200000,1683667380000,2023-05-09T23:23:00.000+02:00,21960000,,,,,,
x4c64722-645bf6d0-6888,2023-05-11,7200000,1683748560000,2023-05-10T21:56:00.000+02:00,26760000,0,26760000,0,0,0,40
x4c64722-645d63f8-5c94,2023-05-12,7200000,1683842040000,2023-05-11T23:54:00.000+02:00,23700000,0,0,23700000,0,600000,70
            """
        ),
        parse_dates=[pywearable.constants.CALENDAR_DATE_COL],
    )
    sleep_summary[pywearable.constants.ISODATE_COL] = pd.to_datetime(
        sleep_summary[pywearable.constants.UNIXTIMESTAMP_IN_MS_COL]
        + sleep_summary[pywearable.constants.TIMEZONEOFFSET_IN_MS_COL],
        unit="ms",
        utc=True,
    ).dt.tz_localize(None)
    return sleep_summary


@pytest.fixture
def sleep_stages():
    """
    pytest.fixture that returns sleep stages in the format required
    by pywearable.
    """
    sleep_stages = pd.read_csv(
        StringIO(
            """sleepSummaryId,timezoneOffsetInMs,unixTimestampInMs,isoDate,durationInMs,type
x4c64722-64595538-6630,7200000,1683576120000,2023-05-08T22:02:00.000+02:00,600000,light
x4c64722-64595538-6630,7200000,1683576720000,2023-05-08T22:12:00.000+02:00,960000,deep
x4c64722-64595538-6630,7200000,1683577680000,2023-05-08T22:28:00.000+02:00,60000,awake
x4c64722-64595538-6630,7200000,1683577740000,2023-05-08T22:29:00.000+02:00,600000,light
x4c64722-64595538-6630,7200000,1683578340000,2023-05-08T22:39:00.000+02:00,720000,deep
x4c64722-64595538-6630,7200000,1683579060000,2023-05-08T22:51:00.000+02:00,1800000,light
x4c64722-64595538-6630,7200000,1683580860000,2023-05-08T23:21:00.000+02:00,2700000,rem
x4c64722-64595538-6630,7200000,1683583560000,2023-05-09T00:06:00.000+02:00,840000,light
x4c64722-64595538-6630,7200000,1683584400000,2023-05-09T00:20:00.000+02:00,1080000,deep
x4c64722-64595538-6630,7200000,1683585480000,2023-05-09T00:38:00.000+02:00,2520000,light
x4c64722-64595538-6630,7200000,1683588000000,2023-05-09T01:20:00.000+02:00,1260000,rem
x4c64722-64595538-6630,7200000,1683589260000,2023-05-09T01:41:00.000+02:00,2760000,light
x4c64722-64595538-6630,7200000,1683592020000,2023-05-09T02:27:00.000+02:00,360000,deep
x4c64722-64595538-6630,7200000,1683592380000,2023-05-09T02:33:00.000+02:00,240000,light
x4c64722-64595538-6630,7200000,1683592620000,2023-05-09T02:37:00.000+02:00,660000,awake
x4c64722-64595538-6630,7200000,1683593280000,2023-05-09T02:48:00.000+02:00,1440000,light
x4c64722-64595538-6630,7200000,1683594720000,2023-05-09T03:12:00.000+02:00,780000,deep
x4c64722-64595538-6630,7200000,1683595500000,2023-05-09T03:25:00.000+02:00,540000,light
x4c64722-64595538-6630,7200000,1683596040000,2023-05-09T03:34:00.000+02:00,2820000,rem
x4c64722-64595538-6630,7200000,1683598860000,2023-05-09T04:21:00.000+02:00,960000,light
x4c64722-64595538-6630,7200000,1683599820000,2023-05-09T04:37:00.000+02:00,60000,awake
x4c64722-64595538-6630,7200000,1683599880000,2023-05-09T04:38:00.000+02:00,3180000,light
x4c64722-645bf6d0-6888,7200000,1683748560000,2023-05-10T21:56:00.000+02:00,26760000,deep
x4c64722-645d63f8-5c94,7200000,1683842040000,2023-05-11T23:54:00.000+02:00,600000,awake
x4c64722-645d63f8-5c94,7200000,1683842640000,2023-05-12T00:04:00.000+02:00,23700000,light
        """
        )
    )
    sleep_stages[pywearable.constants.ISODATE_COL] = pd.to_datetime(
        sleep_stages[pywearable.constants.UNIXTIMESTAMP_IN_MS_COL]
        + sleep_stages[pywearable.constants.TIMEZONEOFFSET_IN_MS_COL],
        unit="ms",
        utc=True,
    ).dt.tz_localize(None)
    return sleep_stages


@pytest.fixture
def sleep_score():
    return pd.Series(
        [88, np.nan, 40, 70],
        index=[
            "x4c64722-64595538-6630",
            "x4c64722-645ab9b4-55c8",
            "x4c64722-645bf6d0-6888",
            "x4c64722-645d63f8-5c94",
        ],
    )


# Fixture for sleep integration tests
@pytest.fixture()
def base_loader(mocker, sleep_summary, sleep_stages):
    """
    Fixture that returns a BaseLoader with mock values
    for the load_sleep_summary and load_sleep_stage
    functions. The value returned by these functions
    are specified by the sleep_summary and
    sleep_stages fixtures.
    """
    loader = pywearable.loader.BaseLoader()
    mocker.patch(
        "pywearable.loader.BaseLoader.load_sleep_summary",
        return_value=sleep_summary,
    )
    mocker.patch(
        "pywearable.loader.BaseLoader.load_sleep_stage", return_value=sleep_stages
    )
    return loader


SLEEP_SERIES_INDEX = pd.Index(
    [
        pd.Timestamp(2023, 5, 9),
        pd.Timestamp(2023, 5, 10),
        pd.Timestamp(2023, 5, 11),
        pd.Timestamp(2023, 5, 12),
    ],
    name=pywearable.constants.CALENDAR_DATE_COL,
)

# Index with days from sleep stages
SLEEP_STAGES_SERIES_INDEX = pd.Index(
    [
        pd.Timestamp(2023, 5, 9),
        pd.Timestamp(2023, 5, 11),
        pd.Timestamp(2023, 5, 12),
    ],
    name=pywearable.constants.CALENDAR_DATE_COL,
)


def test_get_sleep_score(base_loader):
    # Check with default options
    sleep_score = pywearable.sleep.get_sleep_score(base_loader, user_id="1")
    assert type(sleep_score) is pd.Series
    assert sleep_score.name == pywearable.sleep.SLEEP_STATISTIC_SLEEP_SCORE
    assert len(sleep_score) == len(SLEEP_SERIES_INDEX)

    # Check with dictionary
    sleep_score = pywearable.sleep.get_sleep_score(
        base_loader, user_id="1", return_df=False
    )
    assert type(sleep_score) is dict
    assert "1" in sleep_score.keys()
    for day in SLEEP_SERIES_INDEX:
        assert day.date() in sleep_score["1"].keys()


def test_get_time_in_bed(base_loader):
    # Check with default options
    tib = pywearable.sleep.get_time_in_bed(base_loader, user_id="1")
    assert type(tib) is pd.Series
    assert tib.name == pywearable.sleep.SLEEP_STATISTIC_TIB

    tib = pywearable.sleep.get_time_in_bed(base_loader, user_id="1", return_df=False)
    assert type(tib) is dict
    assert "1" in tib.keys()
    for day in SLEEP_SERIES_INDEX:
        assert day.date() in tib["1"].keys()


def test_get_total_sleep_time(base_loader):
    """
    Aim: Test that Total Sleep Time (TST) computed
    starting from a loader returns the proper
    values.
    """
    # Test with default parameters
    tst = pywearable.sleep.get_total_sleep_time(
        base_loader, user_id="1", return_df=True, return_multi_index=True
    )
    assert type(tst) is pd.Series
    assert tst.name == pywearable.sleep.SLEEP_STATISTIC_TST

    tst = pywearable.sleep.get_total_sleep_time(
        base_loader, user_id="1", return_df=False
    )
    assert type(tst) is dict
    assert "1" in tst.keys()
    for day in SLEEP_SERIES_INDEX:
        assert day.date() in tst["1"].keys()


def test_get_sleep_period_time(base_loader):
    """
    Aim: Test that Sleep Period Time (SPT) computed
    starting from a loader returns the proper
    values.
    """
    spt = pywearable.sleep.get_sleep_period_time(
        base_loader, user_id="1", return_df=True, return_multi_index=True
    )
    spt_true = pd.Series(
        [449.0, np.nan, 446.0, 395],
        index=pd.MultiIndex.from_product(
            [
                ["1"],
                [
                    pd.Timestamp(2023, 5, 9),
                    pd.Timestamp(2023, 5, 10),
                    pd.Timestamp(2023, 5, 11),
                    pd.Timestamp(2023, 5, 12),
                ],
            ],
            names=[
                pywearable.constants.USER_COL,
                pywearable.constants.CALENDAR_DATE_COL,
            ],
        ),
        name=pywearable.sleep.SLEEP_STATISTIC_SPT,
    )
    pd.testing.assert_series_equal(spt, spt)


def test_get_sleep_efficiency(base_loader):
    """
    Aim: test that the computation of sleep
    efficiency is correct and that the series
    is reindex correctly
    """
    sleep_eff = pywearable.sleep.get_sleep_efficiency(loader=base_loader, user_id="1")
    assert type(sleep_eff) is pd.Series
    assert len(sleep_eff) > 0

    # Sleep efficiency is computed as the ratio
    # between total sleep time (light+deep+rem in SPT) and time in bed (time from
    # bedtime to wakeup)
    true_sleep_eff = pd.Series(
        [436.0 / (436.0 + 13.0) * 100, np.nan, 100.0, 395.0 / (395.0 + 10.0) * 100],
        index=pd.MultiIndex.from_product(
            [["1"], SLEEP_SERIES_INDEX],
            names=[
                pywearable.constants.USER_COL,
                pywearable.constants.CALENDAR_DATE_COL,
            ],
        ),
        name=pywearable.sleep.SLEEP_STATISTIC_SE,
    )

    pd.testing.assert_series_equal(sleep_eff, true_sleep_eff)


def test_get_sleep_maintenance_efficiency(base_loader):
    """
    Aim: Tet the computation of sleep maintenance efficiency
    through the "get" function.
    """
    sleep_me = pywearable.sleep.get_sleep_maintenance_efficiency(
        loader=base_loader, user_id=["1"]
    )
    assert type(sleep_me) is pd.Series
    assert len(sleep_me) > 0

    # Check that the values are

    # Sleep maintenance efficiency is computed as
    # the ratio between TST (light+deep+rem) and SPT (overall sleep)
    true_sleep_me = pd.Series(
        [], index=SLEEP_SERIES_INDEX, name=pywearable.sleep.SLEEP_STATISTIC_SME
    )

######################################################
#              Compute Functions Tests               #
######################################################


def test_compute_bedtime(sleep_summary: pd.DataFrame):
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    bedtime = pywearable.sleep._compute_bedtime(sleep_summary)
    print(bedtime)
    assert type(bedtime) == pd.Series
    pandas.testing.assert_series_equal(
        bedtime,
        pd.Series(
            [
                pd.Timestamp(2023, 5, 8, 22, 2),
                pd.Timestamp(2023, 5, 9, 23, 23),
                pd.Timestamp(2023, 5, 10, 21, 56),
                pd.Timestamp(2023, 5, 11, 23, 54),
            ],
            index=SLEEP_SERIES_INDEX,
            name=pywearable.sleep.SLEEP_STATISTIC_BEDTIME,
        ),
        check_names=False,
        check_dtype=True,
    )


def test_compute_wakeup_time(sleep_summary: pd.DataFrame):
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    wakeup_time = pywearable.sleep._compute_wakeup_time(sleep_summary)
    assert type(wakeup_time) == pd.Series
    pandas.testing.assert_series_equal(
        wakeup_time,
        pd.Series(
            [
                pd.Timestamp(2023, 5, 9, 5, 31),
                pd.Timestamp(2023, 5, 10, 5, 29),
                pd.Timestamp(2023, 5, 11, 5, 22),
                pd.Timestamp(2023, 5, 12, 6, 39),
            ],
            index=SLEEP_SERIES_INDEX,
            name=pywearable.sleep.SLEEP_STATISTIC_WAKEUP,
        ),
        check_names=False,
        check_dtype=True,
    )


def test_compute_sleep_score(sleep_summary: pd.DataFrame):
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    sleep_score = pywearable.sleep._compute_sleep_score(sleep_summary)
    assert type(sleep_score) == pd.Series
    pandas.testing.assert_series_equal(
        sleep_score,
        pd.Series(
            [88.0, np.nan, 40.0, 70.0],
            index=SLEEP_SERIES_INDEX,
            name=pywearable.sleep.SLEEP_STATISTIC_SLEEP_SCORE,
        ),
        check_names=False,
        check_dtype=True,
    )


def test_compute_time_in_bed(sleep_summary: pd.DataFrame):
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    time_in_bed = pywearable.sleep._compute_time_in_bed(sleep_summary)
    # Time is define as the total time spent in bed: thus
    # it is the duration from the start up to the real wakeup
    assert type(time_in_bed) == pd.Series
    pandas.testing.assert_series_equal(
        time_in_bed,
        pd.Series([449.0, np.nan, 446.0, 405.0], index=SLEEP_SERIES_INDEX),
        check_names=False,
        check_dtype=True,
    )


def test_compute_total_sleep_time(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame
):
    # Total Sleep Time is the total duration of
    # deep, light, and REM sleep in the sleep period time (SPT)
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    tst = pywearable.sleep._compute_total_sleep_time(
        sleep_summary=sleep_summary,
        sleep_stages=sleep_stages,
    )
    correct_tst = pd.Series(
        [436.0, np.nan, 446.0, 395.0],
        index=SLEEP_SERIES_INDEX,
    )
    pd.testing.assert_series_equal(left=tst, right=correct_tst, check_names=False)


def test_compute_sleep_efficiency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame
):
    # Sleep efficiency is the ratio between
    # total sleep time and sleep period time,
    # thus telling how much time was spent
    # sleeping inside the sleep phase
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    sleep_efficiency = pywearable.sleep._compute_sleep_efficiency(
        sleep_summary, sleep_stages
    )
    assert type(sleep_efficiency) == pd.Series
    # First sleep has 13 minutes spent awake, 436 sleeping
    # -> 97.1%
    numpy.testing.assert_almost_equal(
        sleep_efficiency[pd.Timestamp(2023, 5, 9)], 97.1, decimal=1
    )
    # Second sleep has no sleep stages, so it is nan
    # -> 97.1%
    # numpy.testing.assert_equal(sleep_efficiency[pd.Timestamp(2023, 5, 10)], np.nan)
    # Third sleep has no minutes spent awake -> 100%
    numpy.testing.assert_almost_equal(
        sleep_efficiency[pd.Timestamp(2023, 5, 11)], 100.0, decimal=1
    )
    # Fourth sleep has some awake stages
    numpy.testing.assert_almost_equal(
        sleep_efficiency[pd.Timestamp(2023, 5, 12)], 97.5, decimal=1
    )


def test_compute_sleep_maintenance_efficiency(sleep_summary_id_as_idx, sleep_stages):
    sleep_maintenance_efficiency = (
        pywearable.sleep._compute_sleep_maintenance_efficiency(
            sleep_summary=sleep_summary_id_as_idx, sleep_stages=sleep_stages
        )
    )
    assert type(sleep_maintenance_efficiency) == pd.Series
    numpy.testing.assert_almost_equal(
        sleep_maintenance_efficiency["x4c64722-64595538-6630"], 97.1, decimal=1
    )


def test_compute_awake_count(sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame):
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    awake_count = pywearable.sleep._compute_awake_count(sleep_summary, sleep_stages)
    pd.testing.assert_series_equal(
        awake_count,
        pd.Series(
            [3.0, 0.0, 1.0],
            index=SLEEP_STAGES_SERIES_INDEX,
        ),
        check_names=False,
    )


def test_compute_latencies(sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame):
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    sleep_latencies = pywearable.sleep._compute_latencies(sleep_summary, sleep_stages)
    assert type(sleep_latencies) == pd.DataFrame
    assert set(
        [
            pywearable.constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
            pywearable.constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
            pywearable.constants.SLEEP_STAGE_REM_STAGE_VALUE,
            pywearable.constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        ]
    ).issubset(sleep_latencies.columns)
    # Check latencies for a given sleep summary
    assert sleep_latencies.loc["x4c64722-64595538-6630", "deep"] == 10
    assert sleep_latencies.loc["x4c64722-64595538-6630", "awake"] == 26
    assert sleep_latencies.loc["x4c64722-64595538-6630", "light"] == 0
    assert sleep_latencies.loc["x4c64722-64595538-6630", "rem"] == 79
    # Check latencies for sleep summary with only one sleep stage
    assert sleep_latencies.loc["x4c64722-645bf6d0-6888", "deep"] == 0
    assert np.isnan(sleep_latencies.loc["x4c64722-645bf6d0-6888", "light"])
    # Check latencies for sleep summary with no sleep stages
    assert np.isnan(sleep_latencies.loc["x4c64722-645ab9b4-55c8", "rem"])
    assert np.isnan(sleep_latencies.loc["x4c64722-645ab9b4-55c8", "light"])


def test_compute_waso(sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame):
    waso = pywearable.sleep._compute_wake_after_sleep_onset(
        sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL), sleep_stages
    )
    print(waso)
    assert type(waso) == pd.Series
    # Check latencies for a given sleep summary
    assert waso.loc[pd.Timestamp(2023, 5, 9)] == 13.0
    # Check latencies for sleep summary with no wake sleep stage
    assert waso.loc[pd.Timestamp(2023, 5, 11)] == 0
    # Check waso for sleep summary with no sleep stages
    assert np.isnan(waso.loc[pd.Timestamp(2023, 5, 10)])


def test_compute_sleep_onset_latency(
    sleep_summary_id_as_idx: pd.DataFrame, sleep_stages: pd.DataFrame
):
    sol = pywearable.sleep._compute_sleep_onset_latency(
        sleep_summary_id_as_idx, sleep_stages
    )
    assert type(sol) == pd.Series
    # Check onset latency
    assert sol.loc["x4c64722-64595538-6630"] == 0.0
    assert sol.loc["x4c64722-645bf6d0-6888"] == 0.0
    assert sol.loc["x4c64722-645d63f8-5c94"] == 10.0
    # Check onset latency for sleep summary with no sleep stages
    assert np.isnan(sol.loc["x4c64722-645ab9b4-55c8"])


def test_compute_sleep_period_time(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame
):
    # Make sure that we have calendar date as index
    sleep_summary = sleep_summary.set_index(pywearable.constants.CALENDAR_DATE_COL)
    # Compute sleep period time
    spt = pywearable.sleep._compute_sleep_period_time(sleep_summary, sleep_stages)
    # Assert is series
    assert type(spt) == pd.Series
    # Assert we have some values
    assert len(spt) > 0
    # Assert we have the same expected values -> we don't have sleep stages for one night
    # so, we don't have spt for that night
    pandas.testing.assert_series_equal(
        spt,
        pd.Series(
            [449.0, 446.0, 395.0],
            index=SLEEP_STAGES_SERIES_INDEX,
            name=pywearable.sleep.SLEEP_STATISTIC_SPT,
        ),
        check_dtype=True,
        check_names=True,
    )


def test_compute_unmeasurable_duration(sleep_summary_id_as_idx: pd.DataFrame):
    unmeasurable_duration = pywearable.sleep._compute_unmeasurable_duration(
        sleep_summary_id_as_idx
    )
    assert type(unmeasurable_duration) == pd.Series
    # Check sleep onset latency for given sleep summaries
    assert unmeasurable_duration.loc["x4c64722-64595538-6630"] == 0.0
    assert unmeasurable_duration.loc["x4c64722-645bf6d0-6888"] == 0.0


def test_compute_stage_count(
    sleep_summary_id_as_idx: pd.DataFrame, sleep_stages: pd.DataFrame
):
    counts = pywearable.sleep._compute_stage_count(
        sleep_summary_id_as_idx, sleep_stages
    )
    assert type(counts) == pd.DataFrame
    assert (
        counts.loc[
            "x4c64722-64595538-6630", pywearable.constants.SLEEP_STAGE_LIGHT_STAGE_VALUE
        ]
        == 11.0
    )
    assert (
        counts.loc[
            "x4c64722-64595538-6630", pywearable.constants.SLEEP_STAGE_DEEP_STAGE_VALUE
        ]
        == 5.0
    )
    assert (
        counts.loc[
            "x4c64722-64595538-6630",
            pywearable.constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
        ]
        == 3.0
    )
    assert np.isnan(
        counts.loc[
            "x4c64722-645ab9b4-55c8", pywearable.constants.SLEEP_STAGE_DEEP_STAGE_VALUE
        ]
    )
    assert (
        counts.loc[
            "x4c64722-645bf6d0-6888", pywearable.constants.SLEEP_STAGE_DEEP_STAGE_VALUE
        ]
        == 1.0
    )
    assert (
        counts.loc[
            "x4c64722-645d63f8-5c94", pywearable.constants.SLEEP_STAGE_LIGHT_STAGE_VALUE
        ]
        == 1.0
    )
    assert (
        counts.loc[
            "x4c64722-645d63f8-5c94", pywearable.constants.SLEEP_STAGE_REM_STAGE_VALUE
        ]
        == 0.0
    )


def test_compute_rem_count(
    sleep_summary_id_as_idx: pd.DataFrame, sleep_stages: pd.DataFrame
):
    counts = pywearable.sleep._compute_rem_count(sleep_summary_id_as_idx, sleep_stages)
    assert type(counts) == pd.Series
    assert counts.loc["x4c64722-64595538-6630"] == 3.0


def test_compute_light_duration(
    sleep_summary_id_as_idx: pd.DataFrame, sleep_stages: pd.DataFrame
):

    # Setup
    light = pywearable.sleep._compute_light_duration(
        sleep_summary=sleep_summary_id_as_idx, sleep_stages=sleep_stages
    )
    # Check and assert
    pd.testing.assert_series_equal(
        light,
        pd.Series(
            [258.0, np.nan, 0.0, 395.0],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_names=False,
        check_dtype=False,
    )


def test_compute_n2_duration(
    sleep_summary_id_as_idx: pd.DataFrame, sleep_stages: pd.DataFrame
):
    # Setup
    n2 = pywearable.sleep._compute_n2_duration(
        sleep_summary=sleep_summary_id_as_idx, sleep_stages=sleep_stages
    )

    # Check and assert
    pd.testing.assert_series_equal(
        n2,
        pd.Series(
            [np.nan, np.nan, np.nan, np.nan],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_names=False,
        check_dtype=False,
    )


def test_compute_deep_duration(
    sleep_summary_id_as_idx: pd.DataFrame, sleep_stages: pd.DataFrame
):
    # Setup
    deep = pywearable.sleep._compute_deep_duration(
        sleep_summary=sleep_summary_id_as_idx, sleep_stages=sleep_stages
    )

    # Check and assert
    pd.testing.assert_series_equal(
        deep,
        pd.Series(
            [65.0, np.nan, 446.0, 0.0],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_names=False,
        check_dtype=False,
    )


def test_compute_rem_duration(
    sleep_summary_id_as_idx: pd.DataFrame, sleep_stages: pd.DataFrame
):
    # Setup
    rem = pywearable.sleep._compute_rem_duration(
        sleep_summary=sleep_summary_id_as_idx, sleep_stages=sleep_stages
    )

    # Check and assert
    pd.testing.assert_series_equal(
        rem,
        pd.Series(
            [113.0, np.nan, 0.0, 0.0],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_names=False,
        check_dtype=False,
    )


def test_compute_cpd_midpoint(sleep_summary_id_as_idx: pd.DataFrame):
    # sleep_summary.calendarDate = sleep_summary.calendarDate.apply(lambda x: pd.to_datetime(x)).dt.date
    sol = pywearable.sleep._compute_cpd_midpoint(
        sleep_summary=sleep_summary_id_as_idx, chronotype=("22:30", "06:00")
    )
    assert type(sol) == pd.Series
    pandas.testing.assert_series_equal(
        sol,
        pd.Series(
            [0.475, 0.683, 0.987, 1.921],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_dtype=False,
        check_names=False,
        check_exact=False,
        atol=0.01,
    )
    sol2 = pywearable.sleep._compute_cpd_midpoint(
        sleep_summary=sleep_summary_id_as_idx, chronotype=("23:30", "07:00")
    )
    assert type(sol2) == pd.Series
    pandas.testing.assert_series_equal(
        sol2,
        pd.Series(
            [1.475, 1.049, 1.781, 1.625],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_dtype=False,
        check_names=False,
        check_exact=False,
        atol=0.01,
    )
    sol3 = pywearable.sleep._compute_cpd_midpoint(
        sleep_summary=sleep_summary_id_as_idx, chronotype=None
    )
    assert type(sol3) == pd.Series
    pandas.testing.assert_series_equal(
        sol3,
        pd.Series(
            [0.492, 0.679, 0.997, 1.912],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_dtype=False,
        check_names=False,
        check_exact=False,
        atol=0.01,
    )


def test_compute_cpd_duration(sleep_summary_id_as_idx: pd.DataFrame):
    # sleep_summary.calendarDate = sleep_summary.calendarDate.apply(lambda x: pd.to_datetime(x)).dt.date
    sol = pywearable.sleep._compute_cpd_duration(
        sleep_summary=sleep_summary_id_as_idx, chronotype=("22:30", "06:00")
    )
    assert type(sol) == pd.Series
    pandas.testing.assert_series_equal(
        sol,
        pd.Series(
            [0.233, 1.822, 1.335, 1.250],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_dtype=False,
        check_names=False,
        check_exact=False,
        atol=0.01,
    )
    sol2 = pywearable.sleep._compute_cpd_duration(
        sleep_summary=sleep_summary_id_as_idx, chronotype=("22:45", "07:00")
    )
    assert type(sol2) == pd.Series
    pandas.testing.assert_series_equal(
        sol2,
        pd.Series(
            [0.983, 2.446, 1.564, 1.871],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_dtype=False,
        check_names=False,
        check_exact=False,
        atol=0.01,
    )
    sol3 = pywearable.sleep._compute_cpd_duration(
        sleep_summary=sleep_summary_id_as_idx, chronotype=None
    )
    assert type(sol3) == pd.Series
    pandas.testing.assert_series_equal(
        sol3,
        pd.Series(
            [0.421, 1.385, 1.457, 0.890],
            index=[
                "x4c64722-64595538-6630",
                "x4c64722-645ab9b4-55c8",
                "x4c64722-645bf6d0-6888",
                "x4c64722-645d63f8-5c94",
            ],
        ),
        check_dtype=False,
        check_names=False,
        check_exact=False,
        atol=0.01,
    )
