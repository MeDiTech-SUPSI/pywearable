import pandas as pd
import pytest

from pywearable import activity, heart_rate, hrv
from tests.activity_test import SampleActivityLoader, _minute_samples
from tests.heart_rate_test import HeartRateLoader, _samples
from tests.hrv_test import RawBBILoader


@pytest.mark.parametrize("family", ["heart_rate", "hrv", "activity"])
@pytest.mark.parametrize("time_reference", ["local", "utc", "auto"])
def test_hierarchical_aware_bounds_match_selected_clock(family, time_reference):
    start = pd.Timestamp("2024-01-01", tz="Europe/Zurich")
    end = pd.Timestamp("2024-01-03", tz="Europe/Zurich")
    primary = "5min" if family == "hrv" else "1h"
    parent = "2h" if time_reference == "utc" else "daily"
    if family == "heart_rate":
        loader = HeartRateLoader(_samples())
        extract = heart_rate.get_heart_rate_statistics
        statistic = "heart_rate_mean"
    elif family == "hrv":
        loader = RawBBILoader()
        extract = hrv.get_hrv_statistics
        statistic = "hrv_rmssd"
    else:
        loader = SampleActivityLoader(_minute_samples("2024-01-01", [1] * 60))
        extract = activity.get_activity_statistics
        statistic = "activity_steps"
    options = dict(
        loader=loader,
        statistic=statistic,
        resolution=[primary, parent],
        aggregation="mean",
        time_reference=time_reference,
    )
    if time_reference == "utc":
        expected_start = pd.Timestamp("2023-12-31 23:00")
        expected_end = pd.Timestamp("2024-01-02 23:00")
    else:
        expected_start = pd.Timestamp("2024-01-01")
        expected_end = pd.Timestamp("2024-01-03")
    expected = extract(start_date=expected_start, end_date=expected_end, **options)
    actual = extract(start_date=start, end_date=end, **options)
    assert not expected.empty
    pd.testing.assert_frame_equal(actual, expected)
