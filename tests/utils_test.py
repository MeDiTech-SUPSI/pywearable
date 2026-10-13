import datetime

import numpy as np
import pytest

from pywearable.utils import time as time_utils


def test_clock_conversion_round_trip_preserves_subseconds():
    value = datetime.time(23, 59, 59, 123456)

    seconds = time_utils.clock_seconds(value)

    assert time_utils.clock_time(seconds) == value
    assert time_utils.clock_time(seconds + 24 * 60 * 60) == value


def test_shared_circular_clock_helpers_handle_midnight_and_unwrapping():
    seconds = np.array([23.5 * 3600, 0.5 * 3600])

    center = time_utils.circular_mean_seconds(seconds)

    assert time_utils.clock_time(center) == datetime.time(0, 0)
    assert time_utils.circular_mean_time(["23:30", "00:30"]) == datetime.time(
        0, 0
    )
    assert time_utils.circular_std_minutes(seconds) == pytest.approx(
        30.043003, rel=1e-5
    )
    assert time_utils.unwrap_clock_seconds(3600, 12 * 3600) == 25 * 3600
