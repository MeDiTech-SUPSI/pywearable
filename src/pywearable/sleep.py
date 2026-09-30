"""
This module contains all the functions related to analysis
of sleep data.
"""

import datetime
import warnings
from typing import Union

import numpy as np
import pandas as pd

from . import hierarchy, schema
from .loader.base import BaseLoader, resolve_user_ids
from .utils import time as time_utils

SLEEP_STATISTIC_TIB = "sleep_TIB"
SLEEP_STATISTIC_TST = "sleep_TST"
SLEEP_STATISTIC_WASO = "sleep_WASO"
SLEEP_STATISTIC_SPT = "sleep_SPT"
SLEEP_STATISTIC_DEEP_DURATION = "sleep_deep"
SLEEP_STATISTIC_LIGHT_DURATION = "sleep_light"
SLEEP_STATISTIC_REM_DURATION = "sleep_REM"
SLEEP_STATISTIC_NREM_DURATION = "sleep_NREM"
SLEEP_STATISTIC_AWAKE_DURATION = "sleep_awake"
SLEEP_STATISTIC_UNMEASURABLE_DURATION = "sleep_unmeasurable"
SLEEP_STATISTIC_SOL = "sleep_SOL"
SLEEP_STATISTIC_DEEP_LATENCY = "sleep_lat_deep"
SLEEP_STATISTIC_LIGHT_LATENCY = "sleep_lat_light"
SLEEP_STATISTIC_REM_LATENCY = "sleep_lat_REM"
SLEEP_STATISTIC_DEEP_PERCENTAGE = "sleep_%_deep"
SLEEP_STATISTIC_LIGHT_PERCENTAGE = "sleep_%_light"
SLEEP_STATISTIC_NREM_PERCENTAGE = "sleep_%_NREM"
SLEEP_STATISTIC_REM_PERCENTAGE = "sleep_%_REM"
SLEEP_STATISTIC_SE = "sleep_SE"
SLEEP_STATISTIC_SME = "sleep_SME"
SLEEP_STATISTIC_SLEEP_SCORE = "sleep_score"
SLEEP_STATISTIC_AWAKE_COUNT = "sleep_count_awake"
SLEEP_STATISTIC_DEEP_COUNT = "sleep_count_deep"
SLEEP_STATISTIC_LIGHT_COUNT = "sleep_count_light"
SLEEP_STATISTIC_REM_COUNT = "sleep_count_REM"
SLEEP_STATISTIC_BEDTIME = "sleep_bedtime"
SLEEP_STATISTIC_SLEEP_ONSET = "sleep_sleep_onset"
SLEEP_STATISTIC_WAKEUP = "sleep_wakeup"
SLEEP_STATISTIC_GETUP = "sleep_getup"
SLEEP_STATISTIC_MIDPOINT = "sleep_midpoint"
SLEEP_STATISTIC_CPD_MIDPOINT = "sleep_cpd_midpoint"
SLEEP_STATISTIC_CPD_DURATION = "sleep_cpd_duration"
SLEEP_STATISTIC_VALIDATION = "sleep_validation"
SLEEP_STATISTIC_SRI = "sleep_SRI"
SLEEP_STATISTIC_GETUP_LATENCY = "sleep_getup_latency"
SLEEP_STATISTIC_AWAKENING_COUNT = "sleep_awakening_count"
SLEEP_STATISTIC_AWAKENING_INDEX = "sleep_awakening_index"
SLEEP_STATISTIC_MEDIAN_AWAKENING_DURATION = "sleep_median_awakening_duration"
SLEEP_STATISTIC_STAGE_TRANSITION_COUNT = "sleep_stage_transition_count"
SLEEP_STATISTIC_STAGE_TRANSITION_INDEX = "sleep_stage_transition_index"
SLEEP_STATISTIC_STAGE_COVERAGE = "sleep_stage_coverage"

SLEEP_TEMPORAL_STATISTICS = {
    SLEEP_STATISTIC_BEDTIME,
    SLEEP_STATISTIC_SLEEP_ONSET,
    SLEEP_STATISTIC_WAKEUP,
    SLEEP_STATISTIC_GETUP,
    SLEEP_STATISTIC_MIDPOINT,
}
SLEEP_NIGHTLY_RESOLUTIONS = {"night", "nightly", "natural"}


def _require_dataframe(value, name):
    """Return ``value`` after validating that it is a DataFrame."""
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    return value


def _mean_clock_time(values) -> str:
    """Return the circular mean of clock-like values as ``HH:MM``."""
    seconds = [
        time_utils.clock_seconds(value) for value in values if pd.notna(value)
    ]
    return time_utils.clock_time(
        time_utils.circular_mean_seconds(seconds)
    ).strftime("%H:%M")


def get_time_in_bed(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly time in bed.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_TIB,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_period_time(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly sleep period time.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SPT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_total_sleep_time(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return nightly total sleep time.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_TST,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_efficiency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return nightly sleep efficiency.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SE,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_maintenance_efficiency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return nightly sleep maintenance efficiency.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SME,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_deep_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly deep latency.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistics(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_LATENCY,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_light_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    return get_sleep_statistics(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_LATENCY,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_rem_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly rem latency.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_LATENCY,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_light_sleep_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly light sleep duration.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_DURATION,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_deep_sleep_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly deep sleep duration.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_DURATION,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_rem_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly rem duration.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_DURATION,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_nrem_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly nrem duration.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_NREM_DURATION,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_awake_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly awake duration.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_AWAKE_DURATION,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_unmeasurable_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly unmeasurable duration.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_UNMEASURABLE_DURATION,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_light_sleep_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_deep_sleep_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_rem_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_nrem_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_NREM_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_wake_after_sleep_onset(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_WASO,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_onset_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SOL,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_score(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SLEEP_SCORE,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_timestamps(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
) -> pd.DataFrame:
    """Return the main nightly sleep timestamps.

    ``sleep_bedtime`` and ``sleep_getup`` delimit the provider-recorded sleep
    session. ``sleep_sleep_onset`` and ``sleep_wakeup`` delimit observed sleep
    within that session.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which sleep timestamps must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.

    Returns
    -------
    :class:`pandas.DataFrame`
        Bedtime, sleep-onset, wake-up, and get-up columns, indexed by
        ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistic=[
            SLEEP_STATISTIC_BEDTIME,
            SLEEP_STATISTIC_SLEEP_ONSET,
            SLEEP_STATISTIC_WAKEUP,
            SLEEP_STATISTIC_GETUP,
        ],
    )


def get_bedtime(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return nightly bedtime.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_BEDTIME,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_wakeup_time(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly wakeup time.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_WAKEUP,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_midpoint(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly sleep midpoint.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_MIDPOINT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_awake_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return nightly awake count.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_AWAKE_COUNT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_light_sleep_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly light sleep count.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_COUNT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_deep_sleep_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly deep sleep count.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_COUNT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_rem_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly rem count.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_COUNT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_cpd_midpoint(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    chronotype_dict: dict = {},
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly cpd midpoint.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_CPD_MIDPOINT,
        start_date=start_date,
        end_date=end_date,
        chronotype=chronotype_dict,
        loader_kwargs=loader_kwargs,
    )


SLEEP_WAKE_TRAJECTORY_COLUMNS = [
    "calendar_date",
    "state",
    "sleep_stage",
    "state_available",
    "state_coverage",
    "state_source",
    "session_id",
    "main_sleep",
    "session_type",
]


def _fixed_sleep_resolution(resolution: str) -> pd.Timedelta:
    try:
        value = pd.Timedelta(resolution)
    except (TypeError, ValueError) as error:
        raise ValueError("resolution must be a fixed pandas timedelta") from error
    if value <= pd.Timedelta(0):
        raise ValueError("resolution must be positive")
    return value


def _empty_sleep_wake_trajectory() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [[], []], names=[schema.USER_COL, schema.LOCAL_TIME_COL]
    )
    return pd.DataFrame(columns=SLEEP_WAKE_TRAJECTORY_COLUMNS, index=index)


def _sleep_trajectory_bin_slice(
    grid_start: pd.Timestamp,
    resolution: pd.Timedelta,
    size: int,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> tuple[int, int]:
    start = max(interval_start, grid_start)
    grid_end = grid_start + size * resolution
    end = min(interval_end, grid_end)
    if end <= start:
        return 0, 0
    first = max(0, int(np.floor((start - grid_start) / resolution)))
    last = min(size, int(np.ceil((end - grid_start) / resolution)))
    return first, last


def _get_user_sleep_wake_trajectory(
    loader: BaseLoader,
    user_id: str,
    start_date=None,
    end_date=None,
    resolution: str = "1min",
    include_naps: bool = True,
) -> pd.DataFrame:
    resolution_delta = _fixed_sleep_resolution(resolution)
    requested_start = None if start_date is None else pd.Timestamp(start_date)
    requested_end = None if end_date is None else pd.Timestamp(end_date)
    if (
        requested_start is not None
        and requested_end is not None
        and requested_end <= requested_start
    ):
        raise ValueError("end_date must be after start_date")

    load_start = (
        None
        if requested_start is None
        else requested_start - pd.Timedelta(days=1)
    )
    load_end = (
        None if requested_end is None else requested_end + pd.Timedelta(days=1)
    )
    sessions = schema.prepare_sleep_sessions(
        loader.load_sleep_sessions(user_id, load_start, load_end)
    )
    stages = schema.prepare_sleep_stages(
        loader.load_sleep_session_stages(user_id, load_start, load_end)
    )

    if not include_naps and not sessions.empty:
        main_ids = set(sessions.loc[sessions["main_sleep"], "id"])
        sessions = sessions[sessions["id"].isin(main_ids)].reset_index(drop=True)
        stages = stages[
            stages[schema.SLEEP_SUMMARY_ID_COL].isin(main_ids)
        ].reset_index(drop=True)

    available_starts = []
    available_ends = []
    if not sessions.empty:
        available_starts.append(sessions[schema.LOCAL_TIME_COL].min())
        available_ends.append(sessions[schema.LOCAL_END_TIME_COL].max())
    if not stages.empty:
        available_starts.append(stages[schema.LOCAL_TIME_COL].min())
        available_ends.append(
            (
                stages[schema.LOCAL_TIME_COL]
                + pd.to_timedelta(
                    stages[schema.DURATION_IN_MS_COL], unit="ms"
                )
            ).max()
        )
    if requested_start is None:
        if not available_starts:
            return _empty_sleep_wake_trajectory()
        requested_start = min(available_starts).normalize()
    if requested_end is None:
        if not available_ends:
            return _empty_sleep_wake_trajectory()
        requested_end = max(available_ends).normalize() + pd.Timedelta(days=1)

    grid = pd.date_range(
        requested_start, requested_end, freq=resolution_delta, inclusive="left"
    )
    size = len(grid)
    if size == 0:
        return _empty_sleep_wake_trajectory()

    bin_ms = resolution_delta.total_seconds() * 1000
    sleep_ms = np.zeros(size, dtype=float)
    awake_ms = np.zeros(size, dtype=float)
    inferred_awake = np.zeros(size, dtype=bool)
    staged_ms = np.zeros(size, dtype=float)
    stage_overlap: dict[str, np.ndarray] = {}
    session_for_bin = np.full(size, None, dtype=object)

    # Infer wake only when two main sleeps belong to consecutive calendar
    # dates. Missing nights therefore leave the intervening state unknown.
    main_sessions = sessions[sessions["main_sleep"]].sort_values(
        schema.CALENDAR_DATE_COL, kind="stable"
    )
    for current, following in zip(
        main_sessions.itertuples(index=False),
        main_sessions.iloc[1:].itertuples(index=False),
    ):
        if following.calendar_date - current.calendar_date != pd.Timedelta(days=1):
            continue
        if following.local_time < current.local_end_time:
            continue
        first, last = _sleep_trajectory_bin_slice(
            requested_start,
            resolution_delta,
            size,
            current.local_end_time,
            following.local_time,
        )
        for index in range(first, last):
            bin_start = grid[index]
            bin_end = bin_start + resolution_delta
            overlap = min(bin_end, following.local_time) - max(
                bin_start, current.local_end_time
            )
            if overlap > pd.Timedelta(0):
                awake_ms[index] += overlap.total_seconds() * 1000
                inferred_awake[index] = True

    # A recorded session interrupts inferred wake. Preserve its identity even
    # when no stage data are available, in which case its state remains missing.
    for session in sessions.itertuples(index=False):
        first, last = _sleep_trajectory_bin_slice(
            requested_start,
            resolution_delta,
            size,
            session.local_time,
            session.local_end_time,
        )
        for index in range(first, last):
            bin_start = grid[index]
            bin_end = bin_start + resolution_delta
            overlap = min(bin_end, session.local_end_time) - max(
                bin_start, session.local_time
            )
            if overlap <= pd.Timedelta(0):
                continue
            session_for_bin[index] = session.id
            awake_ms[index] = max(
                0, awake_ms[index] - overlap.total_seconds() * 1000
            )
            inferred_awake[index] = awake_ms[index] > 0

    session_lookup = sessions.set_index("id") if not sessions.empty else sessions
    for stage in stages.itertuples(index=False):
        if hasattr(stage, "local_time_uncertain") and stage.local_time_uncertain:
            continue
        stage_start = getattr(stage, schema.LOCAL_TIME_COL)
        stage_end = stage_start + pd.to_timedelta(
            getattr(stage, schema.DURATION_IN_MS_COL), unit="ms"
        )
        session_id = getattr(stage, schema.SLEEP_SUMMARY_ID_COL)
        if not sessions.empty and session_id in session_lookup.index:
            session = session_lookup.loc[session_id]
            stage_start = max(stage_start, session[schema.LOCAL_TIME_COL])
            stage_end = min(stage_end, session[schema.LOCAL_END_TIME_COL])
            if stage_end <= stage_start:
                continue
        stage_type = getattr(stage, schema.SLEEP_STAGE_SLEEP_TYPE_COL)
        if stage_type == schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE:
            continue
        state = (
            "awake"
            if stage_type == schema.SLEEP_STAGE_AWAKE_STAGE_VALUE
            else "sleep"
        )
        if stage_type not in stage_overlap:
            stage_overlap[stage_type] = np.zeros(size, dtype=float)
        first, last = _sleep_trajectory_bin_slice(
            requested_start, resolution_delta, size, stage_start, stage_end
        )
        for index in range(first, last):
            bin_start = grid[index]
            bin_end = bin_start + resolution_delta
            overlap = min(bin_end, stage_end) - max(bin_start, stage_start)
            if overlap <= pd.Timedelta(0):
                continue
            overlap_ms = overlap.total_seconds() * 1000
            if state == "sleep":
                sleep_ms[index] += overlap_ms
            else:
                awake_ms[index] += overlap_ms
            staged_ms[index] += overlap_ms
            stage_overlap[stage_type][index] += overlap_ms
            session_for_bin[index] = session_id

    state = np.full(size, None, dtype=object)
    has_state = (sleep_ms + awake_ms) > 0
    state[has_state & (sleep_ms >= awake_ms)] = "sleep"
    state[has_state & (awake_ms > sleep_ms)] = "awake"
    state_coverage = np.minimum(100, (sleep_ms + awake_ms) / bin_ms * 100)

    source = np.full(size, None, dtype=object)
    source[has_state & inferred_awake & (staged_ms == 0)] = "between_main_sleeps"
    source[has_state & ~inferred_awake & (staged_ms > 0)] = "stage"
    source[has_state & inferred_awake & (staged_ms > 0)] = "mixed"

    detailed_stage = np.full(size, None, dtype=object)
    if stage_overlap:
        stage_names = list(stage_overlap)
        overlap_matrix = np.vstack([stage_overlap[name] for name in stage_names])
        winners = overlap_matrix.argmax(axis=0)
        has_stage = overlap_matrix.max(axis=0) > 0
        detailed_stage[has_stage] = np.asarray(stage_names, dtype=object)[
            winners[has_stage]
        ]
    detailed_stage[(state == "awake") & pd.isna(detailed_stage)] = "awake"

    main_sleep = np.full(size, pd.NA, dtype=object)
    session_type = np.full(size, None, dtype=object)
    if not sessions.empty:
        for index, session_id in enumerate(session_for_bin):
            if session_id is None or session_id not in session_lookup.index:
                continue
            session = session_lookup.loc[session_id]
            main_sleep[index] = bool(session["main_sleep"])
            session_type[index] = session["type"]

    result = pd.DataFrame(
        {
            schema.CALENDAR_DATE_COL: grid.normalize(),
            "state": pd.array(state, dtype="string"),
            "sleep_stage": pd.array(detailed_stage, dtype="string"),
            "state_available": pd.notna(state),
            "state_coverage": state_coverage,
            "state_source": pd.array(source, dtype="string"),
            "session_id": pd.array(session_for_bin, dtype="string"),
            "main_sleep": pd.array(main_sleep, dtype="boolean"),
            "session_type": pd.array(session_type, dtype="string"),
        },
        index=grid,
    )
    result.index.name = schema.LOCAL_TIME_COL
    return pd.concat(
        {user_id: result}, names=[schema.USER_COL, schema.LOCAL_TIME_COL]
    )


def get_sleep_regularity_index(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    resolution: str = "1min",
    include_naps: bool = True,
) -> pd.Series:
    """Return optional daily SRI values from the full-day trajectory."""
    return get_sleep_regularity(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        include_naps=include_naps,
    )[SLEEP_STATISTIC_SRI]


def get_sleep_wake_trajectory(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    resolution: str = "1min",
    include_naps: bool = True,
) -> pd.DataFrame:
    """Return a regular local-time sleep/wake trajectory.

    Wake between consecutive main-sleep sessions is inferred. Recorded
    sessions interrupt that inference and are classified only where stage data
    are available. Missing nights, unmeasurable stages, and other gaps remain
    missing. Non-main sessions are included when the loader exposes them and
    ``include_naps`` is true.
    """
    users = resolve_user_ids(loader, user_id)
    trajectories = [
        _get_user_sleep_wake_trajectory(
            loader,
            user,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
            include_naps=include_naps,
        )
        for user in users
    ]
    trajectories = [trajectory for trajectory in trajectories if not trajectory.empty]
    if not trajectories:
        return _empty_sleep_wake_trajectory()
    return pd.concat(trajectories).sort_index()


def get_sleep_regularity(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    resolution: str = "1min",
    include_naps: bool = True,
) -> pd.DataFrame:
    """Return daily SRI together with comparable-epoch coverage."""
    resolution_delta = _fixed_sleep_resolution(resolution)
    periods_per_day = pd.Timedelta(days=1) / resolution_delta
    if periods_per_day != int(periods_per_day):
        raise ValueError("resolution must divide evenly into 24 hours for SRI")
    periods_per_day = int(periods_per_day)

    trajectory_start = (
        None
        if start_date is None
        else pd.Timestamp(start_date).normalize() - pd.Timedelta(days=1)
    )
    trajectory_end = (
        None if end_date is None else pd.Timestamp(end_date).normalize()
    )
    trajectory = get_sleep_wake_trajectory(
        loader,
        user_id=user_id,
        start_date=trajectory_start,
        end_date=trajectory_end,
        resolution=resolution,
        include_naps=include_naps,
    )
    records = []
    for user, user_trajectory in trajectory.groupby(level=schema.USER_COL):
        states = user_trajectory.droplevel(schema.USER_COL)["state"]
        if states.empty:
            continue
        first_date = states.index.min().normalize()
        last_date = states.index.max().normalize()
        requested_first = (
            first_date + pd.Timedelta(days=1)
            if start_date is None
            else pd.Timestamp(start_date).normalize()
        )
        requested_last = (
            last_date + pd.Timedelta(days=1)
            if end_date is None
            else pd.Timestamp(end_date).normalize()
        )
        for day in pd.date_range(
            requested_first, requested_last, freq="D", inclusive="left"
        ):
            current_index = pd.date_range(
                day,
                day + pd.Timedelta(days=1),
                freq=resolution_delta,
                inclusive="left",
            )
            current = states.reindex(current_index)
            previous = states.reindex(current_index - pd.Timedelta(days=1))
            previous.index = current_index
            comparable = current.notna() & previous.notna()
            comparable_count = int(comparable.sum())
            sri = np.nan
            if comparable_count:
                agreement = current[comparable].eq(previous[comparable]).mean()
                sri = -100 + 200 * agreement
            records.append(
                {
                    schema.USER_COL: user,
                    schema.CALENDAR_DATE_COL: day,
                    SLEEP_STATISTIC_SRI: sri,
                    "sleep_sri_coverage": comparable_count / periods_per_day * 100,
                }
            )
    if not records:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=[schema.USER_COL, schema.CALENDAR_DATE_COL]
        )
        return pd.DataFrame(
            columns=[SLEEP_STATISTIC_SRI, "sleep_sri_coverage"], index=index
        )
    return (
        pd.DataFrame(records)
        .set_index([schema.USER_COL, schema.CALENDAR_DATE_COL])
        .sort_index()
    )


def get_awakening_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return the number of internal awake bouts for each night."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_AWAKENING_COUNT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_awakening_index(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return internal awakenings per hour of total sleep time."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_AWAKENING_INDEX,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_median_awakening_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return the median duration in minutes of internal awake bouts."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_MEDIAN_AWAKENING_DURATION,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_stage_transition_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return direct transitions between detailed sleep stages."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_STAGE_TRANSITION_COUNT,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_stage_transition_index(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return direct detailed-stage transitions per hour of total sleep time."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_STAGE_TRANSITION_INDEX,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_stage_coverage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return the percentage of time in bed with a measurable stage label."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_STAGE_COVERAGE,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_sleep_onset(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return the start of the first observed sleep stage for each night."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SLEEP_ONSET,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_getup_time(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return the provider-recorded sleep-session end for each night."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_GETUP,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_getup_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: dict = {},
) -> pd.DataFrame:
    """Return minutes from final observed sleep offset to session end."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_GETUP_LATENCY,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
    )


def get_cpd_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    chronotype_dict: dict = {},
    loader_kwargs={},
) -> pd.DataFrame:
    """Return nightly cpd duration.

    The result is a :class:`pandas.DataFrame` with the feature and its
    ``__coverage`` column, indexed by ``user_id`` and ``calendar_date``.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_CPD_DURATION,
        start_date=start_date,
        end_date=end_date,
        chronotype=chronotype_dict,
        loader_kwargs=loader_kwargs,
    )


def _compute_sleep_score(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Retrieves sleep score from a sleep summary

    This function retrieves, if available, the sleep scores
    given sleep summary data.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data.

    Returns
    -------
    pd.Series
        Sleep score data.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pandas.DataFrame` .
    """
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    if schema.SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL in sleep_summary.columns:
        sleep_score = sleep_summary[schema.SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL]
    else:
        sleep_score = pd.Series(np.nan, index=sleep_summary.index, dtype=float)
    return sleep_score


def _compute_time_in_bed(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute Time in Bed (TIB) metric for sleep data.

    This function computes the TIB as the total duration of the sleep,
    considering both awake and unmeasurable sleep stages.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        TIB data.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pd.DataFrame`.
    """
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    # TIB = SLEEP_DURATION + AWAKE_DURATION in minutes
    if (schema.DURATION_IN_MS_COL in sleep_summary.columns) and (
        schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL in sleep_summary.columns
    ):
        tib = (
            sleep_summary[schema.DURATION_IN_MS_COL]
            + sleep_summary[schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL]
        ) / (
            1000 * 60
        )  # convert from ms to seconds, and then to minutes
    else:
        tib = pd.Series()
    return tib


def _compute_total_sleep_time(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute Total Sleep Time (TST) metric for sleep data.

    This function computes the TST as the total duration of the N1, N2, N3,
    and REM sleep stages within sleep period time (SPT).

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        TST data.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pd.DataFrame`.
    """
    if not isinstance(sleep_summary, pd.DataFrame):
        raise ValueError(
            f"sleep_summary must be a pd.DataFrame. {type(sleep_summary)} is not a valid type."
        )
    # The shared summary duration includes classified and unmeasurable sleep.
    # Subtract the latter to obtain total sleep time, then fall back to the
    # classified stage totals if a loader does not provide both summary fields.
    summary_tst = pd.Series(np.nan, index=sleep_summary.index, dtype=float)
    if (
        schema.DURATION_IN_MS_COL in sleep_summary.columns
        and schema.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL
        in sleep_summary.columns
    ):
        summary_tst = (
            sleep_summary[schema.DURATION_IN_MS_COL]
            - sleep_summary[
                schema.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL
            ]
        )

    stage_tst = pd.Series(np.nan, index=sleep_summary.index, dtype=float)
    if (
        (schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL in sleep_summary.columns)
        and (
            schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
        and (
            schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
    ):
        stage_tst = sleep_summary[
            [
                schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
                schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
                schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
            ]
        ].sum(axis=1, min_count=3)
    missing_summary = summary_tst.isna()
    summary_tst.loc[missing_summary] = stage_tst.loc[missing_summary]
    return summary_tst / (1000 * 60)


def _compute_sleep_efficiency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Compute Sleep Efficiency (SE) metric for sleep data.

    This function computes the SE as the ratio between Total Sleep Time (TST)
    and Time in Bed (TIB).

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        SE data.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pd.DataFrame`.
    """
    if not isinstance(sleep_summary, pd.DataFrame):
        raise ValueError(
            f"sleep_summary must be a pd.DataFrame. {type(sleep_summary)} is not a valid type."
        )
    # SE = SLEEP_DURATION / (SLEEP_DURATION + AWAKE_DURATION) * 100
    tst = _compute_total_sleep_time(
        sleep_summary=sleep_summary, sleep_stages=sleep_stages
    )
    tib = _compute_time_in_bed(sleep_summary=sleep_summary, sleep_stages=sleep_stages)
    se = tst / tib * 100
    return se


def _compute_sleep_maintenance_efficiency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Compute Sleep Maintenance efficiency (SME) metric for sleep data.

    This function computes the SME as the ratio between
    the sum of all non-wake sleep stages (N1+N2+N2+REM)
    and Sleep Period Time (SPT).

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages data.

    Returns
    -------
    :class:`pd.Series`
        SME data.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pd.DataFrame`.
    """
    if not isinstance(sleep_summary, pd.DataFrame):
        raise ValueError(
            f"sleep_summary must be a pd.DataFrame. {type(sleep_summary)} is not a valid type."
        )
    # SME combines two quantities derived from the same stage trajectory.
    # Provider summaries and epoch grids can differ slightly in resolution;
    # mixing summary TST with stage-derived SPT can otherwise exceed 100%.
    filtered_sleep_stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                sleep_summary[schema.SLEEP_SUMMARY_ID_COL].unique()
            )
        ].reset_index(drop=True)
    )
    known_sleep = filtered_sleep_stages[
        ~filtered_sleep_stages[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(
            [
                schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
            ]
        )
    ]
    stage_tst_by_id = (
        known_sleep.groupby(schema.SLEEP_SUMMARY_ID_COL)[
            schema.DURATION_IN_MS_COL
        ].sum()
        / 60_000
    )
    tst = sleep_summary[schema.SLEEP_SUMMARY_ID_COL].map(stage_tst_by_id)
    spt = _compute_sleep_period_time(
        sleep_summary=sleep_summary, sleep_stages=sleep_stages
    )
    sme = tst / spt * 100
    return sme


def _compute_sleep_stage_percentage(
    sleep_summary: pd.DataFrame, sleep_stage: str, **kwargs
) -> pd.Series:
    """Compute the percentage of time spent in a sleep stage.

    This function computes the percentage of time in a sleep stage
    as the ratio between the time spent in it and the
    total sleep time.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.
    sleep_stage : :class:`str`
        Name of the sleep stage of interest.

    Returns
    -------
    :class:`pd.Series`
        Percentage of time spent in `sleep_stage` per night.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pd.DataFrame`.
    ValueError
        if `sleep_stage` isn't one of "N1","N3","REM","NREM".
    """
    if not isinstance(sleep_summary, pd.DataFrame):
        raise ValueError(
            f"sleep_summary must be a pd.DataFrame. {type(sleep_summary)} is not a valid type."
        )
    if (
        (schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL in sleep_summary.columns)
        and (
            schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
        and (
            schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
    ):
        tot_duration = sleep_summary[
            [
                schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
                schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
                schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
            ]
        ].sum(axis=1, min_count=3)
        if sleep_stage == "NREM":
            perc = (
                (
                    tot_duration
                    - sleep_summary[
                        schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
                    ]
                )
                / tot_duration
                * 100
            )
        else:
            if sleep_stage == "light":
                col = schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            elif sleep_stage == "deep":
                col = schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL
            elif sleep_stage == "REM":
                col = schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            else:
                raise ValueError(
                    f"{sleep_stage} is not a valid value. Select among [light, deep, REM, NREM]"
                )
            perc = (sleep_summary[col]) / tot_duration * 100
    else:
        perc = pd.Series()
    return perc


def _compute_light_percentage(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the percentage of time spent in the light sleep stage.

    This function computes the percentage of time in light
    as the ratio between the time spent in it and the sum of the
    time spent sleeping.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Percentage of time spent in light sleep per night.
    """
    return _compute_sleep_stage_percentage(sleep_summary, "light")


def _compute_deep_percentage(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the percentage of time spent in the deep sleep stage.

    This function computes the percentage of time in deep
    as the ratio between the time spent in it and the sum of the
    time spent sleeping.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Percentage of time spent in deep sleep per night.
    """
    return _compute_sleep_stage_percentage(sleep_summary, "deep")


def _compute_rem_percentage(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the percentage of time spent in the REM sleep stage.

    This function computes the percentage of time in REM
    as the ratio between the time spent in it and the sum of the
    time spent sleeping.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Percentage of time spent in REM per night.
    """
    return _compute_sleep_stage_percentage(sleep_summary, "REM")


def _compute_nrem_percentage(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the percentage of time spent in the NREM sleep stage.

    This function computes the percentage of time in NREM
    as the ratio between the time spent in it and the sum of the
    time spent sleeping.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Percentage of time spent in NREM per night.
    """
    return _compute_sleep_stage_percentage(sleep_summary, "NREM")


def _compute_sleep_stage_duration(
    sleep_summary: pd.DataFrame, stage: str, **kwargs
) -> pd.Series:
    """Compute the duration of a specified sleep stage.

    This function computes the time spent (in minutes)
    in a specific sleep stage.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.
    stage : :class:`str`
        Name of a sleep stage.

    Returns
    -------
    :class:`pd.Series`
        Amount of minutes spent in `sleep_stage` per night.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pd.DataFrame`.
    ValueError
        if `sleep_stage` isn't one of "N1","N2", "N3","REM","NREM","AWAKE","UNMEASURABLE".
    """
    if not isinstance(sleep_summary, pd.DataFrame):
        raise ValueError(
            f"sleep_summary must be a pd.DataFrame. {type(sleep_summary)} is not a valid type."
        )
    if stage == "NREM":
        dur = (
            sleep_summary[schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL]
            + sleep_summary[schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL]
        ) / (1000 * 60)
    else:
        if stage == "light":
            col = schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
        elif stage == "deep":
            col = schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL
        elif stage == "REM":
            col = schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
        elif stage == "awake":
            col = schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL
        elif stage == "unmeasurable":
            col = schema.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL
        else:
            raise ValueError(
                f"{stage} is not a valid value. Select among [awake, deep, light REM, NREM, unmeasurable]"
            )
        dur = (sleep_summary[col]) / (1000 * 60)
    return dur


def _compute_light_duration(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the duration of the light sleep stage.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Amount of minutes spent in light sleep per night.
    """
    return _compute_sleep_stage_duration(sleep_summary, "light")


def _compute_deep_duration(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the duration of the deep sleep stage.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Amount of minutes spent in deep sleep per night.
    """
    return _compute_sleep_stage_duration(sleep_summary, "deep")


def _compute_rem_duration(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the duration of the REM sleep stage.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Amount of minutes spent in REM per night.
    """
    return _compute_sleep_stage_duration(sleep_summary, "REM")


def _compute_nrem_duration(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the duration of the NREM sleep stages.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Amount of minutes spent in NREM per night.
    """
    return _compute_sleep_stage_duration(sleep_summary, "NREM")


def _compute_awake_duration(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the amount of time spent awake during the night

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Amount of minutes spent in awake per night.
    """
    return _compute_sleep_stage_duration(sleep_summary, "awake")


def _compute_unmeasurable_duration(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute the duration of unmeasurable sleep in a night.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pd.Series`
        Amount of minutes where sleep was unmeasurable per night.
    """
    return _compute_sleep_stage_duration(sleep_summary, "unmeasurable")


def _compute_sleep_period_time(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Compute sleep period time.

    This function computes sleep period time, that is the
    duration from first to last period of sleep.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of ``sleep_summary`` must be the unique ``sleep_summary_id``.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        Series with sleep period time as values, and sleep summary ids as index.
    """
    # Check inputs
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    sleep_stages = _require_dataframe(sleep_stages, "sleep_stages")

    if len(sleep_summary) == 0:
        return pd.Series()
    if len(sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)

    filtered_sleep_stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                sleep_summary[schema.SLEEP_SUMMARY_ID_COL].unique()
            )
        ].reset_index(drop=True)
    )
    # Unmeasurable intervals may occur inside a sleep period, but cannot
    # establish sleep onset or final sleep offset themselves.
    sleeping_stages = filtered_sleep_stages[
        ~filtered_sleep_stages[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(
            [
                schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
            ]
        )
    ].reset_index(drop=True)
    durations = {}
    for summary_id, group in sleeping_stages.groupby(
        schema.SLEEP_SUMMARY_ID_COL, sort=False
    ):
        time_col = schema.LOCAL_TIME_COL
        if (
            schema.UTC_TIME_COL in group.columns
            and group[schema.UTC_TIME_COL].notna().all()
        ):
            time_col = schema.UTC_TIME_COL
        start = group[time_col]
        end = start + pd.to_timedelta(
            group[schema.DURATION_IN_MS_COL], unit="ms"
        )
        durations[summary_id] = (end.max() - start.min()).total_seconds() / 60

    spt = sleep_summary[schema.SLEEP_SUMMARY_ID_COL].map(durations).dropna()
    spt.name = SLEEP_STATISTIC_SPT
    return spt


def _compute_sleep_onset_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Compute sleep onset latency.

    This function computes the latency to the
    first sleep stage.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of ``sleep_summary`` must be the unique ``sleep_summary_id``.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        Series with sleep onset latency.
    """
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    sleep_stages = _require_dataframe(sleep_stages, "sleep_stages")
    if len(sleep_summary) == 0:
        return pd.Series()
    if len(sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    # Remove awake latency
    if schema.SLEEP_STAGE_AWAKE_STAGE_VALUE in latencies.columns:
        latencies = latencies.drop([schema.SLEEP_STAGE_AWAKE_STAGE_VALUE], axis=1)
    # Remove unmeasurable latency in case it is there
    if schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE in latencies.columns:
        latencies = latencies.drop(
            [schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE], axis=1
        )
    # Return minimum latency
    return latencies.min(axis=1)


def _compute_wake_after_sleep_onset(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Compute wake after sleep onset.

    This function returns the total number of minutes
    spent awake after the sleep onset.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of ``sleep_summary`` must be the unique ``sleep_summary_id``.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        Series with wake after sleep onset data.
    """
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    sleep_stages = _require_dataframe(sleep_stages, "sleep_stages")
    if len(sleep_summary) == 0:
        return pd.Series()
    if len(sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)

    filtered_sleep_stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                sleep_summary[schema.SLEEP_SUMMARY_ID_COL].unique()
            )
        ].reset_index(drop=True)
    )
    waso_by_id = {}
    for summary_id, group in filtered_sleep_stages.groupby(
        schema.SLEEP_SUMMARY_ID_COL, sort=False
    ):
        time_col = schema.LOCAL_TIME_COL
        if (
            schema.UTC_TIME_COL in group.columns
            and group[schema.UTC_TIME_COL].notna().all()
        ):
            time_col = schema.UTC_TIME_COL
        known_sleep = group[
            ~group[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(
                [
                    schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                    schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
                ]
            )
        ]
        if known_sleep.empty:
            continue
        sleep_onset = known_sleep[time_col].min()
        sleep_offset = (
            known_sleep[time_col]
            + pd.to_timedelta(
                known_sleep[schema.DURATION_IN_MS_COL], unit="ms"
            )
        ).max()
        awake = group[
            group[schema.SLEEP_STAGE_SLEEP_TYPE_COL]
            == schema.SLEEP_STAGE_AWAKE_STAGE_VALUE
        ]
        awake_start = awake[time_col].clip(lower=sleep_onset)
        awake_end = (
            awake[time_col]
            + pd.to_timedelta(
                awake[schema.DURATION_IN_MS_COL], unit="ms"
            )
        ).clip(upper=sleep_offset)
        waso_by_id[summary_id] = (
            (awake_end - awake_start).dt.total_seconds().clip(lower=0).sum() / 60
        )

    waso = sleep_summary[schema.SLEEP_SUMMARY_ID_COL].map(waso_by_id)
    waso.name = SLEEP_STATISTIC_WASO
    return waso


def _compute_light_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Compute light sleep latency from sleep data.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of ``sleep_summary`` must be the unique ``sleep_summary_id``.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.DataFrame`
        A DataFrame with N1 Latency as value, and sleep summary ids as indexes.
    """
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    return pd.Series(latencies[schema.SLEEP_STAGE_LIGHT_STAGE_VALUE])


def _compute_deep_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Compute N3 latency from sleep data.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of ``sleep_summary`` must be the unique ``sleep_summary_id``.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        A series with N3 Latency as value, and sleep summary ids as indexes.
    """
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    return pd.Series(latencies[schema.SLEEP_STAGE_DEEP_STAGE_VALUE])


def _compute_rem_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Computes REM Latency from sleep data.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of ``sleep_summary`` must be the unique ``sleep_summary_id``.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        A series with REM Latency as value, and sleep summary ids as index.
    """
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    return pd.Series(latencies[schema.SLEEP_STAGE_REM_STAGE_VALUE])


def _compute_latencies(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Compute sleep stages latencies.

    This function computes latencies for each sleep stage
    contained in ``sleep_stages`.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of ``sleep_summary`` must be the unique ``sleep_summary_id``.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    pd.DataFrame
        The returned :class:`pd.DataFrame` has as index the
        values of the ids of the sleep summary to which they
        refer to, and one column for each sleep stage:
        awake, light, deep, and rem.

    Raises
    ------
    ValueError
        If `sleep_summary` or `sleep_stages` are not of type :class:`pd.DataFrame`
    """
    # Check input
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    sleep_stages = _require_dataframe(sleep_stages, "sleep_stages")
    # Check lenght of input
    if len(sleep_summary) == 0:
        return pd.DataFrame(
            columns=[
                schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
                schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
                schema.SLEEP_STAGE_REM_STAGE_VALUE,
            ]
        )
    if len(sleep_stages) == 0:
        return pd.DataFrame(
            index=sleep_summary.index,
            columns=[
                schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
                schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
                schema.SLEEP_STAGE_REM_STAGE_VALUE,
            ],
        )
    filtered_sleep_stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                sleep_summary[schema.SLEEP_SUMMARY_ID_COL].unique()
            )
        ].reset_index(drop=True)
    )
    summary_by_id = sleep_summary.set_index(schema.SLEEP_SUMMARY_ID_COL)
    latency_rows = {}
    for summary_id, group in filtered_sleep_stages.groupby(
        schema.SLEEP_SUMMARY_ID_COL, sort=False
    ):
        summary = summary_by_id.loc[summary_id]
        time_col = schema.LOCAL_TIME_COL
        if (
            schema.UTC_TIME_COL in group.columns
            and schema.UTC_TIME_COL in summary.index
            and group[schema.UTC_TIME_COL].notna().all()
            and pd.notna(summary[schema.UTC_TIME_COL])
        ):
            time_col = schema.UTC_TIME_COL
        first_stages = group.groupby(schema.SLEEP_STAGE_SLEEP_TYPE_COL)[
            time_col
        ].min()
        latency_rows[summary_id] = (
            first_stages - summary[time_col]
        ).dt.total_seconds() / 60
    latencies = pd.DataFrame.from_dict(latency_rows, orient="index")
    # Add columns if they are not there
    for stage in [
        schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
        schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
        schema.SLEEP_STAGE_REM_STAGE_VALUE,
    ]:
        if stage not in latencies.columns:
            latencies[stage] = np.nan

    latencies = latencies.rename_axis(None, axis=1)
    latencies = sleep_summary[[schema.SLEEP_SUMMARY_ID_COL]].join(
        latencies, on=schema.SLEEP_SUMMARY_ID_COL
    )
    return latencies.drop(schema.SLEEP_SUMMARY_ID_COL, axis=1)


def _compute_awake_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Count awake stage bouts in the detected sleep session.

    Boundary awake bouts are included, so this is not necessarily the count of
    awakenings after sleep onset.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data with ``sleep_summary_id`` as index.
    sleep_stages : :class:`pandas.DataFrame`
        Sleep stages.

    Returns
    -------
    pd.Series
        Count of awake stage bouts.

    Raises
    ------
    ValueError
        If parameters are not of :class:`pandas.DataFrame` .
    """
    count_stage = _compute_stage_count(
        sleep_summary=sleep_summary, sleep_stages=sleep_stages
    )
    if len(count_stage) == 0:
        return pd.Series()
    if schema.SLEEP_STAGE_AWAKE_STAGE_VALUE in count_stage.columns:
        return count_stage[schema.SLEEP_STAGE_AWAKE_STAGE_VALUE]
    else:
        return pd.Series(index=sleep_summary.index)


def _compute_light_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    count_stage = _compute_stage_count(
        sleep_summary=sleep_summary, sleep_stages=sleep_stages
    )
    if len(count_stage) == 0:
        return pd.Series()
    if schema.SLEEP_STAGE_LIGHT_STAGE_VALUE in count_stage.columns:
        return count_stage[schema.SLEEP_STAGE_LIGHT_STAGE_VALUE]
    else:
        return pd.Series(index=sleep_summary.index)


def _compute_deep_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    count_stage = _compute_stage_count(
        sleep_summary=sleep_summary, sleep_stages=sleep_stages
    )
    if len(count_stage) == 0:
        return pd.Series()
    if schema.SLEEP_STAGE_DEEP_STAGE_VALUE in count_stage.columns:
        return count_stage[schema.SLEEP_STAGE_DEEP_STAGE_VALUE]
    else:
        return pd.Series(index=sleep_summary.index)


def _compute_rem_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    count_stage = _compute_stage_count(
        sleep_summary=sleep_summary, sleep_stages=sleep_stages
    )
    if len(count_stage) == 0:
        return pd.Series()
    if schema.SLEEP_STAGE_REM_STAGE_VALUE in count_stage.columns:
        return count_stage[schema.SLEEP_STAGE_REM_STAGE_VALUE]
    else:
        return pd.Series(index=sleep_summary.index)


def _compute_stage_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    # Check is dataframe
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    sleep_stages = _require_dataframe(sleep_stages, "sleep_stages")
    # Check that we have data
    if len(sleep_summary) == 0:
        return pd.DataFrame()
    if len(sleep_stages) == 0:
        return pd.DataFrame(index=sleep_summary.index)

    # Count maximal contiguous bouts rather than provider/export rows.
    filtered_sleep_stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                sleep_summary[schema.SLEEP_SUMMARY_ID_COL].unique()
            )
        ].reset_index(drop=True)
    )

    # Check that we have sleep stages
    if len(filtered_sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)

    # Count the number of awake sleep stages for each group of sleep stages
    count_df = (
        filtered_sleep_stages.groupby([schema.SLEEP_SUMMARY_ID_COL])[
            schema.SLEEP_STAGE_SLEEP_TYPE_COL
        ]
        .value_counts()
        .reset_index()
        .pivot(
            columns=schema.SLEEP_STAGE_SLEEP_TYPE_COL,
            index=schema.SLEEP_SUMMARY_ID_COL,
            values="count",
        )
    )
    count_df = count_df.fillna(0)
    count_df = pd.merge(
        left=count_df,
        right=sleep_summary.loc[:, [schema.SLEEP_SUMMARY_ID_COL]].reset_index(),
        right_on=schema.SLEEP_SUMMARY_ID_COL,
        left_index=True,
        how="left",
    )
    count_df = count_df.set_index(schema.CALENDAR_DATE_COL).drop(
        schema.SLEEP_SUMMARY_ID_COL, axis=1
    )
    generic_sleep = (
        count_df["asleep"].gt(0)
        if "asleep" in count_df.columns
        else pd.Series(False, index=count_df.index)
    )
    # A missing stage in a detailed trajectory is a genuine zero. Classic
    # trajectories cannot distinguish deep/light/REM, so those counts stay missing;
    # nights without any accepted trajectory remain absent and align to missing later.
    for col in [
        schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
        schema.SLEEP_STAGE_REM_STAGE_VALUE,
        schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
        schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
    ]:
        if not col in count_df.columns:
            count_df[col] = 0.0
        else:
            count_df[col] = count_df[col].fillna(0)
    count_df.loc[
        generic_sleep,
        [
            schema.SLEEP_STAGE_REM_STAGE_VALUE,
            schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
            schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        ],
    ] = np.nan
    return count_df


def _get_sleep_summary_ids(sleep_summary: pd.DataFrame) -> pd.Series:
    if schema.SLEEP_SUMMARY_ID_COL in sleep_summary.columns:
        return sleep_summary[schema.SLEEP_SUMMARY_ID_COL]
    if sleep_summary.index.name == schema.SLEEP_SUMMARY_ID_COL:
        return pd.Series(sleep_summary.index, index=sleep_summary.index)
    raise ValueError("sleep summary data requires sleep_summary_id")


def _compute_summary_end_time(
    sleep_summary: pd.DataFrame, time_col: str
) -> pd.Series:
    """Return the provider sleep-session end in the requested clock."""
    end_col = (
        schema.LOCAL_END_TIME_COL
        if time_col == schema.LOCAL_TIME_COL
        else schema.UTC_END_TIME_COL
    )
    if end_col in sleep_summary.columns:
        end_time = pd.to_datetime(sleep_summary[end_col], errors="coerce")
    else:
        end_time = pd.Series(pd.NaT, index=sleep_summary.index, dtype="datetime64[ns]")

    if time_col not in sleep_summary.columns:
        return end_time
    duration = (
        sleep_summary[schema.DURATION_IN_MS_COL]
        + sleep_summary[schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL].fillna(0)
    )
    reconstructed = pd.to_datetime(
        sleep_summary[time_col], errors="coerce"
    ) + pd.to_timedelta(duration, unit="ms")
    return end_time.fillna(reconstructed)


def _compute_sleep_boundaries(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame
) -> pd.DataFrame:
    """Derive observed sleep onset/offset while retaining session boundaries."""
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    sleep_stages = _require_dataframe(sleep_stages, "sleep_stages")
    boundaries = pd.DataFrame(index=sleep_summary.index)
    for column in [
        "sleep_onset",
        "wakeup",
        "__sleep_onset_elapsed",
        "__wakeup_elapsed",
        "__getup_elapsed",
    ]:
        boundaries[column] = pd.NaT
    boundaries["__uses_utc"] = False
    if sleep_summary.empty or sleep_stages.empty:
        return boundaries

    summary_ids = _get_sleep_summary_ids(sleep_summary)
    stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                summary_ids.unique()
            )
        ].reset_index(drop=True)
    )
    groups = {
        summary_id: group.reset_index(drop=True)
        for summary_id, group in stages.groupby(
            schema.SLEEP_SUMMARY_ID_COL, sort=False
        )
    }
    local_getup = _compute_summary_end_time(sleep_summary, schema.LOCAL_TIME_COL)
    utc_getup = _compute_summary_end_time(sleep_summary, schema.UTC_TIME_COL)
    sleep_labels = {
        schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
        schema.SLEEP_STAGE_REM_STAGE_VALUE,
        "asleep",
    }
    grid_tolerance = pd.Timedelta(minutes=5)

    for calendar_date, summary in sleep_summary.iterrows():
        group = groups.get(summary_ids.loc[calendar_date])
        if group is None or group.empty:
            continue
        use_utc = (
            schema.UTC_TIME_COL in group.columns
            and group[schema.UTC_TIME_COL].notna().all()
            and pd.notna(utc_getup.loc[calendar_date])
        )
        time_col = schema.UTC_TIME_COL if use_utc else schema.LOCAL_TIME_COL
        group = group.sort_values(time_col, kind="stable").reset_index(drop=True)
        durations = pd.to_timedelta(
            group[schema.DURATION_IN_MS_COL], unit="ms"
        )
        ends = group[time_col] + durations
        is_sleep = group[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(sleep_labels)
        if not is_sleep.any():
            continue

        onset_position = group.loc[is_sleep, time_col].idxmin()
        last_sleep_position = ends.where(is_sleep).idxmax()
        last_sleep_end = ends.loc[last_sleep_position]
        getup_elapsed = (
            utc_getup.loc[calendar_date] if use_utc else local_getup.loc[calendar_date]
        )
        boundaries.loc[calendar_date, "sleep_onset"] = group.loc[
            onset_position, schema.LOCAL_TIME_COL
        ]
        boundaries.loc[calendar_date, "__sleep_onset_elapsed"] = group.loc[
            onset_position, time_col
        ]
        boundaries.loc[calendar_date, "__getup_elapsed"] = getup_elapsed
        boundaries.loc[calendar_date, "__uses_utc"] = use_utc

        tail = group.iloc[last_sleep_position + 1 :]
        wakeup_elapsed = pd.NaT
        wakeup_local = pd.NaT
        if tail.empty:
            # Provider stage grids can differ from the exact session end by up
            # to one epoch. Larger unexplained gaps are left unresolved.
            if pd.isna(getup_elapsed):
                wakeup_elapsed = last_sleep_end
                wakeup_local = (
                    group.loc[last_sleep_position, schema.LOCAL_TIME_COL]
                    + durations.loc[last_sleep_position]
                )
            elif abs(getup_elapsed - last_sleep_end) <= grid_tolerance:
                wakeup_elapsed = getup_elapsed
                wakeup_local = local_getup.loc[calendar_date]
        else:
            expected_start = last_sleep_end
            terminal_awake = True
            for position, stage in tail.iterrows():
                if (
                    stage[schema.SLEEP_STAGE_SLEEP_TYPE_COL]
                    != schema.SLEEP_STAGE_AWAKE_STAGE_VALUE
                    or stage[time_col] != expected_start
                ):
                    terminal_awake = False
                    break
                expected_start = ends.loc[position]
            if (
                terminal_awake
                and pd.notna(getup_elapsed)
                and abs(getup_elapsed - expected_start) > grid_tolerance
            ):
                terminal_awake = False
            if terminal_awake:
                wakeup_elapsed = last_sleep_end
                local_uncertain = (
                    "local_time_uncertain" in group.columns
                    and group.loc[
                        last_sleep_position:, "local_time_uncertain"
                    ].fillna(False).any()
                )
                if not local_uncertain:
                    wakeup_local = (
                        group.loc[last_sleep_position, schema.LOCAL_TIME_COL]
                        + durations.loc[last_sleep_position]
                    )

        boundaries.loc[calendar_date, "wakeup"] = wakeup_local
        boundaries.loc[calendar_date, "__wakeup_elapsed"] = wakeup_elapsed

    return boundaries


def _compute_bedtime(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Retrieves bedtimes from a sleep summary

    This function retrieves, if available, the bedtime
    of all sleep occurrences from given sleep summary data.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pandas.Series`
        Series of bedtimes data in :class:`pandas.Timestamp` dtype.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pandas.DataFrame` .
    """
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    if schema.LOCAL_TIME_COL in sleep_summary.columns:
        bedtime = pd.to_datetime(sleep_summary[schema.LOCAL_TIME_COL])
    else:
        bedtime = pd.Series()
    return bedtime


def _compute_sleep_onset(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Return the first observed sleep-stage timestamp for each night."""
    return _compute_sleep_boundaries(sleep_summary, sleep_stages)["sleep_onset"].rename(
        SLEEP_STATISTIC_SLEEP_ONSET
    )


def _compute_wakeup_time(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Return the final observed sleep offset for each night.

    A trailing unmeasurable interval or an unexplained stage gap makes this
    timestamp unavailable rather than treating the session end as wake-up.
    """
    return _compute_sleep_boundaries(sleep_summary, sleep_stages)["wakeup"].rename(
        SLEEP_STATISTIC_WAKEUP
    )


def _compute_getup_time(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Return the provider sleep-session end as a proxy for get-up time."""
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")
    getup = _compute_summary_end_time(sleep_summary, schema.LOCAL_TIME_COL)
    getup.name = SLEEP_STATISTIC_GETUP
    return getup


def _compute_sleep_midpoint(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Return the elapsed-time midpoint between sleep onset and wake-up."""
    boundaries = _compute_sleep_boundaries(sleep_summary, sleep_stages)
    midpoints = pd.Series(pd.NaT, index=sleep_summary.index, dtype="datetime64[ns]")
    for calendar_date, boundary in boundaries.iterrows():
        if pd.isna(boundary["sleep_onset"]) or pd.isna(boundary["wakeup"]):
            continue
        elapsed = boundary["__wakeup_elapsed"] - boundary["__sleep_onset_elapsed"]
        if boundary["__uses_utc"]:
            local_elapsed = boundary["wakeup"] - boundary["sleep_onset"]
            if local_elapsed != elapsed:
                # Without a timezone identifier, a local midpoint across an
                # offset change cannot be reconstructed unambiguously.
                continue
        midpoints.loc[calendar_date] = boundary["sleep_onset"] + elapsed / 2
    midpoints.name = SLEEP_STATISTIC_MIDPOINT
    return midpoints


def _get_sleep_calendar_dates(sleep_summary: pd.DataFrame) -> pd.Series:
    """Return calendar dates aligned to the supplied sleep-summary index."""
    values = (
        sleep_summary[schema.CALENDAR_DATE_COL]
        if schema.CALENDAR_DATE_COL in sleep_summary.columns
        else sleep_summary.index
    )
    dates = pd.Series(pd.to_datetime(values, errors="coerce"), index=sleep_summary.index).dt.normalize()
    if dates.isna().any():
        raise ValueError("CPD requires a valid calendar date for every sleep summary.")
    return dates


def _compute_cpd_midpoint(
    sleep_summary: pd.DataFrame, chronotype: Union[tuple, None], **kwargs
):
    """Computes CPD midpoints from a sleep summary

    This function computes, if possible, the composite phase deviation metric
    with respect to sleep midpoints from all sleep occurrences
    from given sleep summary data.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pandas.Series`
        Series of CPD midpoints data.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pandas.DataFrame` .
    """
    # Check is dataframe
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")

    calendar_dates = _get_sleep_calendar_dates(sleep_summary)
    chronological_index = calendar_dates.sort_values(kind="stable").index

    # get midpoints for period of interest
    midpoints: pd.Series = _compute_sleep_midpoint(
        sleep_summary=sleep_summary, **kwargs
    ).reindex(chronological_index)

    if chronotype is None and midpoints.notna().sum() == 0:
        return pd.Series(np.nan, index=sleep_summary.index, dtype=float)
    if (
        chronotype is None
    ):  # if chronotype isn't explicitely given, calculate the midpoint from the available data
        chronotype_midpoint = _mean_clock_time(midpoints)
    else:  # from specified times
        chronotype_midpoint = _mean_clock_time(chronotype)

    previous_midpoint = None
    previous_calendar_date = None
    cpds = pd.Series(np.nan, index=midpoints.index, dtype=float)

    if len(midpoints) == 1:
        warnings.warn(
            "Only one day is considered: CPD cannot be computed without the previous calendar day."
        )

    for summary_id, midpoint in midpoints.items():
        calendar_date = calendar_dates.loc[summary_id]
        # mistiming component
        chronotype_daily_midpoint = pd.Timestamp(calendar_date).replace(
            hour=int(chronotype_midpoint[:2]), minute=int(chronotype_midpoint[3:])
        )
        # if the expected midpoint is prior to midnight we need to adjust the day
        if 14 < int(chronotype_midpoint[:2]) < 24:
            chronotype_daily_midpoint -= pd.Timedelta(days=1)
        mistiming_component = (chronotype_daily_midpoint - midpoint).total_seconds() / (
            60 * 60
        )
        # irregularity component
        has_previous_day = (
            previous_calendar_date is not None
            and calendar_date - previous_calendar_date == pd.Timedelta(days=1)
            and pd.notna(previous_midpoint)
            and pd.notna(midpoint)
        )
        if has_previous_day:
            previous_day_midpoint_proxy = pd.Timestamp(calendar_date).replace(
                hour=previous_midpoint.hour,
                minute=previous_midpoint.minute,
                second=previous_midpoint.second,
            )

            if 14 < previous_day_midpoint_proxy.hour < 24:
                previous_day_midpoint_proxy -= datetime.timedelta(days=1)

            irregularity_component = (
                previous_day_midpoint_proxy - midpoint
            ).total_seconds() / (60 * 60)

            cpds.loc[summary_id] = np.sqrt(
                mistiming_component**2 + irregularity_component**2
            )

        previous_midpoint = midpoint
        previous_calendar_date = calendar_date

    return cpds.reindex(sleep_summary.index)


def _compute_cpd_duration(
    sleep_summary: pd.DataFrame, chronotype: Union[tuple, None], **kwargs
):
    """Computes CPD durations from a sleep summary

    This function computes, if possible, the composite phase deviation metric
    with respect to sleep durations from all sleep occurrences
    from given sleep summary data.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pandas.Series`
        Series of CPD durations data.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pandas.DataFrame` .
    """
    # Check is dataframe
    sleep_summary = _require_dataframe(sleep_summary, "sleep_summary")

    calendar_dates = _get_sleep_calendar_dates(sleep_summary)
    chronological_index = calendar_dates.sort_values(kind="stable").index
    durations = (
        _compute_total_sleep_time(sleep_summary=sleep_summary) / 60
    ).reindex(chronological_index)

    if chronotype is None:  # from data
        chronotype_sleep_duration = durations.mean()
    else:  # from specified times
        chronotype_start = chronotype[0]
        chronotype_end = chronotype[1]
        chronotype_sleep_duration = (
            datetime.datetime.strptime(chronotype_end, "%H:%M")
            - datetime.datetime.strptime(chronotype_start, "%H:%M")
        ).total_seconds() / (
            60 * 60
        )  # convert to hours
        if chronotype_sleep_duration < 0:  # takes care of sleep-time prior to midnight
            chronotype_sleep_duration += 24

    if len(durations) == 1:
        warnings.warn(
            "Only one day is considered: CPD cannot be computed without the previous calendar day."
        )

    cpds = pd.Series(np.nan, index=durations.index, dtype=float)
    previous_duration = None
    previous_calendar_date = None

    for summary_id, duration in durations.items():
        calendar_date = calendar_dates.loc[summary_id]
        mistiming_component = chronotype_sleep_duration - duration

        has_previous_day = (
            previous_calendar_date is not None
            and calendar_date - previous_calendar_date == pd.Timedelta(days=1)
            and pd.notna(previous_duration)
            and pd.notna(duration)
        )
        if has_previous_day:
            irregularity_component = previous_duration - duration
            cpds.loc[summary_id] = np.sqrt(
                mistiming_component**2 + irregularity_component**2
            )

        previous_duration = duration
        previous_calendar_date = calendar_date

    return cpds.reindex(sleep_summary.index)


def compute_sleep_validation(sleep_summary: pd.DataFrame, **kwargs):
    return sleep_summary["validation"]


def _compute_getup_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Return minutes from final observed sleep offset to session end."""
    boundaries = _compute_sleep_boundaries(sleep_summary, sleep_stages)
    latency = (
        boundaries["__getup_elapsed"] - boundaries["__wakeup_elapsed"]
    ).dt.total_seconds() / 60
    latency = latency.where(latency >= 0)
    latency.name = SLEEP_STATISTIC_GETUP_LATENCY
    return latency


def _compute_awakening_metrics(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame
) -> pd.DataFrame:
    """Return count and median duration of internal awake bouts."""
    metrics = pd.DataFrame(
        np.nan,
        index=sleep_summary.index,
        columns=[
            SLEEP_STATISTIC_AWAKENING_COUNT,
            SLEEP_STATISTIC_MEDIAN_AWAKENING_DURATION,
        ],
    )
    if sleep_summary.empty or sleep_stages.empty:
        return metrics

    summary_ids = _get_sleep_summary_ids(sleep_summary)
    stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                summary_ids.unique()
            )
        ].reset_index(drop=True)
    )
    summary_index_by_id = dict(zip(summary_ids, sleep_summary.index))
    for summary_id, group in stages.groupby(schema.SLEEP_SUMMARY_ID_COL, sort=False):
        time_col = schema.LOCAL_TIME_COL
        if (
            schema.UTC_TIME_COL in group.columns
            and group[schema.UTC_TIME_COL].notna().all()
        ):
            time_col = schema.UTC_TIME_COL
        group = group.sort_values(time_col, kind="stable")
        durations = pd.to_timedelta(
            group[schema.DURATION_IN_MS_COL], unit="ms"
        )
        ends = group[time_col] + durations
        is_sleep = ~group[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(
            [
                schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
            ]
        )
        if not is_sleep.any():
            continue
        onset = group.loc[is_sleep, time_col].min()
        offset = ends.loc[is_sleep].max()
        awake = group[
            group[schema.SLEEP_STAGE_SLEEP_TYPE_COL]
            == schema.SLEEP_STAGE_AWAKE_STAGE_VALUE
        ]
        awake_start = awake[time_col].clip(lower=onset)
        awake_end = (
            awake[time_col]
            + pd.to_timedelta(
                awake[schema.DURATION_IN_MS_COL], unit="ms"
            )
        ).clip(upper=offset)
        bout_durations = (
            (awake_end - awake_start).dt.total_seconds() / 60
        ).loc[lambda values: values > 0]
        calendar_date = summary_index_by_id[summary_id]
        metrics.loc[calendar_date, SLEEP_STATISTIC_AWAKENING_COUNT] = len(
            bout_durations
        )
        if len(bout_durations):
            metrics.loc[
                calendar_date, SLEEP_STATISTIC_MEDIAN_AWAKENING_DURATION
            ] = bout_durations.median()
    return metrics


def _compute_awakening_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    return _compute_awakening_metrics(sleep_summary, sleep_stages)[
        SLEEP_STATISTIC_AWAKENING_COUNT
    ]


def _compute_median_awakening_duration(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    return _compute_awakening_metrics(sleep_summary, sleep_stages)[
        SLEEP_STATISTIC_MEDIAN_AWAKENING_DURATION
    ]


def _compute_awakening_index(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    count = _compute_awakening_count(sleep_summary, sleep_stages)
    tst_hours = _compute_total_sleep_time(sleep_summary) / 60
    result = count / tst_hours.where(tst_hours > 0)
    result.name = SLEEP_STATISTIC_AWAKENING_INDEX
    return result


def _compute_stage_transition_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Count direct transitions among light, deep, and REM sleep.

    Awake, unmeasurable, gaps, and generic ``asleep`` intervals break the
    sequence and are not bridged.
    """
    result = pd.Series(np.nan, index=sleep_summary.index, dtype=float)
    if sleep_summary.empty or sleep_stages.empty:
        return result
    summary_ids = _get_sleep_summary_ids(sleep_summary)
    stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                summary_ids.unique()
            )
        ].reset_index(drop=True)
    )
    summary_index_by_id = dict(zip(summary_ids, sleep_summary.index))
    detailed_stages = {
        schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
        schema.SLEEP_STAGE_REM_STAGE_VALUE,
    }
    for summary_id, group in stages.groupby(schema.SLEEP_SUMMARY_ID_COL, sort=False):
        if "asleep" in set(group[schema.SLEEP_STAGE_SLEEP_TYPE_COL]):
            continue
        time_col = schema.LOCAL_TIME_COL
        if (
            schema.UTC_TIME_COL in group.columns
            and group[schema.UTC_TIME_COL].notna().all()
        ):
            time_col = schema.UTC_TIME_COL
        group = group.sort_values(time_col, kind="stable").reset_index(drop=True)
        if not group[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(detailed_stages).any():
            continue
        previous_end = group[time_col].shift() + pd.to_timedelta(
            group[schema.DURATION_IN_MS_COL].shift(), unit="ms"
        )
        previous_type = group[schema.SLEEP_STAGE_SLEEP_TYPE_COL].shift()
        current_type = group[schema.SLEEP_STAGE_SLEEP_TYPE_COL]
        transitions = (
            current_type.isin(detailed_stages)
            & previous_type.isin(detailed_stages)
            & current_type.ne(previous_type)
            & group[time_col].eq(previous_end)
        )
        result.loc[summary_index_by_id[summary_id]] = int(transitions.sum())
    result.name = SLEEP_STATISTIC_STAGE_TRANSITION_COUNT
    return result


def _compute_stage_transition_index(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    count = _compute_stage_transition_count(sleep_summary, sleep_stages)
    tst_hours = _compute_total_sleep_time(sleep_summary) / 60
    result = count / tst_hours.where(tst_hours > 0)
    result.name = SLEEP_STATISTIC_STAGE_TRANSITION_INDEX
    return result


def _compute_stage_coverage(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Return measurable stage duration as a percentage of time in bed."""
    result = pd.Series(np.nan, index=sleep_summary.index, dtype=float)
    if sleep_summary.empty or sleep_stages.empty:
        return result
    summary_ids = _get_sleep_summary_ids(sleep_summary)
    stages = schema._coalesce_sleep_stage_runs(
        sleep_stages[
            sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                summary_ids.unique()
            )
        ].reset_index(drop=True)
    )
    measurable = stages[
        stages[schema.SLEEP_STAGE_SLEEP_TYPE_COL]
        != schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE
    ]
    measurable_ms = measurable.groupby(schema.SLEEP_SUMMARY_ID_COL)[
        schema.DURATION_IN_MS_COL
    ].sum()
    has_stages = summary_ids.isin(stages[schema.SLEEP_SUMMARY_ID_COL].unique())
    tib_ms = (
        sleep_summary[schema.DURATION_IN_MS_COL]
        + sleep_summary[schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL].fillna(0)
    )
    coverage_by_id = (
        summary_ids.map(measurable_ms).fillna(0).where(has_stages)
        / tib_ms.where(tib_ms > 0)
        * 100
    ).clip(upper=100)
    result.loc[:] = coverage_by_id.to_numpy()
    result.name = SLEEP_STATISTIC_STAGE_COVERAGE
    return result


SLEEP_STATISTICS_DICT = {
    SLEEP_STATISTIC_TIB: _compute_time_in_bed,
    SLEEP_STATISTIC_TST: _compute_total_sleep_time,
    SLEEP_STATISTIC_GETUP_LATENCY: _compute_getup_latency,
    SLEEP_STATISTIC_AWAKENING_COUNT: _compute_awakening_count,
    SLEEP_STATISTIC_AWAKENING_INDEX: _compute_awakening_index,
    SLEEP_STATISTIC_MEDIAN_AWAKENING_DURATION: _compute_median_awakening_duration,
    SLEEP_STATISTIC_STAGE_TRANSITION_COUNT: _compute_stage_transition_count,
    SLEEP_STATISTIC_STAGE_TRANSITION_INDEX: _compute_stage_transition_index,
    SLEEP_STATISTIC_STAGE_COVERAGE: _compute_stage_coverage,
    SLEEP_STATISTIC_SE: _compute_sleep_efficiency,
    SLEEP_STATISTIC_SME: _compute_sleep_maintenance_efficiency,
    SLEEP_STATISTIC_SPT: _compute_sleep_period_time,
    SLEEP_STATISTIC_WASO: _compute_wake_after_sleep_onset,
    SLEEP_STATISTIC_SOL: _compute_sleep_onset_latency,
    SLEEP_STATISTIC_DEEP_DURATION: _compute_deep_duration,
    SLEEP_STATISTIC_LIGHT_DURATION: _compute_light_duration,
    SLEEP_STATISTIC_REM_DURATION: _compute_rem_duration,
    SLEEP_STATISTIC_NREM_DURATION: _compute_nrem_duration,
    SLEEP_STATISTIC_AWAKE_DURATION: _compute_awake_duration,
    SLEEP_STATISTIC_UNMEASURABLE_DURATION: _compute_unmeasurable_duration,
    SLEEP_STATISTIC_DEEP_PERCENTAGE: _compute_deep_percentage,
    SLEEP_STATISTIC_LIGHT_PERCENTAGE: _compute_light_percentage,
    SLEEP_STATISTIC_REM_PERCENTAGE: _compute_rem_percentage,
    SLEEP_STATISTIC_NREM_PERCENTAGE: _compute_nrem_percentage,
    SLEEP_STATISTIC_DEEP_LATENCY: _compute_deep_latency,
    SLEEP_STATISTIC_LIGHT_LATENCY: _compute_light_latency,
    SLEEP_STATISTIC_REM_LATENCY: _compute_rem_latency,
    SLEEP_STATISTIC_SLEEP_SCORE: _compute_sleep_score,
    SLEEP_STATISTIC_AWAKE_COUNT: _compute_awake_count,
    SLEEP_STATISTIC_DEEP_COUNT: _compute_deep_count,
    SLEEP_STATISTIC_LIGHT_COUNT: _compute_light_count,
    SLEEP_STATISTIC_REM_COUNT: _compute_rem_count,
    SLEEP_STATISTIC_BEDTIME: _compute_bedtime,
    SLEEP_STATISTIC_SLEEP_ONSET: _compute_sleep_onset,
    SLEEP_STATISTIC_WAKEUP: _compute_wakeup_time,
    SLEEP_STATISTIC_GETUP: _compute_getup_time,
    SLEEP_STATISTIC_MIDPOINT: _compute_sleep_midpoint,
    SLEEP_STATISTIC_CPD_MIDPOINT: _compute_cpd_midpoint,
    SLEEP_STATISTIC_CPD_DURATION: _compute_cpd_duration,
    SLEEP_STATISTIC_VALIDATION: compute_sleep_validation,
}


def _sleep_statistics_list(statistic):
    if statistic is None:
        statistics = [
            name
            for name in SLEEP_STATISTICS_DICT
            if name != SLEEP_STATISTIC_VALIDATION
        ]
    else:
        statistics = [statistic] if isinstance(statistic, str) else list(statistic)
    if not statistics:
        raise ValueError("statistic must contain at least one sleep feature")
    unknown = set(statistics).difference(SLEEP_STATISTICS_DICT)
    if unknown:
        raise ValueError(
            "statistic must be one among "
            f"{list(SLEEP_STATISTICS_DICT)}, not {sorted(unknown)}"
        )
    return statistics


# TODO reduce repeated sleep-feature calculations


def get_sleep_statistic(
    loader: BaseLoader,
    user_id: Union[str, list],
    statistic: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    chronotype: Union[dict, None] = None,
    loader_kwargs: Union[dict, None] = None,
    resolution: str = "nightly",
    aggregation=None,
    min_coverage=None,
    min_valid_children=None,
) -> pd.DataFrame:
    """Return one sleep feature at nightly or calendar resolution.

    This convenience wrapper delegates to :func:`get_sleep_statistics` and
    returns a :class:`pandas.DataFrame` with coverage information.
    """
    return get_sleep_statistics(
        loader=loader,
        user_id=user_id,
        statistic=statistic,
        start_date=start_date,
        end_date=end_date,
        chronotype=chronotype,
        loader_kwargs=loader_kwargs,
        resolution=resolution,
        aggregation=aggregation,
        min_coverage=min_coverage,
        min_valid_children=min_valid_children,
    )


def get_daily_sleep_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: Union[dict, None] = None,
    chronotype: Union[dict, None] = None,
    aggregation=None,
    min_coverage=None,
    min_valid_children=None,
) -> pd.DataFrame:
    """Return the nightly main-sleep trajectory keyed by local calendar day."""
    return get_sleep_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="daily",
        loader_kwargs=loader_kwargs,
        chronotype=chronotype,
        aggregation=aggregation,
        min_coverage=min_coverage,
        min_valid_children=min_valid_children,
    )


def get_weekly_sleep_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: Union[dict, None] = None,
    chronotype: Union[dict, None] = None,
    aggregation=None,
    min_coverage=None,
    min_valid_children=None,
) -> pd.DataFrame:
    """Return sleep statistics aggregated across local calendar weeks."""
    return get_sleep_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="weekly",
        loader_kwargs=loader_kwargs,
        chronotype=chronotype,
        aggregation=aggregation,
        min_coverage=min_coverage,
        min_valid_children=min_valid_children,
    )


def get_monthly_sleep_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: Union[dict, None] = None,
    chronotype: Union[dict, None] = None,
    aggregation=None,
    min_coverage=None,
    min_valid_children=None,
) -> pd.DataFrame:
    """Return sleep statistics aggregated across local calendar months."""
    return get_sleep_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="monthly",
        loader_kwargs=loader_kwargs,
        chronotype=chronotype,
        aggregation=aggregation,
        min_coverage=min_coverage,
        min_valid_children=min_valid_children,
    )


def get_yearly_sleep_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: Union[dict, None] = None,
    chronotype: Union[dict, None] = None,
    aggregation=None,
    min_coverage=None,
    min_valid_children=None,
) -> pd.DataFrame:
    """Return sleep statistics aggregated across local calendar years."""
    return get_sleep_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="yearly",
        loader_kwargs=loader_kwargs,
        chronotype=chronotype,
        aggregation=aggregation,
        min_coverage=min_coverage,
        min_valid_children=min_valid_children,
    )


def _nightly_sleep_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs: Union[dict, None] = None,
    chronotype: Union[dict, None] = None,
    min_coverage=None,
) -> pd.DataFrame:
    """Return one main-sleep feature row per participant calendar date.

    Nightly coverage is binary feature availability. Requested or inferred
    calendar dates with a non-missing feature have coverage ``1``; dates with
    a missing feature have coverage ``0``.
    """

    # Get user id from the loader
    users = resolve_user_ids(loader, user_id)
    # Convert to datetime
    start_date = None if start_date is None else pd.Timestamp(start_date)
    end_date = None if end_date is None else pd.Timestamp(end_date)

    statistics = _sleep_statistics_list(statistic)
    coverage_threshold = hierarchy.normalize_coverage_thresholds(
        min_coverage, 1
    )[0]

    if loader_kwargs is None:
        loader_kwargs = {}
    if type(loader_kwargs) != dict:
        raise ValueError(
            f"loader_kwargs must be of type dict, not {type(loader_kwargs)}"
        )
    user_frames = {}
    for user in users:
        # Load sleep summary data
        sleep_summary: pd.DataFrame = loader.load_sleep_summary(
            user_id=user,
            start_date=start_date,
            end_date=end_date,
            drop_duplicates=True,
            **loader_kwargs,
        )
        sleep_summary = schema.prepare_sleep_summary(sleep_summary)
        if len(sleep_summary) > 0:
            # Check if we have a chronotype for the user
            user_chronotype = None
            if type(chronotype) is dict:
                if user in chronotype.keys():
                    user_chronotype = chronotype[user]

            sleep_summary = sleep_summary.set_index(schema.CALENDAR_DATE_COL)

            sleep_summary_start_date = sleep_summary[schema.LOCAL_TIME_COL].min()
            if "local_end_time" in sleep_summary.columns:
                sleep_summary_end_date = sleep_summary["local_end_time"].max()
            else:
                sleep_summary_end_date = (
                    sleep_summary[schema.LOCAL_TIME_COL]
                    + pd.to_timedelta(
                        (
                            sleep_summary[schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL]
                            + sleep_summary[schema.DURATION_IN_MS_COL]
                        ),
                        unit="ms",
                    )
                ).max()
            # Load sleep stages
            sleep_stages = loader.load_sleep_stage(
                user_id=user,
                start_date=sleep_summary_start_date,
                end_date=sleep_summary_end_date,
            )
            sleep_stages = schema.prepare_sleep_stages(sleep_stages)
            # Keep only those belonging to sleep summaries
            sleep_stages = sleep_stages[
                sleep_stages[schema.SLEEP_SUMMARY_ID_COL].isin(
                    sleep_summary[schema.SLEEP_SUMMARY_ID_COL].unique()
                )
            ]
            user_statistics_list = []
            for statistic in statistics:
                # Get series with metrics by calendar date.
                ser: pd.Series = SLEEP_STATISTICS_DICT[statistic](
                    sleep_summary=sleep_summary,
                    sleep_stages=sleep_stages,
                    chronotype=user_chronotype,
                ).reindex(sleep_summary.index)

                user_statistics_list.append(ser.rename(statistic))
            user_frame = pd.concat(user_statistics_list, axis=1)
        else:
            user_frame = pd.DataFrame(
                columns=statistics,
                index=pd.DatetimeIndex([], name=schema.CALENDAR_DATE_COL),
            )

        user_start = (
            pd.Timestamp(start_date).normalize()
            if start_date is not None
            else user_frame.index.min().normalize()
            if not user_frame.empty
            else None
        )
        user_end = (
            pd.Timestamp(end_date).normalize()
            if end_date is not None
            else user_frame.index.max().normalize() + pd.Timedelta(days=1)
            if not user_frame.empty
            else None
        )
        if user_start is not None and user_end is not None:
            expected = pd.date_range(
                user_start, user_end, freq="D", inclusive="left"
            )
            user_frame = user_frame.reindex(expected)
            user_frame.index.name = schema.CALENDAR_DATE_COL
        if user_frame.empty:
            continue
        for name in statistics:
            coverage_col = f"{name}__coverage"
            user_frame[coverage_col] = user_frame[name].notna().astype(float)
            if coverage_threshold is not None:
                user_frame.loc[
                    user_frame[coverage_col] < coverage_threshold, name
                ] = np.nan
        user_frames[user] = user_frame
    if not user_frames:
        empty_index = pd.MultiIndex.from_arrays(
            [[], []], names=[schema.USER_COL, schema.CALENDAR_DATE_COL]
        )
        columns = [
            column
            for name in statistics
            for column in (name, f"{name}__coverage")
        ]
        return pd.DataFrame(columns=columns, index=empty_index)
    sleep_stats_df = pd.concat(
        user_frames, names=[schema.USER_COL, schema.CALENDAR_DATE_COL]
    ).sort_index()

    columns = [
        column
        for name in statistics
        for column in (name, f"{name}__coverage")
    ]
    return sleep_stats_df[columns]


def _hierarchical_sleep_statistics(
    loader,
    statistics,
    users,
    start_date,
    end_date,
    chain,
    loader_kwargs,
    chronotype,
    aggregation,
    min_coverage,
    min_valid_children,
):
    if chain[0] != "nightly":
        raise ValueError("sleep hierarchies must start with nightly resolution")
    if any(target not in {"weekly", "monthly", "yearly"} for target in chain[1:]):
        raise ValueError(
            "nightly sleep features can be aggregated to weekly, monthly, or yearly"
        )
    aggregation_levels = hierarchy.normalize_aggregation_levels(
        aggregation, len(chain) - 1
    )
    coverage_thresholds = hierarchy.normalize_coverage_thresholds(
        min_coverage, len(chain)
    )
    valid_child_thresholds = hierarchy.normalize_valid_child_thresholds(
        min_valid_children, len(chain)
    )
    hierarchy.warn_if_wide(statistics, aggregation_levels)
    nightly = _nightly_sleep_statistics(
        loader=loader,
        statistic=statistics,
        user_id=users,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
        chronotype=chronotype,
        min_coverage=coverage_thresholds[0],
    )
    frames = {}
    bounds = {}
    for user in users:
        if user in nightly.index.get_level_values(schema.USER_COL):
            user_frame = nightly.xs(user).copy()
        else:
            user_frame = pd.DataFrame(columns=statistics)
        if start_date is None:
            if user_frame.empty:
                continue
            user_start = user_frame.index.min().normalize()
        else:
            user_start = pd.Timestamp(start_date).normalize()
        if end_date is None:
            if user_frame.empty:
                continue
            user_end = user_frame.index.max().normalize() + pd.Timedelta(days=1)
        else:
            user_end = pd.Timestamp(end_date).normalize()
        expected = pd.date_range(user_start, user_end, freq="D", inclusive="left")
        user_frame = user_frame.reindex(expected)
        user_frame.index.name = "window_start"
        user_frame.insert(0, "period_start", expected)
        user_frame.insert(1, "period_end", expected + pd.Timedelta(days=1))
        frames[user] = user_frame
        bounds[user] = (user_start, user_end)

    if not frames:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=["user_id", schema.CALENDAR_DATE_COL]
        )
        return pd.DataFrame(index=index)
    trajectory = pd.concat(frames, names=["user_id", "window_start"])
    active_users = list(frames)
    rename = {feature: f"{feature}__nightly" for feature in statistics}
    trajectory = trajectory.rename(columns=rename)
    temporal = {
        rename[feature]
        for feature in SLEEP_TEMPORAL_STATISTICS
        if feature in rename
    }

    for level_index, target in enumerate(chain[1:]):
        specification = aggregation_levels[level_index]
        parent = hierarchy.calendar_windows(active_users, target, bounds)
        trajectory, temporal = hierarchy.aggregate_level(
            trajectory,
            parent,
            target_resolution=target,
            aggregation=specification,
            temporal_features=temporal,
            membership="label",
            min_coverage=coverage_thresholds[level_index + 1],
            min_valid_children=valid_child_thresholds[level_index + 1],
        )
    return trajectory.rename_axis(
        index={"window_start": schema.CALENDAR_DATE_COL}
    )


def get_sleep_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    resolution: Union[str, list, tuple] = "nightly",
    loader_kwargs: Union[dict, None] = None,
    chronotype: Union[dict, None] = None,
    aggregation=None,
    min_coverage=None,
    min_valid_children=None,
) -> pd.DataFrame:
    """Return main-sleep features at night level or along a hierarchy.

    A scalar weekly, monthly, or yearly resolution is shorthand for a
    ``nightly`` hierarchy ending at that calendar resolution. Resolution
    sequences explicitly define each increasingly larger level, for example
    ``["nightly", "weekly", "monthly"]``. Upper levels give each valid child
    row equal weight and return immediate ``valid / expected`` coverage in
    ``[0, 1]``. Nightly coverage is binary feature availability: ``1`` when
    the feature is present and ``0`` when it is missing on an expected
    calendar date. At upper levels, coverage is the fraction of expected
    nights containing a valid feature.

    ``aggregation`` selects the operation or branching operations at each
    upper level; ``None`` uses the shared standard operations. A
    ``min_coverage`` sequence supplies one threshold per resolution. A scalar
    threshold on a scalar calendar request applies to that calendar level.
    ``min_valid_children`` independently requires an absolute number of valid
    child windows at each upper level; a scalar applies to every upper level,
    while a sequence uses ``None`` for the nightly level.
    """
    chain, hierarchical = hierarchy.normalize_resolution_chain(resolution)
    if start_date is not None and end_date is not None:
        if (
            pd.Timestamp(end_date).normalize()
            <= pd.Timestamp(start_date).normalize()
        ):
            raise ValueError("end_date must be after start_date")
    if not hierarchical:
        target = chain[0]
        if target in SLEEP_NIGHTLY_RESOLUTIONS or target == "daily":
            if aggregation is not None:
                raise ValueError("aggregation requires an upper sleep resolution")
            if min_valid_children is not None:
                raise ValueError("min_valid_children requires an upper resolution")
            return _nightly_sleep_statistics(
                loader=loader,
                statistic=statistic,
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                loader_kwargs=loader_kwargs,
                chronotype=chronotype,
                min_coverage=min_coverage,
            )
        if target not in {"weekly", "monthly", "yearly"}:
            raise ValueError(
                "sleep statistics resolution must be nightly, daily, weekly, "
                "monthly, or yearly"
            )
        chain = ["nightly", target]
        if min_coverage is None:
            min_coverage = [None, None]
        elif isinstance(min_coverage, (list, tuple)):
            if len(min_coverage) != 2:
                raise ValueError(
                    "calendar sleep coverage requires nightly and calendar thresholds"
                )
        else:
            min_coverage = [None, min_coverage]
        if min_valid_children is None:
            min_valid_children = [None, None]
        elif isinstance(min_valid_children, (list, tuple)):
            if len(min_valid_children) != 2:
                raise ValueError(
                    "calendar sleep valid-child thresholds require nightly and "
                    "calendar values"
                )
        else:
            min_valid_children = [None, min_valid_children]

    users = resolve_user_ids(loader, user_id)
    statistics = _sleep_statistics_list(statistic)
    return _hierarchical_sleep_statistics(
        loader,
        statistics,
        users,
        start_date,
        end_date,
        chain,
        loader_kwargs,
        chronotype,
        aggregation,
        min_coverage,
        min_valid_children,
    )


__all__ = [
    "SLEEP_NIGHTLY_RESOLUTIONS",
    "SLEEP_STATISTICS_DICT",
    "SLEEP_STATISTIC_AWAKENING_COUNT",
    "SLEEP_STATISTIC_AWAKENING_INDEX",
    "SLEEP_STATISTIC_AWAKE_COUNT",
    "SLEEP_STATISTIC_AWAKE_DURATION",
    "SLEEP_STATISTIC_BEDTIME",
    "SLEEP_STATISTIC_CPD_DURATION",
    "SLEEP_STATISTIC_CPD_MIDPOINT",
    "SLEEP_STATISTIC_DEEP_COUNT",
    "SLEEP_STATISTIC_DEEP_DURATION",
    "SLEEP_STATISTIC_DEEP_LATENCY",
    "SLEEP_STATISTIC_DEEP_PERCENTAGE",
    "SLEEP_STATISTIC_GETUP",
    "SLEEP_STATISTIC_GETUP_LATENCY",
    "SLEEP_STATISTIC_LIGHT_COUNT",
    "SLEEP_STATISTIC_LIGHT_DURATION",
    "SLEEP_STATISTIC_LIGHT_LATENCY",
    "SLEEP_STATISTIC_LIGHT_PERCENTAGE",
    "SLEEP_STATISTIC_MEDIAN_AWAKENING_DURATION",
    "SLEEP_STATISTIC_MIDPOINT",
    "SLEEP_STATISTIC_NREM_DURATION",
    "SLEEP_STATISTIC_NREM_PERCENTAGE",
    "SLEEP_STATISTIC_REM_COUNT",
    "SLEEP_STATISTIC_REM_DURATION",
    "SLEEP_STATISTIC_REM_LATENCY",
    "SLEEP_STATISTIC_REM_PERCENTAGE",
    "SLEEP_STATISTIC_SE",
    "SLEEP_STATISTIC_SLEEP_ONSET",
    "SLEEP_STATISTIC_SLEEP_SCORE",
    "SLEEP_STATISTIC_SME",
    "SLEEP_STATISTIC_SOL",
    "SLEEP_STATISTIC_SPT",
    "SLEEP_STATISTIC_SRI",
    "SLEEP_STATISTIC_STAGE_COVERAGE",
    "SLEEP_STATISTIC_STAGE_TRANSITION_COUNT",
    "SLEEP_STATISTIC_STAGE_TRANSITION_INDEX",
    "SLEEP_STATISTIC_TIB",
    "SLEEP_STATISTIC_TST",
    "SLEEP_STATISTIC_UNMEASURABLE_DURATION",
    "SLEEP_STATISTIC_VALIDATION",
    "SLEEP_STATISTIC_WAKEUP",
    "SLEEP_STATISTIC_WASO",
    "SLEEP_TEMPORAL_STATISTICS",
    "SLEEP_WAKE_TRAJECTORY_COLUMNS",
    "get_awakening_count",
    "get_awakening_index",
    "get_awake_count",
    "get_awake_duration",
    "get_bedtime",
    "get_cpd_duration",
    "get_cpd_midpoint",
    "get_daily_sleep_statistics",
    "get_deep_latency",
    "get_deep_sleep_count",
    "get_deep_sleep_duration",
    "get_deep_sleep_percentage",
    "get_getup_latency",
    "get_getup_time",
    "get_light_latency",
    "get_light_sleep_count",
    "get_light_sleep_duration",
    "get_light_sleep_percentage",
    "get_median_awakening_duration",
    "get_monthly_sleep_statistics",
    "get_nrem_duration",
    "get_nrem_percentage",
    "get_rem_count",
    "get_rem_duration",
    "get_rem_latency",
    "get_rem_percentage",
    "get_sleep_efficiency",
    "get_sleep_maintenance_efficiency",
    "get_sleep_midpoint",
    "get_sleep_onset",
    "get_sleep_onset_latency",
    "get_sleep_period_time",
    "get_sleep_regularity",
    "get_sleep_regularity_index",
    "get_sleep_score",
    "get_sleep_statistic",
    "get_sleep_statistics",
    "get_sleep_timestamps",
    "get_sleep_wake_trajectory",
    "get_stage_coverage",
    "get_stage_transition_count",
    "get_stage_transition_index",
    "get_time_in_bed",
    "get_total_sleep_time",
    "get_unmeasurable_duration",
    "get_wake_after_sleep_onset",
    "get_wakeup_time",
    "get_weekly_sleep_statistics",
    "get_yearly_sleep_statistics",
]
