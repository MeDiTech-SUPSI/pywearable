"""
This module contains all the functions related to analysis
of sleep data.
"""

import datetime
import warnings
from typing import Union

import numpy as np
import pandas as pd

from . import sleep_schema as constants, utils
from .loader.base import BaseLoader

# TODO add support for quantiles

SLEEP_STATISTIC_TIB = "time_in_bed"
SLEEP_STATISTIC_TST = "total_sleep_time"
SLEEP_STATISTIC_WASO = "wake_after_sleep_onset"
SLEEP_STATISTIC_SPT = "sleep_period_time"
SLEEP_STATISTIC_DEEP_DURATION = "deep_sleep_duration"
SLEEP_STATISTIC_LIGHT_DURATION = "light_sleep_duration"
SLEEP_STATISTIC_REM_DURATION = "rem_sleep_duration"
SLEEP_STATISTIC_NREM_DURATION = "nrem_sleep_duration"
SLEEP_STATISTIC_AWAKE_DURATION = "awake_duration"
SLEEP_STATISTIC_UNMEASURABLE_DURATION = "unmeasurable_duration"
SLEEP_STATISTIC_SOL = "sleep_onset_latency"
SLEEP_STATISTIC_DEEP_LATENCY = "deep_sleep_latency"
SLEEP_STATISTIC_LIGHT_LATENCY = "light_sleep_latency"
SLEEP_STATISTIC_REM_LATENCY = "rem_sleep_latency"
SLEEP_STATISTIC_DEEP_PERCENTAGE = "deep_sleep_percentage"
SLEEP_STATISTIC_LIGHT_PERCENTAGE = "light_sleep_percentage"
SLEEP_STATISTIC_NREM_PERCENTAGE = "nrem_sleep_percentage"
SLEEP_STATISTIC_REM_PERCENTAGE = "rem_sleep_percentage"
SLEEP_STATISTIC_SE = "sleep_efficiency"
SLEEP_STATISTIC_SME = "sleep_maintenance_efficiency"
SLEEP_STATISTIC_SLEEP_SCORE = "sleep_score"
SLEEP_STATISTIC_AWAKE_COUNT = "awake_count"
SLEEP_STATISTIC_DEEP_COUNT = "deep_sleep_count"
SLEEP_STATISTIC_LIGHT_COUNT = "light_sleep_count"
SLEEP_STATISTIC_REM_COUNT = "rem_sleep_count"
SLEEP_STATISTIC_BEDTIME = "bedtime"
SLEEP_STATISTIC_WAKEUP = "wakeup"
SLEEP_STATISTIC_MIDPOINT = "midpoint"
SLEEP_STATISTIC_CPD_MIDPOINT = "cpd_midpoint"
SLEEP_STATISTIC_CPD_DURATION = "cpd_duration"
SLEEP_STATISTIC_VALIDATION = "validation"
SLEEP_STATISTIC_SRI = "sleep_regularity_index"
SLEEP_STATISTIC_GETUP_LATENCY = "getup_latency"

SLEEP_DATETIME_STATISTICS = [
    SLEEP_STATISTIC_BEDTIME,
    SLEEP_STATISTIC_MIDPOINT,
    SLEEP_STATISTIC_WAKEUP,
]

SLEEP_KIND_MAPPING = {
    "mean": {
        SLEEP_STATISTIC_BEDTIME: "",  # utils.mean_time,
        SLEEP_STATISTIC_WAKEUP: "",  # utils.mean_time,
        SLEEP_STATISTIC_MIDPOINT: "",  # utils.mean_time,
    },
    "std": {
        SLEEP_STATISTIC_BEDTIME: "",  # utils.std_time,
        SLEEP_STATISTIC_WAKEUP: "",  # utils.std_time,
        SLEEP_STATISTIC_MIDPOINT: "",  # utils.std_time,
    },
    "min": {
        SLEEP_STATISTIC_BEDTIME: "",  # utils.get_earliest_bedtime,
        SLEEP_STATISTIC_WAKEUP: "",  # utils.get_earliest_wakeup_time,
        SLEEP_STATISTIC_MIDPOINT: "",  # utils.get_earliest_wakeup_time,
    },
    "max": {
        SLEEP_STATISTIC_BEDTIME: "",  # utils.get_latest_bedtime,
        SLEEP_STATISTIC_WAKEUP: "",  # utils.get_latest_wakeup_time,
        SLEEP_STATISTIC_MIDPOINT: "",  # utils.get_latest_wakeup_time,
    },
}


def get_time_in_bed(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get time in bed (TIB) sleep metric.

    This function computes time in bed (TIB) metric as the total duration
    of the sleep, given by the sum of all sleep stages and unmeasurable sleep
    as well. The value is reported in minutes.

    Depending on the value of the ``kind`` parameter, this function
    returns TIB for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_time_in_bed(loader, user_id, start_date, end_date, kind=None)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 501.0,
            datetime.date(2023, 9, 10): 514.0,
            datetime.date(2023, 9, 11): 455.0,
            datetime.date(2023, 9, 12): 437.0,
            datetime.date(2023, 9, 13): 437.0,
            datetime.date(2023, 9, 14): 402.0,
            datetime.date(2023, 9, 15): 469.0}

        pywearable.sleep.get_time_in_bed(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'TIB': 463.14285714285717,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15']
                    }
            }


    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which TIB must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform TIB over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and TIB as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `TIB`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_TIB,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_period_time(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> pd.DataFrame:
    """Get sleep period time (SPT) sleep metric.

    This function computes sleep period time (SPT) metric as the duration from the first to
    the last period of sleep. Depending on the value of the ``kind`` parameter, this function
    returns SPT for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_sleep_period_time(loader, user_id, start_date, end_date, kind=None)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 501.0,
            datetime.date(2023, 9, 10): 514.0,
            datetime.date(2023, 9, 11): 455.0,
            datetime.date(2023, 9, 12): 437.0,
            datetime.date(2023, 9, 13): 437.0,
            datetime.date(2023, 9, 14): 402.0,
            datetime.date(2023, 9, 15): 496.0}

        pywearable.sleep.get_sleep_period_time(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'SPT': 463.14285714285717,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which SPT must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform SPT over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and SPT as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `SPT`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SPT,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_total_sleep_time(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> pd.DataFrame:
    """Get total sleep time (TST) sleep metric.

    This function computes total sleep time (TST) metric as the total duration of
    N1, N2, N3, and REM in SPT (see :func:`get_sleep_period_time`).
    Depending on the value of the ``kind`` parameter, this function
    returns TST for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_total_sleep_time(loader, user_id, start_date, end_date, kind=None)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 497.0,
            datetime.date(2023, 9, 10): 512.0,
            datetime.date(2023, 9, 11): 455.0,
            datetime.date(2023, 9, 12): 435.0,
            datetime.date(2023, 9, 13): 435.0,
            datetime.date(2023, 9, 14): 402.0,
            datetime.date(2023, 9, 15): 494.0}

        pywearable.sleep.get_total_sleep_time(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'TST': 461.42857142857144,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which TST must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform TST over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and TST as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `TST`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_TST,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_efficiency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> pd.DataFrame:
    """Get sleep efficiency (SE) sleep metric.

    This function computes sleep efficiency (SE) metric as the ratio
    between TST (see :func:`get_total_sleep_time`) and TIB (see :func:`get_time_in_bed`).
    .. math:: SE = TST / TIB
    Depending on the value of the ``kind`` parameter, this function
    returns SE for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_sleep_efficiency(loader, user_id, start_date, end_date, kind=None)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 99.20159680638723,
            datetime.date(2023, 9, 10): 99.61089494163424,
            datetime.date(2023, 9, 11): 100.0,
            datetime.date(2023, 9, 12): 99.54233409610984,
            datetime.date(2023, 9, 13): 99.54233409610984,
            datetime.date(2023, 9, 14): 100.0,
            datetime.date(2023, 9, 15): 99.59677419354838}

        pywearable.sleep.get_sleep_efficiency(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'SE': 99.64199059054135,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which SE must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform SE over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and SE as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `SE`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SE,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_maintenance_efficiency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> pd.DataFrame:
    """Get sleep maintenance efficiency (SME) sleep metric.

    This function computes sleep maintenance efficiency (SME) metric as the ratio
    between TST (see :func:`get_total_sleep_time`) and SPT (see :func:`get_sleep_period_time`).

    .. math:: SME = \\frac{TST}{SPT}

    Depending on the value of the ``kind`` parameter, this function
    returns SME for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_sleep_maintenance_efficiency(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 99.20159680638723,
            datetime.date(2023, 9, 10): 99.61089494163424,
            datetime.date(2023, 9, 11): 100.0,
            datetime.date(2023, 9, 12): 99.54233409610984,
            datetime.date(2023, 9, 13): 99.54233409610984,
            datetime.date(2023, 9, 14): 100.0,
            datetime.date(2023, 9, 15): 99.59677419354838}

        pywearable.sleep.get_sleep_maintenance_efficiency(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'SE': 99.64199059054135,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which SME must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform SME over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and SME as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `SME`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SME,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_deep_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> pd.Series:
    """Get N1 sleep latency.

    This function computes the latency to the first stage of N1 sleep.
    Depending on the value of the ``kind`` parameter, this function
    returns N1 latency for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_n1_latency(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 0.0,
            datetime.date(2023, 9, 10): 0.0,
            datetime.date(2023, 9, 11): 0.0,
            datetime.date(2023, 9, 12): 0.0,
            datetime.date(2023, 9, 13): 0.0,
            datetime.date(2023, 9, 14): 0.0,
            datetime.date(2023, 9, 15): 0.0}

        pywearable.sleep.get_n1_latency(loader, user_id, start_date, end_date, average=True)
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'Lat_N1': 0.0,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which N1 sleep latency must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform N1 latency over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and N1 latency as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `Lat_N1`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistics(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_LATENCY,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_light_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> pd.Series:
    return get_sleep_statistics(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_LATENCY,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_rem_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get REM sleep latency.

    This function computes the latency to the first stage of REM sleep.
    Depending on the value of the ``kind`` parameter, this function
    returns REM latency for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_rem_latency(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 47.0,
            datetime.date(2023, 9, 10): 119.0,
            datetime.date(2023, 9, 11): 69.0,
            datetime.date(2023, 9, 12): 57.0,
            datetime.date(2023, 9, 13): 79.0,
            datetime.date(2023, 9, 14): 167.0,
            datetime.date(2023, 9, 15): 52.0}

        pywearable.sleep.get_rem_latency(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'Lat_REM': 84.28571428571429,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which REM sleep latency must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform REM latency over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and REM latency as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `Lat_REM`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_LATENCY,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_light_sleep_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get light sleep duration.

    This function returns the absolute time spent in light sleep stage for
    the given user(s). Duration is reported in minutes.
    Depending on the value of the ``kind`` parameter, this function
    returns N1 duration for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_light_sleep_duration(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 280.0,
            datetime.date(2023, 9, 10): 256.0,
            datetime.date(2023, 9, 11): 333.0,
            datetime.date(2023, 9, 12): 312.0,
            datetime.date(2023, 9, 13): 288.0,
            datetime.date(2023, 9, 14): 233.0,
            datetime.date(2023, 9, 15): 295.0}

        pywearable.sleep.get_n1_duration(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'N1': 285.2857142857143,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        Initialized instance of :class:`pywearable.loader.base.BaseLoader`, required in order to properly load data.
    user_id : :class:`str` or :class:`list`, optional
        IDs of the users for which N1 sleep data have to be extracted, by default "all"
    start_date : :class:`datetime.datetime`, optional
        Start date from which sleep data should be extracted, by default None.
        If None is used, then the ``start_date`` will be the first day with available sleep data
        for the given ``user_id``, by default None
    end_date : :class:`datetime.datetime`, optional
        End date up to which sleep data should be extracted, by default None.
        If None is used, then the ``end_date`` will be the last day with available sleep data
        for the given ``user_id``, by default None
    kind : :class:`str` or None, optional
        Whether to transform N1 duration over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and N1 duration as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `N1`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_DURATION,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_deep_sleep_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get deep sleep duration.

    This function returns the absolute time spent in deep sleep stage for
    the given user(s). Depending on the value of the ``kind`` parameter, this function
    returns N1 duration for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Example::

        pywearable.sleep.get_n3_duration(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 53.0,
            datetime.date(2023, 9, 10): 112.0,
            datetime.date(2023, 9, 11): 37.0,
            datetime.date(2023, 9, 12): 49.0,
            datetime.date(2023, 9, 13): 66.0,
            datetime.date(2023, 9, 14): 76.0,
            datetime.date(2023, 9, 15): 83.0}

        pywearable.sleep.get_n3_duration(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'N3': 68.0,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        Initialized instance of :class:`pywearable.loader.base.BaseLoader`, required in order to properly load data.
    user_id : :class:`str` or :class:`list`, optional
        IDs of the users for which N3 sleep data have to be extracted, by default "all"
    start_date : :class:`datetime.datetime`, optional
        Start date from which sleep data should be extracted, by default None.
        If None is used, then the ``start_date`` will be the first day with available sleep data
        for the given ``user_id``, by default None
    end_date : :class:`datetime.datetime`, optional
        End date up to which sleep data should be extracted, by default None.
        If None is used, then the ``end_date`` will be the last day with available sleep data
        for the given ``user_id``, by default None
    kind : :class:`str` or None, optional
        Whether to transform N3 duration over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and N2 duration as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `N3`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_DURATION,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_rem_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get REM sleep duration.

    This function returns the absolute time spent in REM sleep stage for
    the given user(s). Depending on the value of the ``kind`` parameter, this function
    returns REM duration for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Duration is reported in minutes.

    Example::

        pylabfront.sleep.get_rem_duration(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 164.0,
            datetime.date(2023, 9, 10): 144.0,
            datetime.date(2023, 9, 11): 85.0,
            datetime.date(2023, 9, 12): 74.0,
            datetime.date(2023, 9, 13): 81.0,
            datetime.date(2023, 9, 14): 93.0,
            datetime.date(2023, 9, 15): 116.0}

        pylabfront.sleep.get_rem_duration(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'REM': 108.14285714285714,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        Initialized instance of :class:`pywearable.loader.base.BaseLoader`, required in order to properly load data.
    user_id : :class:`str` or :class:`list`, optional
        IDs of the users for which REM sleep data have to be extracted, by default "all"
    start_date : :class:`datetime.datetime`, optional
        Start date from which sleep data should be extracted, by default None.
        If None is used, then the ``start_date`` will be the first day with available sleep data
        for the given ``user_id``, by default None
    end_date : :class:`datetime.datetime`, optional
        End date up to which sleep data should be extracted, by default None.
        If None is used, then the ``end_date`` will be the last day with available sleep data
        for the given ``user_id``, by default None
    kind : :class:`str` or None, optional
        Whether to transform REM duration over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and REM duration as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `REM`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_DURATION,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_nrem_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get NREM sleep duration.

    This function returns the absolute time spent in NREM sleep stage for
    the given user(s). Depending on the value of the ``kind`` parameter, this function
    returns N1 duration for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Duration is reported in minutes.

    Example::

        pywearable.sleep.get_nrem_duration(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 333.0,
            datetime.date(2023, 9, 10): 368.0,
            datetime.date(2023, 9, 11): 370.0,
            datetime.date(2023, 9, 12): 361.0,
            datetime.date(2023, 9, 13): 354.0,
            datetime.date(2023, 9, 14): 309.0,
            datetime.date(2023, 9, 15): 433.0}

        pywearable.sleep.get_nrem_duration(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'NREM': 361.14285714285717,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        Initialized instance of :class:`pywearable.loader.base.BaseLoader`, required in order to properly load data.
    user_id : :class:`str` or :class:`list`, optional
        IDs of the users for which sleep data have to be extracted, by default "all".
    start_date : :class:`datetime.datetime`, optional
        Start date from which sleep data should be extracted, by default None.
        If None is used, then the ``start_date`` will be the first day with available sleep data.
        for the given ``user_id``, by default None
    end_date : :class:`datetime.datetime`, optional
        End date up to which sleep data should be extracted, by default None.
        If None is used, then the ``end_date`` will be the last day with available sleep data.
        for the given ``user_id``, by default None
    kind : :class:`str` or None, optional
        Whether to transform NREM duration over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and NREM duration as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `NREM`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_NREM_DURATION,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_awake_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get awake sleep duration.

    This function returns the absolute time spent in awake stage for
    the given user(s). Depending on the value of the ``kind`` parameter, this function
    returns awake duration for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Duration is reported in minutes.

    Example::

        pywearable.sleep.get_awake_duration(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 4.0,
            datetime.date(2023, 9, 10): 2.0,
            datetime.date(2023, 9, 11): 0.0,
            datetime.date(2023, 9, 12): 2.0,
            datetime.date(2023, 9, 13): 2.0,
            datetime.date(2023, 9, 14): 0.0,
            datetime.date(2023, 9, 15): 5.0}

        pywearable.sleep.get_awake_duration(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'AWAKE': 2.142857142857143,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        Initialized instance of :class:`pywearable.loader.base.BaseLoader`, required in order to properly load data.
    user_id : :class:`str` or :class:`list`, optional
        IDs of the users for which sleep data have to be extracted, by default "all"
    start_date : :class:`datetime.datetime`, optional
        Start date from which sleep data should be extracted, by default None.
        If None is used, then the ``start_date`` will be the first day with available sleep data
        for the given ``user_id``, by default None
    end_date : :class:`datetime.datetime`, optional
        End date up to which sleep data should be extracted, by default None.
        If None is used, then the ``end_date`` will be the last day with available sleep data
        for the given ``user_id``, by default None
    kind : :class:`str` or None, optional
        Whether to transform awake duration over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and awake duration as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `AWAKE`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_AWAKE_DURATION,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_unmeasurable_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get unmeasurable sleep duration.

    This function returns the absolute time spent in unmeasurable sleep stage for
    the given user(s). Depending on the value of the ``kind`` parameter, this function
    returns unmeasurable sleep duration for each calendar day (``kind=None``)
    from ``start_date`` to ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Duration is reported in minutes.

    Example::

        pywearable.sleep.get_unmeasurable_duration(loader, user_id, start_date, end_date)

        >> {'f457c562-2159-431c-8866-dfa9a917d9b8':
            datetime.date(2023, 9, 9): 0.0,
            datetime.date(2023, 9, 10): 0.0,
            datetime.date(2023, 9, 11): 0.0,
            datetime.date(2023, 9, 12): 0.0,
            datetime.date(2023, 9, 13): 0.0,
            datetime.date(2023, 9, 14): 0.0,
            datetime.date(2023, 9, 15): 0.0}

        pywearable.sleep.get_unmeasurable_duration(loader, user_id, start_date, end_date, kind='mean')
        >> {
                'f457c562-2159-431c-8866-dfa9a917d9b8':
                    {
                        'UNMEASURABLE': 0.0,
                        'days': ['2023-09-09',
                                    '2023-09-10',
                                    '2023-09-11',
                                    '2023-09-12',
                                    '2023-09-13',
                                    '2023-09-14',
                                    '2023-09-15'
                                ]
                    }
            }

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        Initialized instance of :class:`pywearable.loader.base.BaseLoader`, required in order to properly load data.
    user_id : :class:`str` or :class:`list`, optional
        IDs of the users for which sleep data have to be extracted, by default "all"
    start_date : :class:`datetime.datetime`, optional
        Start date from which sleep data should be extracted, by default None.
        If None is used, then the ``start_date`` will be the first day with available sleep data
        for the given ``user_id``, by default None
    end_date : :class:`datetime.datetime`, optional
        End date up to which sleep data should be extracted, by default None.
        If None is used, then the ``end_date`` will be the last day with available sleep data
        for the given ``user_id``, by default None
    kind : :class:`str` or None, optional
        Whether to transform unmeasurable sleep duration over days, or to return the value for each day, by default None.
        Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and unmeasurable sleep duration as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `UNMEASURABLE`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_UNMEASURABLE_DURATION,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_light_sleep_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_deep_sleep_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_rem_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_nrem_percentage(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_NREM_PERCENTAGE,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_wake_after_sleep_onset(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_WASO,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_onset_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SOL,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_score(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SLEEP_SCORE,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_timestamps(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
) -> Union[dict, pd.DataFrame]:
    """Get the timestamps of the beginning and the end of sleep occurrences.

    Returns for every day, the time when the user fell asleep and when he woke up
    The information is based on the sleep summaries of the user

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
    :class:`dict`
        Dictionary with user ids as primary keys, calendar dates as secondary keys,
        and sleep timestamps (bedtime, wakeup_time) as values.
    """
    data_dict = {}

    user_id = utils.get_user_ids(loader, user_id)

    for user in user_id:
        try:
            bedtime_dict = get_bedtime(
                loader, user, start_date, end_date, return_df=False
            )[user]
            wakeup_time_dict = get_wakeup_time(
                loader, user, start_date, end_date, return_df=False
            )[user]
            data_dict[user] = {
                k: (bedtime_dict.get(k, None), wakeup_time_dict.get(k, None))
                for k in bedtime_dict.keys()
            }
        except:
            data_dict[user] = {}
            raise
    return data_dict


def get_bedtime(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get bedtime sleep metric.

    Depending on the value of the ``kind`` parameter, this function
    returns bedtime for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which bedtime must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform bedtime over days, or to return the value for each day, by default None.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and bedtime as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `bedtime`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_BEDTIME,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_wakeup_time(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get wake-up time sleep metric.

    This function computes the wake-up time metric.
    The value is reported in a datetime.datetime format.

    Depending on the value of the ``kind`` parameter, this function
    returns wakeup time for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which wakeup time must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform wakeup time over days, or to return the value for each day, by default None.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and wakeup times as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `wakeupTime`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_WAKEUP,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_midpoint(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Get midpoint sleep metric.

    This function computes the sleep midpoint metric.
    For every night of sleep the midpoint between bedtime and wakeup time is computed.
    The value is reported in a datetime.datetime format.

    Depending on the value of the ``kind`` parameter, this function
    returns midpoint for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which midpoints must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform midpoints over days, or to return the value for each day, by default None.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and midpoints as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `midpoint`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_MIDPOINT,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_awake_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    """Get the number of awakenings

    Returns the number of times the user(s) of interest woke up during the night.
    Depending on the value of the ``kind`` parameter, this function
    returns the count of awake stages for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Parameters
    ----------
    loader : :class:`pylabfront.loader.Loader`
        Initialized instance of data loader.
    user_id : :class:`str`, optional
        ID of the user for which awakenings are computed, by default "all".
    start_date : :class:`datetime.datetime`, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime`, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform count of awake stages over days, or to return the value for each
        day, by default None. Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and count of awake stages as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `countAwake`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_AWAKE_COUNT,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_light_sleep_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    """Get the number of light sleep stages.

    Returns the number of light sleep stages for the user(s) of interest.
    This is checked. If ``average`` is set to True, the average number of
    awakenings per night during the period between ``start_date`` and ``end_date`` is returned.

    Parameters
    ----------
    loader : :class:`pylabfront.loader.Loader`
        Initialized instance of data loader.
    user_id : :class:`str`, optional
        ID of the user for which awakenings are computed, by default "all".
    start_date : :class:`datetime.datetime`, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime`, optional
        End date for data retrieval, by default None
    average : :class:`bool`, optional
        Whether to return only the average number of awakenings per night for the user, by default False

    Returns
    -------
    :class:`dict`
        Dictionary with user ids as primary keys, dates as secondary keys, and number of awakenings as values.
        If `average` is set to True, the average and the dates used for its calculation are returned as values.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_LIGHT_COUNT,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_deep_sleep_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    """Get the number of N3 sleep stages.

    Returns the number of N3 sleep stages for the user(s) of interest.
    Depending on the value of the ``kind`` parameter, this function
    returns the count of N3 stages for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Parameters
    ----------
    loader : :class:`pylabfront.loader.Loader`
        Initialized instance of data loader.
    user_id : :class:`str`, optional
        ID of the user for which awakenings are computed, by default "all".
    start_date : :class:`datetime.datetime`, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime`, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform count of N3 stages over days, or to return the value for each
        day, by default None. Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and count of N3 stages as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `countN3`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_DEEP_COUNT,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_rem_count(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    """Get the number of REM sleep stages.

    Returns the number of REM sleep stages for the user(s) of interest.
    Depending on the value of the ``kind`` parameter, this function
    returns the count of REM stages for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days. The applied
    transformation depends on the value of the ``kind`` parameter:

        - `mean`: computes the average across days
        - `std`: computes the standard deviation across days
        - `min`: computes the minimum value across days
        - `max`: computes the maximum value across days

    Parameters
    ----------
    loader : :class:`pylabfront.loader.Loader`
        Initialized instance of data loader.
    user_id : :class:`str`, optional
        ID of the user for which awakenings are computed, by default "all".
    start_date : :class:`datetime.datetime`, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime`, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to transform count of REM stages over days, or to return the value for each
        day, by default None. Valid options are:

            - `mean`
            - `std`
            - `min`
            - `max`

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and count of REM stages as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `countREM`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """

    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_REM_COUNT,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_cpd_midpoint(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    chronotype_dict: dict = {},
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Computes composite phase deviation (CPD) with regard to sleep midpoints

    Returns a measure of sleep regularity, in terms of stability of rest midpoints.
    The measure is computed for the period between `start_date` and `end_date`
    for every user_id of interest.

    Parameters
    ----------
    loader : `pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which CPD must be retrieved, by default 'all'
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to compute a transformation of CPD over days, or to return the value for each
        day, by default None.
    chronotype_dict : :class:`dict`, optional
        Dictionary specifying for every user his usual sleeping time and waking time in format HH:MM, as a tuple.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and CPD midpoints as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `CPD_midpoint`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_CPD_MIDPOINT,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        chronotype_dict=chronotype_dict,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_sleep_regularity_index(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    """Get Sleep Regularity Index (SRI)."""
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_SRI,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_getup_latency(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    """Get get-up latency sleep metric.

    This function computes the time spent in bed after the last actual
    sleep stage until the final wakeup time.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_GETUP_LATENCY,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_cpd_duration(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    chronotype_dict: dict = {},
    kind: Union[str, None] = None,
    kind_args=[],
    kind_kwargs={},
    loader_kwargs={},
    return_df: bool = True,
    return_multi_index: bool = True,
) -> dict:
    """Computes composite phase deviation (CPD) with regard to sleep duration

    Returns a measure of sleep regularity, in terms of sleep duration.
    The measure is computed for the period between `start_date` and `end_date`
    for every user_id of interest.

    Parameters
    ----------
    loader : `pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which CPD must be retrieved, by default 'all'
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to compute a transformation of CPD over days, or to return the value for each
        day, by default None.
    chronotype_dict : :class:`dict`, optional
        Dictionary specifying for every user his usual sleeping time and waking time in format HH:MM, as a tuple.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and CPD midpoints as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `CPD_duration`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_sleep_statistic(
        loader=loader,
        user_id=user_id,
        statistic=SLEEP_STATISTIC_CPD_DURATION,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        chronotype_dict=chronotype_dict,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
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
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    if constants.SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL in sleep_summary.columns:
        sleep_score = sleep_summary[constants.SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL]
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
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    # TIB = SLEEP_DURATION + AWAKE_DURATION in minutes
    if (constants.SLEEP_SUMMARY_DURATION_IN_MS_COL in sleep_summary.columns) and (
        constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL in sleep_summary.columns
    ):
        tib = (
            sleep_summary[constants.SLEEP_SUMMARY_DURATION_IN_MS_COL]
            + sleep_summary[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL]
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
    # TST = N1+N2+N3+REM
    if (
        (constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL in sleep_summary.columns)
        and (
            constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
        and (
            constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
    ):
        tst = (
            sleep_summary[constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL]
            + sleep_summary[constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL]
            + sleep_summary[constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL]
        ) / (
            1000 * 60
        )  # convert from ms to seconds, and then to minutes
    else:
        tst = pd.Series()
    return tst


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
    # SME = (N1+N2+N3+REM) / (SPT) * 100
    tst = _compute_total_sleep_time(
        sleep_summary=sleep_summary, sleep_stages=sleep_stages
    )
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
        (constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL in sleep_summary.columns)
        and (
            constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
        and (
            constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            in sleep_summary.columns
        )
    ):
        tot_duration = (
            sleep_summary[
                constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            ].fillna(0)
            + sleep_summary[
                constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL
            ].fillna(0)
            + sleep_summary[
                constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            ].fillna(0)
        )
        if sleep_stage == "NREM":
            perc = (
                (
                    tot_duration
                    - sleep_summary[
                        constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
                    ].fillna(0)
                )
                / tot_duration
                * 100
            )
        else:
            if sleep_stage == "light":
                col = constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            elif sleep_stage == "deep":
                col = constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL
            elif sleep_stage == "REM":
                col = constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
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
        tot_duration = (
            sleep_summary[
                constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
            ].fillna(0)
            + sleep_summary[
                constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL
            ].fillna(0)
            + sleep_summary[
                constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            ].fillna(0)
        )
        dur = (
            tot_duration
            - sleep_summary[
                constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
            ].fillna(0)
        ) / (1000 * 60)
    else:
        if stage == "light":
            col = constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL
        elif stage == "deep":
            col = constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL
        elif stage == "REM":
            col = constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL
        elif stage == "awake":
            col = constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL
        elif stage == "unmeasurable":
            col = constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL
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
        The index of the ``sleep_summary`` must be set to the unique `sleepSummaryId`.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        Series with sleep period time as values, and sleep summary ids as index.
    """
    # Check inputs
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    sleep_stages = utils.check_is_df(sleep_stages, "sleep_stages")

    if len(sleep_summary) == 0:
        return pd.Series()
    if len(sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)

    def sleep_diff(group):
        start = group[constants.UNIXTIMESTAMP_IN_MS_COL]
        duration = group[constants.SLEEP_STAGE_DURATION_IN_MS_COL]
        return (start.iloc[-1] + duration.iloc[-1] - start.iloc[0]) / (1000 * 60)

    # Get only sleep stages of interest
    filtered_sleep_stages = sleep_stages[
        sleep_stages[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL].isin(
            sleep_summary[constants.SLEEP_SUMMARY_ID_COL].unique()
        )
    ].reset_index(drop=True)
    spt = pd.Series(
        filtered_sleep_stages[
            filtered_sleep_stages[constants.SLEEP_STAGE_SLEEP_TYPE_COL]
            != constants.SLEEP_STAGE_AWAKE_STAGE_VALUE
        ]
        .reset_index(drop=True)
        .groupby(constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL)
        .apply(sleep_diff),
        name=SLEEP_STATISTIC_SPT,
    )
    spt = pd.merge(
        left=spt,
        right=sleep_summary.loc[:, constants.SLEEP_SUMMARY_ID_COL].reset_index(),
        left_index=True,
        right_on=constants.SLEEP_SUMMARY_ID_COL,
        how="left",
    )
    spt = spt.set_index(constants.CALENDAR_DATE_COL, drop=True)[SLEEP_STATISTIC_SPT]
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
        The index of the ``sleep_summary`` must be set to the unique `sleepSummaryId`.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        Series with sleep onset latency.
    """
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    sleep_stages = utils.check_is_df(sleep_stages, "sleep_stages")
    if len(sleep_summary) == 0:
        return pd.Series()
    if len(sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    # Remove awake latency
    if constants.SLEEP_STAGE_AWAKE_STAGE_VALUE in latencies.columns:
        latencies = latencies.drop([constants.SLEEP_STAGE_AWAKE_STAGE_VALUE], axis=1)
    # Remove unmeasurable latency in case it is there
    if constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE in latencies.columns:
        latencies = latencies.drop(
            [constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE], axis=1
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
        The index of the ``sleep_summary`` must be set to the unique `sleepSummaryId`.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        Series with wake after sleep onset data.
    """
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    sleep_stages = utils.check_is_df(sleep_stages, "sleep_stages")
    if len(sleep_summary) == 0:
        return pd.Series()
    if len(sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)

    # Let's add calendar date indicator to sleep stages
    sleep_stages = pd.merge(
        left=sleep_stages,
        right=sleep_summary.reset_index(names=[constants.CALENDAR_DATE_COL]).loc[
            :,
            [
                constants.CALENDAR_DATE_COL,
                constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL,
            ],
        ],
        left_on=constants.SLEEP_STAGE_SLEEP_SUMMARY_ID_COL,
        right_on=constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL,
        how="left",
    )
    # Get only sleep stages belonging to the sleep summaries of interest
    filtered_sleep_stages = sleep_stages[
        sleep_stages[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL].isin(
            sleep_summary[constants.SLEEP_SUMMARY_ID_COL].unique()
        )
    ].reset_index(drop=True)
    # Get first sleep stages that are awake and remove them
    first_sleep_stages = filtered_sleep_stages.drop_duplicates(
        subset=[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL], keep="first"
    )
    first_sleep_stages = first_sleep_stages[
        first_sleep_stages[constants.SLEEP_STAGE_SLEEP_TYPE_COL]
        == constants.SLEEP_STAGE_AWAKE_STAGE_VALUE
    ]
    filtered_sleep_stages = filtered_sleep_stages[
        ~filtered_sleep_stages.index.isin(first_sleep_stages.index)
    ]
    # Get last sleep stages that are awake and remove them
    last_sleep_stages = filtered_sleep_stages.drop_duplicates(
        subset=[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL], keep="last"
    )
    last_sleep_stages = last_sleep_stages[
        last_sleep_stages[constants.SLEEP_STAGE_SLEEP_TYPE_COL]
        == constants.SLEEP_STAGE_AWAKE_STAGE_VALUE
    ]
    filtered_sleep_stages = filtered_sleep_stages[
        ~filtered_sleep_stages.index.isin(last_sleep_stages.index)
    ]
    # Compute WASO with groupby operation
    waso = pd.Series(
        filtered_sleep_stages[
            filtered_sleep_stages[constants.SLEEP_STAGE_SLEEP_TYPE_COL]
            == constants.SLEEP_STAGE_AWAKE_STAGE_VALUE
        ]
        .groupby(constants.CALENDAR_DATE_COL)[constants.DURATION_IN_MS_COL]
        .sum()
        .rename(SLEEP_STATISTIC_WASO),
    ) / (1000 * 60)
    # Set values to 0 for sleep summaries with no awake stages
    waso = waso.reindex(sleep_summary.index)
    waso.loc[
        (waso.index.isin(sleep_stages[constants.CALENDAR_DATE_COL])) & (waso.isna())
    ] = 0
    return waso


def _compute_light_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Compute light sleep latency from sleep data.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of the ``sleep_summary`` must be set to the unique `sleepSummaryId`.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.DataFrame`
        A DataFrame with N1 Latency as value, and sleep summary ids as indexes.
    """
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    return pd.Series(latencies[constants.SLEEP_STAGE_LIGHT_STAGE_VALUE])


def _compute_deep_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Compute N3 latency from sleep data.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of the ``sleep_summary`` must be set to the unique `sleepSummaryId`.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        A series with N3 Latency as value, and sleep summary ids as indexes.
    """
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    return pd.Series(latencies[constants.SLEEP_STAGE_DEEP_STAGE_VALUE])


def _compute_rem_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Computes REM Latency from sleep data.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_summary`.
        The index of the ``sleep_summary`` must be set to the unique `sleepSummaryId`.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages, that can be extracted using :method:`pywearable.loader.base.BaseLoader.load_sleep_stage`.

    Returns
    -------
    :class:`pd.Series`
        A series with REM Latency as value, and sleep summary ids as index.
    """
    latencies = _compute_latencies(sleep_summary, sleep_stages)
    return pd.Series(latencies[constants.SLEEP_STAGE_REM_STAGE_VALUE])


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
        The index of the ``sleep_summary`` must be set to the unique `sleepSummaryId`.
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
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    sleep_stages = utils.check_is_df(sleep_stages, "sleep_stages")
    # Check lenght of input
    if len(sleep_summary) == 0:
        return pd.DataFrame(
            columns=[
                constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
                constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
                constants.SLEEP_STAGE_REM_STAGE_VALUE,
            ]
        )
    if len(sleep_stages) == 0:
        return pd.DataFrame(
            index=sleep_summary.index,
            columns=[
                constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
                constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
                constants.SLEEP_STAGE_REM_STAGE_VALUE,
            ],
        )
    # Get only sleep stages with valid sleepSummaryId
    filtered_sleep_stages = sleep_stages[
        sleep_stages[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL].isin(
            sleep_summary[constants.SLEEP_SUMMARY_ID_COL].unique()
        )
    ].reset_index(drop=True)
    # Compute latencies by groupby operation
    latencies = (
        filtered_sleep_stages.groupby(
            [
                constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL,
                constants.SLEEP_STAGE_SLEEP_TYPE_COL,
            ]
        )[constants.ISODATE_COL].first()
        - sleep_summary.set_index(constants.SLEEP_SUMMARY_ID_COL)[constants.ISODATE_COL]
    ).dt.total_seconds() / (60)

    # Pivot
    latencies = latencies.rename("latency")
    latencies = latencies.reset_index().pivot(
        columns=constants.SLEEP_STAGE_SLEEP_TYPE_COL,
        index=constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL,
        values="latency",
    )
    # Add columns if they are not there
    for stage in [
        constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
        constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
        constants.SLEEP_STAGE_REM_STAGE_VALUE,
    ]:
        if stage not in latencies.columns:
            latencies[stage] = np.nan

    latencies = latencies.rename_axis(None, axis=1)
    # Add missing sleepSummary from sleep_summary index
    latencies = pd.merge(
        left=sleep_summary.loc[:, [constants.SLEEP_SUMMARY_ID_COL]],
        right=latencies,
        left_on=constants.SLEEP_SUMMARY_ID_COL,
        right_index=True,
        how="outer",
    )
    latencies = latencies.drop(constants.SLEEP_SUMMARY_ID_COL, axis=1)
    return latencies


def _compute_awake_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Retrieves awakenings counts score from sleep stages.

    This function retrieves the count of awakenings
    events occured per night.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data with sleepSummaryId as index.
    sleep_stages : :class:`pandas.DataFrame`
        Sleep stages.

    Returns
    -------
    pd.Series
        Count of awakenings.

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
    if constants.SLEEP_STAGE_AWAKE_STAGE_VALUE in count_stage.columns:
        return count_stage[constants.SLEEP_STAGE_AWAKE_STAGE_VALUE]
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
    if constants.SLEEP_STAGE_LIGHT_STAGE_VALUE in count_stage.columns:
        return count_stage[constants.SLEEP_STAGE_LIGHT_STAGE_VALUE]
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
    if constants.SLEEP_STAGE_DEEP_STAGE_VALUE in count_stage.columns:
        return count_stage[constants.SLEEP_STAGE_DEEP_STAGE_VALUE]
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
    if constants.SLEEP_STAGE_REM_STAGE_VALUE in count_stage.columns:
        return count_stage[constants.SLEEP_STAGE_REM_STAGE_VALUE]
    else:
        return pd.Series(index=sleep_summary.index)


def _compute_stage_count(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    # Check is dataframe
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    sleep_stages = utils.check_is_df(sleep_stages, "sleep_stages")
    # Check that we have data
    if len(sleep_summary) == 0:
        return pd.DataFrame()
    if len(sleep_stages) == 0:
        return pd.DataFrame(index=sleep_summary.index)

    # Get only sleep_stages with sleepSummaryId contained in sleep_summary
    filtered_sleep_stages = sleep_stages[
        sleep_stages[constants.SLEEP_SUMMARY_ID_COL].isin(
            sleep_summary[constants.SLEEP_SUMMARY_ID_COL].unique()
        )
    ].reset_index(drop=True)

    # Check that we have sleep stages
    if len(filtered_sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)

    # Count the number of awake sleep stages for each group of sleep stages
    count_df = (
        filtered_sleep_stages.groupby([constants.SLEEP_SUMMARY_ID_COL])[
            constants.SLEEP_STAGE_SLEEP_TYPE_COL
        ]
        .value_counts()
        .reset_index()
        .pivot(
            columns=constants.SLEEP_STAGE_SLEEP_TYPE_COL,
            index=constants.SLEEP_SUMMARY_ID_COL,
            values="count",
        )
    )
    count_df = count_df.fillna(0)
    count_df = pd.merge(
        left=count_df,
        right=sleep_summary.loc[:, [constants.SLEEP_SUMMARY_ID_COL]].reset_index(),
        right_on=constants.SLEEP_SUMMARY_ID_COL,
        left_index=True,
        how="left",
    )
    count_df = count_df.set_index(constants.CALENDAR_DATE_COL).drop(
        constants.SLEEP_SUMMARY_ID_COL, axis=1
    )
    for col in [
        constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
        constants.SLEEP_STAGE_REM_STAGE_VALUE,
        constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
        constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
        constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
    ]:
        if not col in count_df.columns:
            count_df[col] = np.nan
    return count_df


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
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    if constants.ISODATE_COL in sleep_summary.columns:
        bedtime = pd.to_datetime(sleep_summary[constants.ISODATE_COL])
    else:
        bedtime = pd.Series()
    return bedtime


def _compute_wakeup_time(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Retrieves wake up time from a sleep summary

    This function retrieves, if available, the wake up time from
    given sleep summary data.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pandas.Series`
        Series of wake up time data in :class:`pandas.Timestamp` dtype.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pandas.DataFrame` .
    """

    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")

    if "local_end_time" in sleep_summary.columns:
        return sleep_summary["local_end_time"]

    if np.all(
        [
            constants.ISODATE_COL in sleep_summary.columns,
            constants.SLEEP_SUMMARY_DURATION_IN_MS_COL in sleep_summary.columns,
            constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL in sleep_summary.columns,
        ]
    ):
        wakeup_time = sleep_summary[constants.ISODATE_COL] + pd.to_timedelta(
            sleep_summary[constants.DURATION_IN_MS_COL]
            + sleep_summary[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL].fillna(0),
            unit="ms",
        )
    else:
        wakeup_time = pd.Series()
    return wakeup_time


def _compute_sleep_midpoint(sleep_summary: pd.DataFrame, **kwargs) -> pd.Series:
    """Retrieves midpoints from a sleep summary

    This function retrieves, if available, the midpoint
    of sleep occurrences from given sleep summary data.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summary data.

    Returns
    -------
    :class:`pandas.Series`
        Series of sleep midpoints data in :class:`pandas.Timestamp` dtype.

    Raises
    ------
    ValueError
        If `sleep_summary` is not a :class:`pandas.DataFrame` .
    """
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    wakeup_time = _compute_wakeup_time(sleep_summary=sleep_summary, **kwargs)
    bedtime = _compute_bedtime(sleep_summary=sleep_summary, **kwargs)
    if len(wakeup_time) == len(bedtime) and len(wakeup_time) != 0:
        midpoints = pd.Series(
            data=[bed + (wake - bed) / 2 for bed, wake in zip(bedtime, wakeup_time)],
            index=wakeup_time.index,
        )
    else:
        midpoints = pd.Series()
    return midpoints


def _get_sleep_calendar_dates(sleep_summary: pd.DataFrame) -> pd.Series:
    """Return calendar dates aligned to the supplied sleep-summary index."""
    values = (
        sleep_summary[constants.CALENDAR_DATE_COL]
        if constants.CALENDAR_DATE_COL in sleep_summary.columns
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
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")

    calendar_dates = _get_sleep_calendar_dates(sleep_summary)
    chronological_index = calendar_dates.sort_values(kind="stable").index

    # get midpoints for period of interest
    midpoints: pd.Series = _compute_sleep_midpoint(
        sleep_summary=sleep_summary, **kwargs
    ).reindex(chronological_index)

    if (
        chronotype is None
    ):  # if chronotype isn't explicitely given, calculate the midpoint from the available data
        chronotype_midpoint = utils.mean_time(midpoints)
    else:  # from specified times
        chronotype_midpoint = utils.mean_time(chronotype)

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
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")

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


def _resample_sleep_stages(sleep_stages: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample sleep stages with the desired resolution.

    Parameters
    ----------
    sleep_stages : :class:`pandas.DataFrame`
        Sleep stages to be resampled.
    rule : :class:`str`
        The target resolution.

    Returns
    -------
    :class:`pandas.DataFrame`
        Sleep stages resampled with resolution defined by ``rule``.
    """
    sleep_stages = sleep_stages.copy()
    sleep_stages["endDate"] = sleep_stages[constants.ISODATE_COL] + pd.to_timedelta(
        sleep_stages[constants.DURATION_IN_MS_COL] - 1000, unit="ms"
    )

    sleep_stages = sleep_stages.melt(
        id_vars=[
            constants.UNIXTIMESTAMP_IN_MS_COL,
            constants.SLEEP_STAGE_SLEEP_SUMMARY_ID_COL,
            constants.SLEEP_STAGE_SLEEP_TYPE_COL,
        ],
        value_vars=[constants.ISODATE_COL, "endDate"],
    )

    sleep_stages = (
        sleep_stages.sort_values(by="value")
        .rename(columns={"value": constants.ISODATE_COL})
        .drop("variable", axis=1)
        .set_index(constants.ISODATE_COL)
    )

    sleep_stages = sleep_stages[~sleep_stages.index.duplicated(keep="first")]
    sleep_stages = (
        sleep_stages.groupby(constants.SLEEP_STAGE_SLEEP_SUMMARY_ID_COL)
        .resample(rule=rule, include_groups=False)
        .ffill()
        .droplevel(level=constants.SLEEP_STAGE_SLEEP_SUMMARY_ID_COL)
        .sort_index()
    )
    sleep_stages["sleep"] = "sleep"
    awake_labels = [constants.SLEEP_STAGE_AWAKE_STAGE_VALUE, "awake", "wake"]
    sleep_stages.loc[
        sleep_stages[constants.SLEEP_STAGE_SLEEP_TYPE_COL].isin(awake_labels),
        "sleep",
    ] = "awake"
    return sleep_stages


def _compute_sleep_regularity_index(
    sleep_summary: pd.DataFrame,
    sleep_stages: pd.DataFrame,
    first_avg_wakeup: bool = True,
    last_avg_bedtime: bool = True,
    **kwargs,
) -> pd.Series:
    """Compute sleep regularity index (SRI).

    The SRI measures the similarity of an individual's sleep-wake patterns
    from one day to the next, based on binary sleep-wake state classifications.
    It calculates the percentage probability of being in the same state
    (sleep vs. wake) at any two time points 24-h apart.
    The SRI is scaled to range from 0 (random) to 100 (perfectly regular):

    .. math::

        SRI_i = -100+\\frac{200}{M}\\sum_{j=1}^M\\delta(s_{i,j},s_{i+1,j})

    where M is the number of daily epochs, :math:`s_{i,j}=0` for sleep and
    :math:`s_{i,j} = 1` for wake, and :math:`\\delta(s_{i,j},s_{i+1,j})=1`
    for :math:`s_{i,j}=s_{i+1,j}` and 0 otherwise.

    Parameters
    ----------
    sleep_summary : :class:`pandas.DataFrame`
        Sleep summaries indexed by calendarDate.
    sleep_stages : :class:`pandas.DataFrame`
        Sleep stages.
    first_avg_wakeup : :class:`bool`, optional
        Whether to use the average wakeup value on the first date, by default True.
    last_avg_bedtime : :class:`bool`, optional
        Whether to use the average bedtime value on the last date, by default True.

    Returns
    -------
    :class:`pandas.Series`
        Daily SRI values indexed by calendarDate.
    """
    sleep_stages = sleep_stages.copy()
    sleep_summary = sleep_summary.copy()

    # Ensure one row per day
    sleep_summary = sleep_summary.asfreq("D")

    min_date = sleep_summary.index.min()
    max_date = sleep_summary.index.max()

    if first_avg_wakeup:
        mean_wakeup = utils.mean_time(_compute_wakeup_time(sleep_summary=sleep_summary))
        mean_wakeup = pd.Timedelta(
            hours=int(mean_wakeup.split(":")[0]),
            minutes=int(mean_wakeup.split(":")[1]),
        )
    if last_avg_bedtime:
        mean_bedtime = utils.mean_time(_compute_bedtime(sleep_summary=sleep_summary))
        mean_bedtime = pd.Timedelta(
            hours=int(mean_bedtime.split(":")[0]),
            minutes=int(mean_bedtime.split(":")[1]),
        )

    # Resample sleep stages to one-minute resolution
    sleep_stages = _resample_sleep_stages(sleep_stages=sleep_stages, rule="60s")

    sri_series = pd.Series(
        index=sleep_summary.index, name=SLEEP_STATISTIC_SRI, dtype=float
    )

    for idx, summary_data in sleep_summary.iterrows():
        if pd.isnull(summary_data[constants.SLEEP_SUMMARY_ID_COL]):
            continue
        if idx > min_date:
            if pd.isnull(
                sleep_summary.loc[
                    idx - pd.Timedelta(days=1),
                    constants.SLEEP_SUMMARY_ID_COL,
                ]
            ):
                continue

        previous_day_data = sleep_stages.truncate(
            before=idx - pd.Timedelta(days=1),
            after=idx - pd.Timedelta(seconds=1),
            copy=True,
        )
        current_day_data = sleep_stages.truncate(
            before=idx,
            after=idx + pd.Timedelta(days=1, seconds=-1),
            copy=True,
        )

        if len(previous_day_data) == 0 and not (idx == min_date and first_avg_wakeup):
            sri_series.loc[idx] = np.nan
        elif len(current_day_data) == 0 and not (idx == max_date and last_avg_bedtime):
            sri_series.loc[idx] = np.nan
        else:
            if idx == min_date and not first_avg_wakeup:
                continue
            elif idx == max_date and not last_avg_bedtime:
                continue

            previous_day_minute_data = pd.Series(
                index=pd.timedelta_range(
                    start=0, end="1 day", freq="min", closed="left"
                )
            )
            current_day_minute_data = pd.Series(
                index=pd.timedelta_range(
                    start=0, end="1 day", freq="min", closed="left"
                )
            )

            # Convert timestamps to timedelta for day-level comparison
            previous_day_data.index = previous_day_data.index - (
                idx - pd.Timedelta(days=1)
            )
            current_day_data.index = current_day_data.index - idx

            # Mark sleep periods
            previous_day_minute_data.loc[
                previous_day_data[previous_day_data["sleep"] == "sleep"].index
            ] = 0
            current_day_minute_data.loc[
                current_day_data[current_day_data["sleep"] == "sleep"].index
            ] = 0

            if idx == min_date and first_avg_wakeup:
                if mean_bedtime < mean_wakeup:
                    previous_day_minute_data.loc[
                        pd.Timedelta(value="0 day") : mean_bedtime
                    ] = 1
                    previous_day_minute_data.loc[mean_bedtime:mean_wakeup] = 0
                else:
                    previous_day_minute_data.loc[
                        pd.Timedelta(value="0 day") : mean_wakeup
                    ] = 0

            if idx == max_date and last_avg_bedtime:
                current_day_minute_data.loc[
                    mean_bedtime : pd.Timedelta(value="1 day")
                ] = 0

            # Fill remaining with 1 (wake)
            previous_day_minute_data = previous_day_minute_data.fillna(1)
            current_day_minute_data = current_day_minute_data.fillna(1)

            comparison = previous_day_minute_data.astype(
                "int"
            ) == current_day_minute_data.astype("int")
            sri = -100 + 200 / (24 * 60) * comparison.sum()
            sri_series.loc[idx] = sri

    return sri_series


def _compute_getup_latency(
    sleep_summary: pd.DataFrame, sleep_stages: pd.DataFrame, **kwargs
) -> pd.Series:
    """Compute get-up latency.

    Parameters
    ----------
    sleep_summary : :class:`pd.DataFrame`
        Sleep summaries data.
    sleep_stages : :class:`pd.DataFrame`
        Sleep stages data.

    Returns
    -------
    :class:`pd.Series`
        Get-up latency in minutes.
    """
    sleep_summary = utils.check_is_df(sleep_summary, "sleep_summary")
    sleep_stages = utils.check_is_df(sleep_stages, "sleep_stages")

    if len(sleep_summary) == 0:
        return pd.Series()
    if len(sleep_stages) == 0:
        return pd.Series(index=sleep_summary.index)

    wakeup_times = _compute_wakeup_time(sleep_summary)
    awake_labels = [constants.SLEEP_STAGE_AWAKE_STAGE_VALUE, "awake", "wake"]

    stages_with_date = pd.merge(
        left=sleep_stages,
        right=sleep_summary.reset_index(names=[constants.CALENDAR_DATE_COL])[
            [constants.CALENDAR_DATE_COL, constants.SLEEP_SUMMARY_ID_COL]
        ],
        on=constants.SLEEP_SUMMARY_ID_COL,
        how="inner",
    )

    stages_with_date["end_time"] = stages_with_date[
        constants.ISODATE_COL
    ] + pd.to_timedelta(stages_with_date[constants.DURATION_IN_MS_COL], unit="ms")

    # Identify the last recorded stage for each night to check validity
    last_stages = (
        stages_with_date.sort_values(by=constants.ISODATE_COL)
        .groupby(constants.CALENDAR_DATE_COL)
        .last()
    )
    valid_mask = last_stages[constants.SLEEP_STAGE_SLEEP_TYPE_COL].isin(awake_labels)

    # Filter for true sleep stages to find the final sleep offset
    sleep_mask = ~stages_with_date[constants.SLEEP_STAGE_SLEEP_TYPE_COL].isin(
        awake_labels + [constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE]
    )
    last_sleep_end = (
        stages_with_date[sleep_mask]
        .groupby(constants.CALENDAR_DATE_COL)["end_time"]
        .max()
    )

    # Calculate latency and apply NaN if the night didn't end with a wake stage
    getup_latency = (wakeup_times - last_sleep_end).dt.total_seconds() / 60.0
    getup_latency = getup_latency.where(valid_mask, np.nan)

    # Ensure the index matches the original summary and handle nights with zero valid stages
    getup_latency = getup_latency.reindex(sleep_summary.index)

    return getup_latency


SLEEP_STATISTICS_DICT = {
    SLEEP_STATISTIC_TIB: _compute_time_in_bed,
    SLEEP_STATISTIC_TST: _compute_total_sleep_time,
    SLEEP_STATISTIC_GETUP_LATENCY: _compute_getup_latency,
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
    SLEEP_STATISTIC_WAKEUP: _compute_wakeup_time,
    SLEEP_STATISTIC_MIDPOINT: _compute_sleep_midpoint,
    SLEEP_STATISTIC_CPD_MIDPOINT: _compute_cpd_midpoint,
    SLEEP_STATISTIC_CPD_DURATION: _compute_cpd_duration,
    SLEEP_STATISTIC_VALIDATION: compute_sleep_validation,
    SLEEP_STATISTIC_SRI: _compute_sleep_regularity_index,
}

# TODO reduce number of function calls
# TODO work by calendarDate and not by sleepSummaryId
# TODO add consistent bedtime/wakeup time on rolling basis


def get_sleep_statistic(
    loader: BaseLoader,
    user_id: Union[str, list],
    statistic: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = None,
    kind_args: Union[list, tuple] = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
    **kwargs,
) -> dict:
    """Get sleep statistic from sleep data.

    This function is used by several functions of the module to get a single
    sleep statistic starting from sleep data.
    If multiple statistics are required, then it is more efficient to
    use the :func:`get_sleep_statistics` function that computes all the
    statistics by loading only once sleep data. The following statistics
    can be computed:

        - Time in Bed (``statistic="TIB"``)
        - Total Sleep Time (``statistic="TST"``)
        - Wake After Sleep Onset (``statistic="WASO"``)
        - Sleep Onset Latency (``statistic="SOL"``)
        - Sleep Period Time (``statistic="SPT"``)
        - N1 Sleep Duration (``statistic="N1"``)
        - N2 Sleep Duration (``statistic="N2"``)
        - N3 Sleep Duration (``statistic="N3"``)
        - REM Sleep Duration (``statistic="REM"``)
        - NREM Sleep Duration (``statistic="NREM"``)
        - Awake Duration (``statistic="AWAKE"``)
        - Unmeasurable Sleep Duration (``statistic="UNMEASURABLE"``)
        - N1 Sleep Latency (``statistic="Lat_N1"``)
        - N2 Sleep Latency (``statistic="Lat_N2"``)
        - N3 Sleep Latency (``statistic="Lat_N3"``)
        - REM Sleep Latency (``statistic="Lat_REM"``)
        - N1 Sleep Percentage (``statistic="%N1"``)
        - N2 Sleep Percentage (``statistic="%N2"``)
        - N3 Sleep Percentage (``statistic="%N3"``)
        - Sleep Efficiency (``statistic="SE"``)
        - Sleep Maintenance Efficiency (``statistic="SME"``)
        - Sleep Score (``statistic="SCORE"``)
        - Number of Awakenings (``statistic="Awakenings"``)

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`
        The id(s) for which the sleep statistic must be computed.
    metric : :class:`str`
        The name of the sleep statistic which must be computed.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str` or None, optional
        Whether to perform a transform of the sleep statistic over days, by default None.
        If `None`, then the sleep statistic is returned with a value for each day, otherwise
        a transformation is applied. Valid options are:

            - 'mean'
            - 'min'
            - 'max'
            - 'std'

    Returns
    -------
    dict
        If ``average==False``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (as :class:`datetime.date`) as keys and sleep statistic as values.
        If ``average==True``, dictionary with ``user_id`` as key, and a nested dictionary with the
        name of the sleep statistic as key and its value, and an additional `days` keys that contains an array of all
        calendar days over which the average was computed.
    """
    return get_sleep_statistics(
        loader=loader,
        user_id=user_id,
        statistic=statistic,
        start_date=start_date,
        end_date=end_date,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
        **kwargs,
    )


def get_sleep_statistics(
    loader: BaseLoader,
    user_id: Union[str, list, None],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    statistic: Union[str, list, tuple, None] = None,
    chronotype: Union[dict, None] = {},
    kind=None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
    **kwargs,
) -> Union[pd.DataFrame, dict]:

    # Get user id from the loader
    user_id = utils.get_user_ids(loader, user_id)
    # Convert to datetime
    start_date = None if start_date is None else pd.Timestamp(start_date)
    end_date = None if end_date is None else pd.Timestamp(end_date)

    # Check metrics
    if statistic is None:
        # Return the shared night-level features; optional metrics need an
        # explicit request and may not be available for every loader.
        statistics = [
            name for name in SLEEP_STATISTICS_DICT
            if name not in (SLEEP_STATISTIC_GETUP_LATENCY, SLEEP_STATISTIC_VALIDATION, SLEEP_STATISTIC_SRI)
        ]
    elif type(statistic) == str:
        statistics = [statistic]
    else:
        statistics = statistic

    # Check args and kwargs
    if type(kind_args) != list:
        raise ValueError(f"kind_args must be an iterable, not {type(kind_args)}")
    if type(kind_kwargs) != dict:
        raise ValueError(f"kind_kwargs must be of type dict, not {type(kind_kwargs)}")
    if type(loader_kwargs) != dict:
        raise ValueError(
            f"loader_kwargs must be of type dict, not {type(loader_kwargs)}"
        )

    user_frames = {}
    for user in user_id:
        # Load sleep summary data
        sleep_summary: pd.DataFrame = loader.load_sleep_summary(
            user_id=user,
            start_date=start_date,
            end_date=end_date,
            drop_duplicates=True,
            **loader_kwargs,
        )
        if len(sleep_summary) > 0:
            # Check if we have a chronotype for the user
            user_chronotype = None
            if type(chronotype) is dict:
                if user in chronotype.keys():
                    user_chronotype = chronotype[user]

            sleep_summary = sleep_summary.set_index(constants.CALENDAR_DATE_COL)

            sleep_summary_start_date = sleep_summary[constants.ISODATE_COL].min()
            if "local_end_time" in sleep_summary.columns:
                sleep_summary_end_date = sleep_summary["local_end_time"].max()
            else:
                sleep_summary_end_date = (
                    sleep_summary[constants.ISODATE_COL]
                    + pd.to_timedelta(
                        (
                            sleep_summary[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL]
                            + sleep_summary[constants.SLEEP_SUMMARY_DURATION_IN_MS_COL]
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
            # Keep only those belonging to sleep summaries
            sleep_stages = sleep_stages[
                sleep_stages[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL].isin(
                    sleep_summary[constants.SLEEP_SUMMARY_ID_COL].unique()
                )
            ]
            user_statistics_list = []
            for statistic in statistics:
                # Get series with metrics by calendarDate
                ser: pd.Series = SLEEP_STATISTICS_DICT[statistic](
                    sleep_summary=sleep_summary,
                    sleep_stages=sleep_stages,
                    chronotype=user_chronotype,
                )

                user_statistics_list.append(ser.rename(statistic))
            user_frames[user] = pd.concat(user_statistics_list, axis=1)
    if not user_frames:
        return {} if not return_df else pd.DataFrame(columns=statistics)
    sleep_stats_df = pd.concat(
        user_frames, names=[constants.USER_COL, constants.CALENDAR_DATE_COL]
    )

    # Perform kind operation by user
    if not (kind is None):
        # For bedtime, wakeup, and midpoint columns we need to use a different
        # approach
        aggregated_sleep_stats_df = (
            sleep_stats_df.drop(
                columns=SLEEP_DATETIME_STATISTICS + ["validation"], errors="ignore"
            )
            .groupby(level=constants.USER_COL)
            .agg(kind, *kind_args, **kind_kwargs)
        )
        for statistic in SLEEP_DATETIME_STATISTICS:
            if statistic in sleep_stats_df:
                if kind in SLEEP_KIND_MAPPING.keys():
                    mapping_function = SLEEP_KIND_MAPPING[kind][statistic]
                    # We need to check that we have a valid kind operation
                else:
                    warnings.warn(
                        f"Kind operation {kind} is not supported for datetime values. Please choose one among {[x for x in SLEEP_KIND_MAPPING.keys()]}"
                    )
                    mapping_function = lambda x: np.nan
                temp_col = (
                    sleep_stats_df[statistic]
                    .groupby(level=constants.USER_COL)
                    .apply(mapping_function)
                )

                aggregated_sleep_stats_df = pd.concat(
                    (aggregated_sleep_stats_df, temp_col), axis=1
                )
        return aggregated_sleep_stats_df if return_df else aggregated_sleep_stats_df.to_dict(orient="index")

    if not return_df:
        result = {}
        for user, frame in user_frames.items():
            if len(statistics) == 1:
                result[user] = {
                    pd.Timestamp(date).date(): value
                    for date, value in frame[statistics[0]].items()
                }
            else:
                result[user] = {
                    pd.Timestamp(date).date(): values
                    for date, values in frame.to_dict(orient="index").items()
                }
        return result

    if not return_multi_index:
        return sleep_stats_df.reset_index()

    # Return series if only statistic
    if len(statistics) == 1:
        return sleep_stats_df[statistics[0]]
    else:
        return sleep_stats_df
