"""
This module contains all the functions related to the analysis
of heart rate data.
"""

import datetime
import warnings
from typing import Union

import pandas as pd

from . import heart_rate_schema
from .loader.base import BaseLoader
from .utils import time as time_utils

HEART_RATE_STATISTIC_COUNT = "heart_rate_count"
HEART_RATE_STATISTIC_MAX = "heart_rate_max"
HEART_RATE_STATISTIC_MEAN = "heart_rate_mean"
HEART_RATE_STATISTIC_MEDIAN = "heart_rate_median"
HEART_RATE_STATISTIC_MIN = "heart_rate_min"
HEART_RATE_STATISTIC_P10 = "heart_rate_p10"
HEART_RATE_STATISTIC_P90 = "heart_rate_p90"
HEART_RATE_STATISTIC_REST = "heart_rate_rest"

HEART_RATE_STATISTICS = [
    HEART_RATE_STATISTIC_COUNT,
    HEART_RATE_STATISTIC_MAX,
    HEART_RATE_STATISTIC_MEAN,
    HEART_RATE_STATISTIC_MEDIAN,
    HEART_RATE_STATISTIC_MIN,
    HEART_RATE_STATISTIC_P10,
    HEART_RATE_STATISTIC_P90,
    HEART_RATE_STATISTIC_REST,
]


def get_daily_rest_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> pd.Series:
    """Get daily rest heart rate (RHR).

    This function computes the resting heart rate statistic.
    Rest heart rate is computed as the lowest
    30 minute average over a period of a calendar day.
    Rest heart rate is reported for each day
    from ``start_date`` to ``end_date``and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a :class:`pandas.Series` with a :class:`pandas.MultiIndex`
    as index, with the two levels of the index being
    ``user_id`` and ``local_time``::

        user_id         local_time
        user_id_0       2023-05-18      41.0
                        2023-05-19      40.0
                        2023-05-20      37.0
        Name: heart_rate_rest, dtype: float64

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which RHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series`
        Rest heart rate at the daily level.
    """

    return get_heart_rate_statistics(
        loader=loader,
        user_id=user_id,
        statistic=HEART_RATE_STATISTIC_REST,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
        resolution="daily",
        time_reference=time_reference,
    )


def get_daily_max_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> pd.Series:
    """Get daily maximum heart rate.

    This function computes the maximum heart rate.
    Maximum heart rate is computed as the maximum
    heart rate value over a period of a calendar day.
    Maximum heart rate is reported for each day
    from ``start_date`` to ``end_date`` (exclusive) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    he function returns a :class:`pandas.Series` with a :class:`pandas.MultiIndex`
    as index, with the two levels of the index being
    ``user_id`` and ``local_time``::

        user_id         local_time
        userId0         2023-05-18      155.0
                        2023-05-19      132.0
                        2023-05-20      91.0
        Name: heart_rate_max, dtype: float64


    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which maxHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series`
        Daily maximum heart rate.
    """

    return get_heart_rate_statistics(
        loader=loader,
        user_id=user_id,
        statistic=HEART_RATE_STATISTIC_MAX,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
        resolution="daily",
        time_reference=time_reference,
    )


def get_daily_min_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
    time_reference: str = "auto",
) -> pd.Series:
    """Get daily minimum heart rate.

    This function computes the minimum heart rate statistic.
    Minimum heart rate is computed as the minimum
    heart rate value over a period of a calendar day.
    Minimum heart rate is reported for each day
    from ``start_date`` to ``end_date`` (excluded) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a
    :class:`pandas.Series` with a :class:`pandas.MultiIndex` as index,
    with the two levels of the index being
    ``user_id`` and ``local_time``::

        user_id         local_time
        user_id_0       2023-05-18      38.0
                        2023-05-19      38.0
                        2023-05-20      34.0
        Name: heart_rate_min, dtype: float64


    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which minHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series`
        Daily minimum heart rate.
    """
    return get_heart_rate_statistics(
        loader=loader,
        user_id=user_id,
        statistic=HEART_RATE_STATISTIC_MIN,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def get_daily_mean_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
    time_reference: str = "auto",
) -> pd.Series:
    """Get daily mean heart rate.

    This function computes the mean heart rate statistic.
    Minimum heart rate is computed as the mean
    heart rate value over a period of a calendar day.
    Mean heart rate is reported for each day
    from ``start_date`` to ``end_date`` (excluded) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a
    :class:`pandas.Series` with a :class:`pandas.MultiIndex` as index,
    with the two levels of the index being
    ``user_id`` and ``local_time``::

        user_id         local_time
        user_id_0       2023-05-18      52.0
                        2023-05-19      54.0
                        2023-05-20      55.0
        Name: heart_rate_mean, dtype: float64


    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which minHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series`
        Daily mean heart rate.
    """
    return get_heart_rate_statistics(
        loader=loader,
        user_id=user_id,
        statistic=HEART_RATE_STATISTIC_MEAN,
        start_date=start_date,
        end_date=end_date,
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def _compute_hr_statistics(
    heart_rate: pd.DataFrame, rule: str, time_col: str, calendar_based: bool
) -> pd.Series:
    """Computes a heart rate statistics on a daily basis.

    This function computes a specific statistic for
    the heart rate data available at a daily basis.

    Parameters
    ----------
    heart_rate : :class:`pandas.DataFrame`
        Heart rate data.
    statistic : :class:`str`
        Name of the statistic to be computed

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with time (utc and local) as index, and statistics as columns
    """

    if not isinstance(heart_rate, pd.DataFrame):
        raise ValueError(
            f"heart_rate must be a pd.DataFrame. {type(heart_rate)} is not a valid type."
        )

    required_columns = {time_col, heart_rate_schema.HEART_RATE_COL}
    missing_columns = required_columns.difference(heart_rate.columns)
    if missing_columns:
        raise ValueError(
            "heart_rate is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    heart_rate = heart_rate.dropna(subset=list(required_columns)).copy()
    if heart_rate.empty:
        return pd.DataFrame()

    conflict_col = heart_rate_schema.TIMESTAMP_CONFLICT_COL
    if conflict_col in heart_rate.columns and heart_rate[conflict_col].any():
        warnings.warn("Excluding ambiguous same-timestamp heart-rate readings from aggregation.")
        heart_rate = heart_rate.loc[~heart_rate[conflict_col]].copy()
    if heart_rate.empty:
        return pd.DataFrame()

    # Every statistic must use the same set of physical instants. Loaders may
    # retain repeated, identical exports, so collapse those timestamps once
    # before computing both ordinary statistics and quantiles.
    duplicate_time_col = (
        heart_rate_schema.UTC_TIME_COL
        if heart_rate_schema.UTC_TIME_COL in heart_rate.columns
        and heart_rate[heart_rate_schema.UTC_TIME_COL].notna().any()
        else time_col
    )
    heart_rate = heart_rate.drop_duplicates(subset=duplicate_time_col)

    heart_rate_stats = (
        heart_rate.resample(rule=rule, on=time_col, label="left", closed="left")[
            heart_rate_schema.HEART_RATE_COL
        ]
        .agg(["count", "max", "min", "mean", "median", "std"])
        .astype("float")
        .add_prefix("heart_rate_")
    )

    # TODO see how we can implement heart rate wearing time
    # given that we have different source resolutions
    # for heart rate
    heart_rate_wearing_time = (
        heart_rate.resample(rule="1min", on=time_col, label="left", closed="left")[
            heart_rate_schema.HEART_RATE_COL
        ]
        .count()
        .gt(0)
        .resample(rule=rule, label="left", closed="left")
        .sum()
        #  / rule_seconds
        # * 5
    )  # * pd.Timedelta(rule)

    heart_rate_quantiles = (
        heart_rate.resample(rule=rule, on=time_col, label="left", closed="left")[
            heart_rate_schema.HEART_RATE_COL
        ]
        .quantile([0.1, 0.9])
        .unstack(level=-1)
    )
    # Rename columns from 0.10, 0.90 to p10, p90
    heart_rate_quantiles.columns = [
        f"heart_rate_p{int(q*100)}" for q in heart_rate_quantiles.columns
    ]
    heart_rate_stats = pd.concat((heart_rate_stats, heart_rate_quantiles), axis=1)

    # Let's add back timezone info
    if (
        not calendar_based
        and time_col == heart_rate_schema.UTC_TIME_COL
        and heart_rate_schema.TIMEZONE_COL in heart_rate.columns
    ):
        tz_lookup = (
            heart_rate[
                [heart_rate_schema.UTC_TIME_COL, heart_rate_schema.TIMEZONE_COL]
            ]
            .sort_values(heart_rate_schema.UTC_TIME_COL)
            .drop_duplicates(subset=heart_rate_schema.UTC_TIME_COL, keep="last")
        )
        heart_rate_stats = heart_rate_stats.reset_index()

        heart_rate_stats = pd.merge_asof(
            heart_rate_stats.sort_values("utc_time"),
            tz_lookup,
            on=heart_rate_schema.UTC_TIME_COL,
            direction="nearest",
        )
        heart_rate_stats = heart_rate_stats.set_index(
            [heart_rate_schema.UTC_TIME_COL, heart_rate_schema.TIMEZONE_COL]
        )

    # We can extract rest heart rate only if resolution is at
    # daily level, otherwise it doesn't have any sense
    if rule == "d":
        hr_one_min_agg = (
            heart_rate.sort_values(time_col)
            .set_index(time_col)
            .resample("1min", closed="left", label="left")[
                heart_rate_schema.HEART_RATE_COL
            ]
            .mean()
        )
        # Perform 30-minute average and get the minimum by day
        rest_hr = (
            hr_one_min_agg.groupby(pd.Grouper(freq="D"))
            .apply(
                lambda x: x.rolling("30min", min_periods=20, closed="left").mean().min()
            )
            .rename("heart_rate_rest")
        )
        heart_rate_stats = pd.concat((heart_rate_stats, rest_hr), axis=1)
    return heart_rate_stats


def get_minute_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a minute level.

    This function computes the specified heart rate statistics
    over periods of one minute.
    Aggregations are performed at ``UTC`` level,
    and time-zone is reported to recover
    local time, if needed.
    Statistics are reported for each minute
    from ``start_date`` to ``end_date`` (excluded) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a
    :class:`pandas.Series` or :class:`pandas.DataFrame` with a :class:`pandas.MultiIndex` as index,
    with the two levels of the index being
    ``user_id`` and ``utc_time``.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    statistic: :class:`str`
        The list of heart rate statistics to be computed.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which minHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series` or :class:`pandas.DataFrame`
        Minute-level heart rate statistics.
    """
    return get_heart_rate_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="60s",
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def get_hourly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a hourly level.

    This function computes the specified heart rate statistics
    over periods of one hour.
    Aggregations are performed at ``UTC`` level,
    and time-zone is reported to recover
    local time, if needed.
    Statistics are reported for each hour
    from ``start_date`` to ``end_date`` (excluded) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a
    :class:`pandas.Series` or :class:`pandas.DataFrame` with a :class:`pandas.MultiIndex` as index,
    with the two levels of the index being
    ``user_id`` and ``utc_time``.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    statistic: :class:`str`
        The list of heart rate statistics to be computed.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which minHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series` or :class:`pandas.DataFrame`
        Hour-level heart rate statistics.
    """
    return get_heart_rate_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="1h",
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def get_daily_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a daily level.

    This function computes the specified heart rate statistics
    over periods of one day.
    Aggregations are performed using local
    time as time source.
    Statistics are reported for each day
    from ``start_date`` to ``end_date`` (excluded) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a
    :class:`pandas.Series` or :class:`pandas.DataFrame` with a :class:`pandas.MultiIndex` as index,
    with the two levels of the index being
    ``user_id`` and ``local_time``.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    statistic: :class:`str`
        The list of heart rate statistics to be computed.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which minHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series` or :class:`pandas.DataFrame`
        Daily-level heart rate statistics.
    """
    return get_heart_rate_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="daily",
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def get_weekly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a weekly level.

    This function computes the specified heart rate statistics
    over periods of one week.
    Aggregations are performed using local
    time as time source.
    Statistics are reported for each calendar week
    from ``start_date`` to ``end_date`` (excluded) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a
    :class:`pandas.Series` or :class:`pandas.DataFrame` with a :class:`pandas.MultiIndex` as index,
    with the two levels of the index being
    ``user_id`` and ``local_time``.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    statistic: :class:`str`
        The list of heart rate statistics to be computed.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which minHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series` or :class:`pandas.DataFrame`
        Week-level heart rate statistics.
    """
    return get_heart_rate_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="weekly",
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def get_monthly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a minute level.

    This function computes the specified heart rate statistics
    over periods of one minute.
    Aggregations are performed at ``UTC`` level,
    and time-zone is reported to recover
    local time, if needed.
    Statistics are reported for each minute
    from ``start_date`` to ``end_date`` (excluded) and
    for the users specified by ``user_id``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.
    The function returns a
    :class:`pandas.Series` or :class:`pandas.DataFrame` with a :class:`pandas.MultiIndex` as index,
    with the two levels of the index being
    ``user_id`` and ``local_time``.


    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    statistic: :class:`str`
        The list of heart rate statistics to be computed.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which minHR must be retrieved, by default "all"
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.

    Returns
    -------
    :class:`pandas.Series` or :class:`pandas.DataFrame`
        Monthly-level heart rate statistics.
    """
    return get_heart_rate_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="monthly",
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def get_yearly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> Union[pd.Series, pd.DataFrame]:
    return get_heart_rate_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="yearly",
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
    )


def get_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    resolution: Union[pd.Timedelta, str] = "day",
    loader_kwargs: dict = {},
    time_reference: str = "auto",
) -> Union[pd.Series, pd.DataFrame]:
    """Get all the available statistics based on heart rate.

    This function computes statistics
    based on heart rate data.
    The statistics are reported for each day
    from ``start_date`` to ``end_date`` and
    for the users specified by ``user_id``.
    The type of statistic to be computed can be set using
    the ``statistic`` parameter. Available options are:

        - ``"heart_rate_count"`` for the count of heart rate samples
        - ``"heart_rate_max"`` for maximum heart rate
        - ``"heart_rate_mean"`` for mean heart rate
        - ``"heart_rate_min"`` for minimum heart rate
        - ``"heart_rate_rest"`` for rest heart rate (only available
            if ``resolution`` is ``"day"``)
        - ``"heart_rate_std"`` for heart rate standard deviation
        - ``"heart_rate_p10"`` for the 10-th percentile of heart rate
        - ``"heart_rate_p90"`` for the 90-th percentile of heart rate


    Both single statistics and multiple statistics can
    be computed. If all the available statistics have
    to be computed, then the ``statistic`` parameter
    can be set to ``None``.
    The function used to load heart rate data is the
    :func:`~loader.BaseLoader.load_heart_rate`. If
    loader-specific arguments are required, it is
    possible to set them using the ``loader_kwargs`` argument.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which the heart rate statistic must be retrieved, by default "all".
    statistic : :class:`str` or :class:`list` or ```None```.
        Statistic(s) to be computed. If ``None``, all heart rate statistics are computed.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    loader_kwargs: :class:`dict`
        Keyword arguemnts for the ``load_heart_rate`` loading function of the ``loader``,
        by default no keyword arguments are passed.
    resolution: :class:`str`
        The desired temporal resolution at which statistics need to be
        extracted. For sub-daily temporal resolutions, any value that
        can be accepted by :fun:`pandas.resample` works. For daily, weekly,
        monthly, or yearly resolutions, the resolution must be
        specified as ``"daily"``, ``"weekly"``, ``"monthly"`` or ``"yearly"``.
    time_reference: :class:`str`
        Clock used to construct windows. ``"auto"`` uses UTC for fixed-duration
        subdaily windows and local time for calendar resolutions. ``"utc"`` and
        ``"local"`` override that default.

    Returns
    -------
    :class:`pandas.DataFrame` or :class:`pandas.Series`
        Heart rate statistics at the defined resolution.
    """
    # Get user id from the loader
    if user_id == "all" or user_id is None:
        user_id = loader.get_user_ids()
    if type(user_id) is str:
        user_id = [user_id]

    # Convert to datetime
    start_date = None if start_date is None else pd.Timestamp(start_date)
    end_date = None if end_date is None else pd.Timestamp(end_date)

    # Get specification for resampling strategy
    resampling_spec = time_utils.get_resampling_spec(
        resolution=resolution, time_reference=time_reference
    )
    time_col = resampling_spec.time_col
    rule = resampling_spec.rule
    calendar_based = resampling_spec.calendar_based

    available_time_columns = getattr(
        loader, "heart_rate_time_columns", heart_rate_schema.TIME_COLUMNS
    )
    if time_col not in available_time_columns:
        if (
            time_reference == "auto"
            and heart_rate_schema.LOCAL_TIME_COL in available_time_columns
        ):
            warnings.warn(
                "UTC heart-rate timestamps are unavailable; using local-time "
                "windows instead."
            )
            time_col = heart_rate_schema.LOCAL_TIME_COL
        else:
            raise ValueError(
                f"The loader does not provide heart-rate timestamps in {time_col}."
            )

    # Check statistic
    if statistic is None:
        statistics = [
            requested_statistic
            for requested_statistic in HEART_RATE_STATISTICS
            if requested_statistic != HEART_RATE_STATISTIC_REST or rule == "d"
        ]
    elif type(statistic) == str:
        statistics = [statistic]
    else:
        statistics = statistic

    for requested_statistic in statistics:
        if requested_statistic not in HEART_RATE_STATISTICS:
            raise ValueError(
                f"statistic must be one among {HEART_RATE_STATISTICS}, not {requested_statistic}"
            )
        if requested_statistic == HEART_RATE_STATISTIC_REST and rule != "d":
            raise ValueError(
                "heart_rate_rest is available only at daily resolution"
            )
    # Check args and kwargs
    if type(loader_kwargs) != dict:
        raise ValueError(
            f"loader_kwargs must be of type dict, not {type(loader_kwargs)}"
        )

    start_date = heart_rate_schema.normalize_time_bound(start_date, time_col)
    end_date = heart_rate_schema.normalize_time_bound(end_date, time_col)

    heart_rate_stats_df_list: list[pd.DataFrame] = []
    for user in user_id:

        # TODO we may need to split the computation in chuncks if
        # window is too long
        heart_rate = loader.load_heart_rate(
            user_id=user,
            start_date=(
                None if start_date is None else start_date - pd.Timedelta(days=2)
            ),
            end_date=None if end_date is None else end_date + pd.Timedelta(days=2),
            time_col=time_col,
            **loader_kwargs,
        )
        if len(heart_rate) > 0:
            heart_rate = heart_rate_schema.filter_by_time(
                heart_rate,
                start_date=start_date,
                end_date=end_date,
                time_col=time_col,
            )

            user_heart_rate_stats = _compute_hr_statistics(
                heart_rate, rule=rule, time_col=time_col, calendar_based=calendar_based
            )
            if len(user_heart_rate_stats) > 0:
                if calendar_based and start_date is not None:
                    canonical_start = time_utils.get_calendar_period_start(
                        start_date, rule
                    )
                    if canonical_start != start_date:
                        user_heart_rate_stats = user_heart_rate_stats.rename(
                            index={canonical_start: start_date}
                        )
                heart_rate_stats_df_list.append(
                    user_heart_rate_stats.assign(user_id=user)
                )

    if len(heart_rate_stats_df_list) > 0:
        heart_rate_stats_df = pd.concat(heart_rate_stats_df_list, axis=0)
        index_levels = heart_rate_stats_df.index.names
        heart_rate_stats_df = heart_rate_stats_df.set_index(
            "user_id", append=True
        ).reorder_levels(["user_id"] + index_levels)

        # Reindex based on start/end dates
        if not (start_date is None) or (not (end_date is None)):
            grid_start = (
                start_date
                if start_date is not None
                else heart_rate_stats_df.index.get_level_values(level=1).min()
            )
            grid_end = (
                heart_rate_stats_df.index.get_level_values(level=1).max()
                if end_date is None
                else end_date
            )
            if calendar_based:
                new_idx = time_utils.get_calendar_window_starts(
                    start=grid_start,
                    end=grid_end,
                    rule=rule,
                    include_end=end_date is None,
                )
            else:
                new_idx = pd.date_range(
                    start=grid_start.floor(rule),
                    end=grid_end,
                    inclusive="left" if end_date is not None else "both",
                    freq=rule,
                )

            index_time_name = time_col
            has_timezone = "timezone" in heart_rate_stats_df.index.names
            if not calendar_based and has_timezone:
                tz = heart_rate_stats_df.reset_index(level="timezone").reindex(
                    pd.MultiIndex.from_product(
                        [user_id, new_idx], names=["user_id", index_time_name]
                    )
                )["timezone"]
                heart_rate_stats_df = heart_rate_stats_df.droplevel(["timezone"])

            heart_rate_stats_df = heart_rate_stats_df.reindex(
                pd.MultiIndex.from_product(
                    [user_id, new_idx], names=["user_id", index_time_name]
                )
            )

            if not calendar_based and has_timezone:
                heart_rate_stats_df = pd.concat(
                    (heart_rate_stats_df, tz), axis=1
                ).set_index("timezone", append=True)

        # Return series if only statistic
        if len(statistics) == 1:
            return heart_rate_stats_df[statistics[0]]
        else:
            return heart_rate_stats_df[list(statistics)]
    else:
        if (not (start_date is None)) and (not (end_date is None)):
            if calendar_based:
                new_idx = time_utils.get_calendar_window_starts(
                    start=start_date,
                    end=end_date,
                    rule=rule,
                )
            else:
                new_idx = pd.date_range(
                    start=start_date.floor(rule),
                    end=end_date,
                    inclusive="left",
                    freq=rule,
                )
            return pd.DataFrame(
                columns=statistics,
                index=pd.MultiIndex.from_product([user_id, new_idx]),
            )[statistics].squeeze(axis=1)
        else:
            return pd.DataFrame(
                columns=statistics,
                index={"user_id": user_id},
            )[
                statistics
            ].squeeze(axis=1)
