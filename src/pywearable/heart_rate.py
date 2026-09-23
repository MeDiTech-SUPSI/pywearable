"""
This module contains all the functions related to the analysis
of heart rate data.
"""

import datetime
import warnings
from typing import Union

import numpy as np
import pandas as pd

from . import cycle, heart_rate_schema, hierarchy
from .loader.base import BaseLoader
from .utils import time as time_utils

HEART_RATE_STATISTIC_COUNT = "heart_rate_count"
HEART_RATE_STATISTIC_MAX = "heart_rate_max"
HEART_RATE_STATISTIC_MEAN = "heart_rate_mean"
HEART_RATE_STATISTIC_MEDIAN = "heart_rate_median"
HEART_RATE_STATISTIC_MIN = "heart_rate_min"
HEART_RATE_STATISTIC_P10 = "heart_rate_p10"
HEART_RATE_STATISTIC_P90 = "heart_rate_p90"
HEART_RATE_STATISTIC_STD = "heart_rate_std"
HEART_RATE_STATISTIC_REST = "heart_rate_rest"

HEART_RATE_STATISTICS = [
    HEART_RATE_STATISTIC_COUNT,
    HEART_RATE_STATISTIC_MAX,
    HEART_RATE_STATISTIC_MEAN,
    HEART_RATE_STATISTIC_MEDIAN,
    HEART_RATE_STATISTIC_MIN,
    HEART_RATE_STATISTIC_P10,
    HEART_RATE_STATISTIC_P90,
    HEART_RATE_STATISTIC_STD,
    HEART_RATE_STATISTIC_REST,
]

HEART_RATE_REDUCTIONS = {
    HEART_RATE_STATISTIC_COUNT: "count",
    HEART_RATE_STATISTIC_MAX: "max",
    HEART_RATE_STATISTIC_MEAN: "mean",
    HEART_RATE_STATISTIC_MEDIAN: "median",
    HEART_RATE_STATISTIC_MIN: "min",
    HEART_RATE_STATISTIC_STD: "std",
}
HEART_RATE_QUANTILES = {
    HEART_RATE_STATISTIC_P10: 0.1,
    HEART_RATE_STATISTIC_P90: 0.9,
}


def get_daily_rest_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
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
        min_coverage=min_coverage,
    )


def get_daily_max_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
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
        min_coverage=min_coverage,
    )


def get_daily_min_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
    time_reference: str = "auto",
    min_coverage=None,
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
        min_coverage=min_coverage,
    )


def get_daily_mean_heart_rate(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    loader_kwargs={},
    time_reference: str = "auto",
    min_coverage=None,
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
        min_coverage=min_coverage,
    )


def _prepare_heart_rate_samples(
    heart_rate: pd.DataFrame, time_col: str
) -> pd.DataFrame:
    """Validate and deduplicate samples before any heart-rate calculation."""
    if not isinstance(heart_rate, pd.DataFrame):
        raise ValueError(
            "heart_rate must be a pd.DataFrame. "
            f"{type(heart_rate)} is not a valid type."
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
        return heart_rate

    conflict_col = heart_rate_schema.TIMESTAMP_CONFLICT_COL
    if conflict_col in heart_rate.columns and heart_rate[conflict_col].any():
        warnings.warn(
            "Excluding ambiguous same-timestamp heart-rate readings from aggregation."
        )
        heart_rate = heart_rate.loc[~heart_rate[conflict_col]].copy()
    if heart_rate.empty:
        return heart_rate

    # Every statistic must use the same set of physical instants. Loaders may
    # retain repeated, identical exports, so collapse those timestamps once
    # before computing both ordinary statistics and quantiles.
    duplicate_time_col = (
        heart_rate_schema.UTC_TIME_COL
        if heart_rate_schema.UTC_TIME_COL in heart_rate.columns
        and heart_rate[heart_rate_schema.UTC_TIME_COL].notna().any()
        else time_col
    )
    return heart_rate.drop_duplicates(subset=duplicate_time_col)


def _infer_heart_rate_cadence(timestamps) -> Union[pd.Timedelta, None]:
    """Infer a nominal cadence when one clustered difference is prevalent."""
    timestamps = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna()
    differences = (
        timestamps.drop_duplicates()
        .sort_values()
        .diff()
        .dt.total_seconds()
        .dropna()
    )
    differences = differences[differences > 0]
    if len(differences) < 10:
        return None

    values = differences.to_numpy(dtype=float)
    rounded = np.where(values < 10, np.round(values, 2), np.round(values))
    mode = float(pd.Series(rounded).value_counts().index[0])
    tolerance = max(0.05, mode * 0.1)
    cluster = values[np.abs(values - mode) <= tolerance]
    if len(cluster) / len(values) < 0.2:
        return None

    cadence = float(np.median(cluster))
    median_difference = float(np.median(values))
    if not 0.5 <= cadence / median_difference <= 2:
        return None
    return pd.Timedelta(seconds=cadence)


def _heart_rate_cadences(heart_rate: pd.DataFrame, time_col: str):
    """Return the optional source column and inferred cadence per stream."""
    source_col = "source" if "source" in heart_rate.columns else None
    if source_col is None or heart_rate[source_col].isna().all():
        return None, {"heart_rate": _infer_heart_rate_cadence(heart_rate[time_col])}

    streams = heart_rate[source_col].astype("string").fillna("unknown")
    cadences = {
        source: _infer_heart_rate_cadence(heart_rate.loc[streams.eq(source), time_col])
        for source in streams.unique()
    }
    return source_col, cadences


def _coverage_window_label(value, resolution):
    """Return the canonical primary-window label containing ``value``."""
    value = pd.Timestamp(value)
    canonical = hierarchy.canonical_resolution(resolution)
    if canonical in hierarchy.CALENDAR_RESOLUTIONS:
        spec = time_utils.get_resampling_spec(canonical, time_reference="local")
        return time_utils.get_calendar_period_start(value, spec.rule)
    return value.floor(canonical)


def _primary_coverage_context(
    heart_rate: pd.DataFrame,
    resolution,
    time_col: str,
    source_col,
    cadences,
):
    """Pre-aggregate valid samples by primary window and source."""
    if heart_rate.empty:
        return {}, cadences

    streams = (
        pd.Series("heart_rate", index=heart_rate.index, dtype="string")
        if source_col is None
        else heart_rate[source_col].astype("string").fillna("unknown")
    )
    labels = heart_rate[time_col].map(
        lambda value: _coverage_window_label(value, resolution)
    )
    counts = (
        pd.DataFrame(
            {
                "window_start": labels,
                "stream": streams,
                "timestamp": heart_rate[time_col],
            }
        )
        .groupby(["window_start", "stream"], sort=False)["timestamp"]
        .nunique()
    )
    by_window = {
        window_start: rows.droplevel("window_start").to_dict()
        for window_start, rows in counts.groupby(level="window_start", sort=False)
    }
    return by_window, cadences


def _heart_rate_window_coverage(
    counts,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cadences,
):
    """Return observed/expected samples and unresolved stream labels."""
    if len(cadences) == 1:
        stream = next(iter(cadences))
        cadence = cadences[stream]
        unresolved = {stream} if cadence is None else set()
    else:
        if not counts:
            streams = set(cadences)
        else:
            streams = set(counts)
        unresolved = {stream for stream in streams if cadences.get(stream) is None}
        known = {cadences[stream] for stream in streams if stream not in unresolved}
        if unresolved or len(known) != 1:
            return np.nan, unresolved or {"mixed_cadence"}
        cadence = next(iter(known))

    if cadence is None or end <= start:
        return np.nan, unresolved
    expected = (end - start) / cadence
    if expected <= 0:
        return np.nan, unresolved
    observed = sum(counts.values())
    return min(1.0, observed / expected), set()


def _heart_rate_period_end(start, resolution):
    canonical = hierarchy.canonical_resolution(resolution)
    if canonical in hierarchy.CALENDAR_RESOLUTIONS:
        spec = time_utils.get_resampling_spec(canonical, time_reference="local")
        period_start = time_utils.get_calendar_period_start(start, spec.rule)
        if spec.rule == "d":
            return period_start + pd.Timedelta(days=1)
        return pd.date_range(period_start, periods=2, freq=spec.rule)[1]
    return start + pd.Timedelta(canonical)


def _attach_primary_heart_rate_coverage(
    statistics_frame,
    contexts,
    statistics,
    resolution,
    time_col,
    start_date,
    end_date,
    min_coverage,
    unresolved_streams,
):
    """Attach raw-to-primary coverage and apply its optional threshold."""
    coverage = []
    for index in statistics_frame.index:
        index = index if isinstance(index, tuple) else (index,)
        user, window_start = index[:2]
        window_start = pd.Timestamp(window_start)
        window_end = _heart_rate_period_end(window_start, resolution)
        physical_start = (
            max(window_start, start_date)
            if start_date is not None
            else window_start
        )
        physical_end = min(window_end, end_date) if end_date is not None else window_end
        context = contexts.get(user)
        if context is None:
            value, unresolved = np.nan, {"heart_rate"}
        else:
            counts_by_window, cadences = context
            canonical_start = _coverage_window_label(window_start, resolution)
            value, unresolved = _heart_rate_window_coverage(
                counts_by_window.get(canonical_start, {}),
                physical_start,
                physical_end,
                cadences,
            )
        coverage.append(value)
        if min_coverage is not None and pd.isna(value):
            unresolved_streams.update((user, stream) for stream in unresolved)

    coverage = pd.Series(coverage, index=statistics_frame.index, dtype=float)
    for statistic in statistics:
        statistics_frame[f"{statistic}__coverage"] = coverage
    if min_coverage is not None:
        invalid = coverage.isna() | coverage.lt(min_coverage)
        statistics_frame.loc[invalid, statistics] = np.nan
    return statistics_frame


def _warn_unknown_primary_coverage(unresolved_streams):
    if unresolved_streams:
        warnings.warn(
            "Primary HR coverage could not be determined for "
            f"{len(unresolved_streams)} user/source stream(s); features requiring "
            "a primary coverage threshold were excluded."
        )


def _resting_heart_rate(heart_rate: pd.DataFrame, time_col: str):
    """Return the minimum 30-minute mean with at least 20 observed minutes."""
    if heart_rate.empty:
        return np.nan
    minute = (
        heart_rate.sort_values(time_col)
        .set_index(time_col)[heart_rate_schema.HEART_RATE_COL]
        .resample("1min", closed="left", label="left")
        .mean()
    )
    return minute.rolling("30min", min_periods=20, closed="left").mean().min()


def _summarize_heart_rate_values(values: pd.Series, statistics) -> dict:
    """Calculate requested scalar reductions from one interval's values."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    result = {}
    for statistic in statistics:
        if statistic in HEART_RATE_REDUCTIONS:
            result[statistic] = float(
                getattr(values, HEART_RATE_REDUCTIONS[statistic])()
            )
        elif statistic in HEART_RATE_QUANTILES:
            result[statistic] = values.quantile(HEART_RATE_QUANTILES[statistic])
    return result


def _compute_hr_statistics(
    heart_rate: pd.DataFrame,
    statistics,
    rule: str,
    time_col: str,
    calendar_based: bool,
) -> pd.DataFrame:
    """Compute heart-rate statistics over one resampling rule."""
    if heart_rate.empty:
        return pd.DataFrame()

    grouped = heart_rate.resample(
        rule=rule, on=time_col, label="left", closed="left"
    )[heart_rate_schema.HEART_RATE_COL]
    frames = []
    reductions = [
        HEART_RATE_REDUCTIONS[statistic]
        for statistic in statistics
        if statistic in HEART_RATE_REDUCTIONS
    ]
    if reductions:
        reduced = grouped.agg(reductions).astype("float")
        reduced = reduced.rename(
            columns={operation: f"heart_rate_{operation}" for operation in reductions}
        )
        frames.append(reduced)

    quantile_statistics = [
        statistic for statistic in statistics if statistic in HEART_RATE_QUANTILES
    ]
    if quantile_statistics:
        quantiles = [
            HEART_RATE_QUANTILES[statistic]
            for statistic in quantile_statistics
        ]
        quantile_frame = grouped.quantile(quantiles).unstack(level=-1)
        quantile_frame.columns = quantile_statistics
        frames.append(quantile_frame)

    if HEART_RATE_STATISTIC_REST in statistics:
        rest = pd.Series(
            {
                label: _resting_heart_rate(frame, time_col)
                for label, frame in heart_rate.groupby(
                    pd.Grouper(key=time_col, freq="D")
                )
            },
            name=HEART_RATE_STATISTIC_REST,
            dtype=float,
        )
        frames.append(rest.to_frame())

    heart_rate_stats = pd.concat(frames, axis=1).reindex(columns=statistics)

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

    return heart_rate_stats


def get_minute_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a minute level.

    This function computes the specified heart rate statistics
    over periods of one minute.
    ``time_reference="auto"`` uses UTC windows; local windows can be requested
    explicitly.
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
        min_coverage=min_coverage,
    )


def get_hourly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a hourly level.

    This function computes the specified heart rate statistics
    over periods of one hour.
    ``time_reference="auto"`` uses UTC windows; local windows can be requested
    explicitly.
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
        min_coverage=min_coverage,
    )


def get_daily_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a daily level.

    This function computes the specified heart rate statistics
    over periods of one day.
    ``time_reference="auto"`` uses local calendar days.
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
        min_coverage=min_coverage,
    )


def get_weekly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a weekly level.

    This function computes the specified heart rate statistics
    over periods of one week.
    ``time_reference="auto"`` uses local calendar weeks.
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
        min_coverage=min_coverage,
    )


def get_monthly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
) -> Union[pd.Series, pd.DataFrame]:
    """Get heart rate statistics at a monthly level.

    This function computes the specified heart rate statistics
    over calendar months. ``time_reference="auto"`` uses local time.
    Statistics are reported for each month
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
        min_coverage=min_coverage,
    )


def get_yearly_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    min_coverage=None,
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
        min_coverage=min_coverage,
    )


def _heart_rate_statistics_list(statistic, allow_rest: bool):
    if statistic is None:
        return [
            value
            for value in HEART_RATE_STATISTICS
            if allow_rest or value != HEART_RATE_STATISTIC_REST
        ]
    statistics = [statistic] if isinstance(statistic, str) else list(statistic)
    unknown = set(statistics).difference(HEART_RATE_STATISTICS)
    if unknown:
        raise ValueError(
            f"statistic must be one among {HEART_RATE_STATISTICS}, "
            f"not {sorted(unknown)}"
        )
    if HEART_RATE_STATISTIC_REST in statistics and not allow_rest:
        raise ValueError(
            "heart_rate_rest is available only at daily resolution or cycle resolution"
        )
    return statistics


def _get_heart_rate_statistics_single_resolution(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    resolution: Union[pd.Timedelta, str] = "day",
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    primary_min_coverage=None,
    return_primary_coverage: bool = False,
    unresolved_streams=None,
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

    statistics = _heart_rate_statistics_list(statistic, allow_rest=rule == "d")
    # Check args and kwargs
    if type(loader_kwargs) != dict:
        raise ValueError(
            f"loader_kwargs must be of type dict, not {type(loader_kwargs)}"
        )

    start_date = heart_rate_schema.normalize_time_bound(start_date, time_col)
    end_date = heart_rate_schema.normalize_time_bound(end_date, time_col)
    if (
        start_date is not None
        and end_date is not None
        and end_date <= start_date
    ):
        raise ValueError("end_date must be after start_date")

    heart_rate_stats_df_list: list[pd.DataFrame] = []
    coverage_contexts = {}
    needs_coverage = return_primary_coverage or primary_min_coverage is not None
    unresolved_streams = set() if unresolved_streams is None else unresolved_streams
    for user in user_id:

        # TODO we may need to split the computation in chuncks if
        # window is too long
        loaded_heart_rate = loader.load_heart_rate(
            user_id=user,
            start_date=(
                None if start_date is None else start_date - pd.Timedelta(days=2)
            ),
            end_date=None if end_date is None else end_date + pd.Timedelta(days=2),
            time_col=time_col,
            **loader_kwargs,
        )
        if len(loaded_heart_rate) > 0:
            loaded_heart_rate = _prepare_heart_rate_samples(
                loaded_heart_rate, time_col
            )
            heart_rate = heart_rate_schema.filter_by_time(
                loaded_heart_rate,
                start_date=start_date,
                end_date=end_date,
                time_col=time_col,
            )
            if needs_coverage:
                source_col, cadences = _heart_rate_cadences(
                    loaded_heart_rate, time_col
                )
                coverage_contexts[user] = _primary_coverage_context(
                    heart_rate,
                    resolution,
                    time_col,
                    source_col,
                    cadences,
                )

            user_heart_rate_stats = _compute_hr_statistics(
                heart_rate,
                statistics=statistics,
                rule=rule,
                time_col=time_col,
                calendar_based=calendar_based,
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

        if needs_coverage:
            heart_rate_stats_df = _attach_primary_heart_rate_coverage(
                heart_rate_stats_df,
                coverage_contexts,
                statistics,
                resolution,
                time_col,
                start_date,
                end_date,
                primary_min_coverage,
                unresolved_streams,
            )
            columns = [
                column
                for statistic_name in statistics
                for column in (
                    statistic_name,
                    f"{statistic_name}__coverage",
                )
            ]
            return heart_rate_stats_df[columns]

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
            heart_rate_stats_df = pd.DataFrame(
                columns=statistics,
                index=pd.MultiIndex.from_product(
                    [user_id, new_idx], names=["user_id", time_col]
                ),
            )
            if needs_coverage:
                return _attach_primary_heart_rate_coverage(
                    heart_rate_stats_df,
                    coverage_contexts,
                    statistics,
                    resolution,
                    time_col,
                    start_date,
                    end_date,
                    primary_min_coverage,
                    unresolved_streams,
                )
            return heart_rate_stats_df[statistics].squeeze(axis=1)
        else:
            index = pd.MultiIndex.from_arrays(
                [[], []], names=["user_id", time_col]
            )
            result = pd.DataFrame(columns=statistics, index=index)
            if needs_coverage:
                for statistic_name in statistics:
                    result[f"{statistic_name}__coverage"] = pd.Series(
                        index=index, dtype=float
                    )
                return result
            return result[statistics].squeeze(axis=1)


def _interval_heart_rate_statistics(heart_rate, statistics):
    """Calculate only the requested statistics for one irregular interval."""
    result = _summarize_heart_rate_values(
        heart_rate[heart_rate_schema.HEART_RATE_COL], statistics
    )
    if HEART_RATE_STATISTIC_REST in statistics:
        result[HEART_RATE_STATISTIC_REST] = _resting_heart_rate(
            heart_rate, heart_rate_schema.LOCAL_TIME_COL
        )
    return result


def _cycle_heart_rate_trajectory(
    loader,
    users,
    statistics,
    cycle_type,
    start_date,
    end_date,
    loader_kwargs,
    cycle_kwargs,
    primary_min_coverage=None,
    return_primary_coverage=False,
    unresolved_streams=None,
):
    intervals = cycle.get_intervals(
        loader,
        cycle_type=cycle_type,
        user_id=users,
        start_date=start_date,
        end_date=end_date,
        **cycle_kwargs,
    )
    records = []
    needs_coverage = return_primary_coverage or primary_min_coverage is not None
    unresolved_streams = set() if unresolved_streams is None else unresolved_streams
    for user in users:
        if user not in intervals.index.get_level_values(0):
            continue
        user_intervals = intervals.xs(user)
        valid = user_intervals[user_intervals["interval_valid"]]
        heart_rate = pd.DataFrame(
            columns=[
                heart_rate_schema.LOCAL_TIME_COL,
                heart_rate_schema.HEART_RATE_COL,
            ]
        )
        source_col = None
        cadences = {"heart_rate": None}
        if not valid.empty:
            coverage_padding = (
                pd.Timedelta(days=2) if needs_coverage else pd.Timedelta(0)
            )
            heart_rate = loader.load_heart_rate(
                user_id=user,
                start_date=valid["start_time"].min() - coverage_padding,
                end_date=valid["end_time"].max() + coverage_padding,
                time_col=heart_rate_schema.LOCAL_TIME_COL,
                **loader_kwargs,
            )
            heart_rate = _prepare_heart_rate_samples(
                heart_rate, heart_rate_schema.LOCAL_TIME_COL
            ).sort_values(heart_rate_schema.LOCAL_TIME_COL, kind="stable")
            if needs_coverage:
                source_col, cadences = _heart_rate_cadences(
                    heart_rate, heart_rate_schema.LOCAL_TIME_COL
                )
        timestamps = pd.DatetimeIndex(
            heart_rate[heart_rate_schema.LOCAL_TIME_COL]
        )
        for label, interval in user_intervals.iterrows():
            values = {statistic: float("nan") for statistic in statistics}
            coverage = np.nan
            if interval["interval_valid"]:
                left = timestamps.searchsorted(interval["start_time"], side="left")
                right = timestamps.searchsorted(interval["end_time"], side="left")
                data = heart_rate.iloc[left:right]
                computed = _interval_heart_rate_statistics(data, statistics)
                values = {statistic: computed[statistic] for statistic in statistics}
                if needs_coverage:
                    if source_col is None:
                        counts = {
                            "heart_rate": data[
                                heart_rate_schema.LOCAL_TIME_COL
                            ].nunique()
                        }
                    else:
                        counts = (
                            data.assign(
                                __stream=data[source_col]
                                .astype("string")
                                .fillna("unknown")
                            )
                            .groupby("__stream")[heart_rate_schema.LOCAL_TIME_COL]
                            .nunique()
                            .to_dict()
                        )
                    coverage, unresolved = _heart_rate_window_coverage(
                        counts,
                        interval["start_time"],
                        interval["end_time"],
                        cadences,
                    )
                    if primary_min_coverage is not None and pd.isna(coverage):
                        unresolved_streams.update(
                            (user, stream) for stream in unresolved
                        )
                if primary_min_coverage is not None and (
                    pd.isna(coverage) or coverage < primary_min_coverage
                ):
                    values = {
                        statistic: float("nan") for statistic in statistics
                    }
            records.append(
                {
                    "user_id": user,
                    "window_start": label,
                    "period_start": interval["start_time"],
                    "period_end": interval["end_time"],
                    "interval_valid": bool(interval["interval_valid"]),
                    "end_source": interval["end_source"],
                    **values,
                    **(
                        {
                            f"{statistic}__coverage": coverage
                            for statistic in statistics
                        }
                        if return_primary_coverage
                        else {}
                    ),
                }
            )
    if not records:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=["user_id", "window_start"]
        )
        return pd.DataFrame(
            columns=[
                "period_start",
                "period_end",
                *statistics,
                *(
                    [f"{statistic}__coverage" for statistic in statistics]
                    if return_primary_coverage
                    else []
                ),
            ],
            index=index,
        )
    return pd.DataFrame(records).set_index(["user_id", "window_start"]).sort_index()


def _single_resolution_hr_trajectory(
    loader,
    users,
    statistics,
    start_date,
    end_date,
    resolution,
    loader_kwargs,
    time_reference,
    primary_min_coverage,
    unresolved_streams,
):
    result = _get_heart_rate_statistics_single_resolution(
        loader=loader,
        statistic=statistics,
        user_id=users,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
        primary_min_coverage=primary_min_coverage,
        return_primary_coverage=primary_min_coverage is not None,
        unresolved_streams=unresolved_streams,
    )
    if isinstance(result, pd.Series):
        result = result.to_frame()
    result = result.copy()
    if not isinstance(result.index, pd.MultiIndex) or result.index.nlevels < 2:
        index = pd.MultiIndex.from_arrays(
            [[], []], names=["user_id", "window_start"]
        )
        return pd.DataFrame(
            columns=["period_start", "period_end", *statistics], index=index
        )
    while result.index.nlevels > 2:
        result = result.droplevel(-1)
    result.index = result.index.set_names(["user_id", "window_start"])

    starts = pd.Series(
        result.index.get_level_values("window_start"), index=result.index
    )
    canonical = hierarchy.canonical_resolution(resolution)
    if canonical in hierarchy.CALENDAR_RESOLUTIONS:
        spec = time_utils.get_resampling_spec(canonical, time_reference="local")
        canonical_starts = starts.map(
            lambda value: time_utils.get_calendar_period_start(value, spec.rule)
        )
        if spec.rule == "d":
            ends = canonical_starts + pd.Timedelta(days=1)
        else:
            ends = canonical_starts.map(
                lambda value: pd.date_range(value, periods=2, freq=spec.rule)[1]
            )
    else:
        ends = starts + pd.Timedelta(canonical)
    period_starts = starts.copy()
    period_ends = pd.to_datetime(ends)
    if start_date is not None:
        requested_start = pd.Timestamp(start_date)
        period_starts = period_starts.map(
            lambda value: max(pd.Timestamp(value), requested_start)
        )
    if end_date is not None:
        requested_end = pd.Timestamp(end_date)
        period_ends = period_ends.map(
            lambda value: min(pd.Timestamp(value), requested_end)
        )
    result.insert(0, "period_start", pd.to_datetime(period_starts).to_numpy())
    result.insert(1, "period_end", pd.to_datetime(period_ends).to_numpy())
    return result


def _primary_heart_rate_trajectory_for_cycle_hierarchy(
    loader,
    users,
    statistics,
    resolution,
    parent_intervals,
    loader_kwargs,
    time_reference,
    primary_min_coverage,
    unresolved_streams,
):
    """Calculate the primary windows within each user's valid cycle span."""
    trajectories = []
    interval_users = parent_intervals.index.get_level_values(0)
    for user in users:
        if user not in interval_users:
            continue
        user_intervals = parent_intervals.xs(user)
        valid = user_intervals[user_intervals["interval_valid"]]
        if valid.empty:
            continue
        trajectory = _single_resolution_hr_trajectory(
            loader,
            [user],
            statistics,
            valid["start_time"].min(),
            valid["end_time"].max(),
            resolution,
            loader_kwargs,
            time_reference,
            primary_min_coverage,
            unresolved_streams,
        )
        if not trajectory.empty:
            trajectories.append(trajectory)
    if trajectories:
        return pd.concat(trajectories).sort_index()
    index = pd.MultiIndex.from_arrays(
        [[], []], names=["user_id", "window_start"]
    )
    return pd.DataFrame(
        columns=["period_start", "period_end", *statistics], index=index
    )


def _hierarchical_heart_rate_statistics(
    loader,
    statistics,
    users,
    start_date,
    end_date,
    chain,
    aggregation,
    min_coverage,
    min_valid_children,
    loader_kwargs,
    time_reference,
    cycle_kwargs,
):
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
    primary = chain[0]
    primary_reference = time_reference
    if time_reference == "auto" and any(
        value in cycle.CYCLE_TYPES or value in hierarchy.CALENDAR_RESOLUTIONS
        for value in chain[1:]
    ):
        # A hierarchy must use one coherent clock. Local calendar and sleep
        # intervals therefore make their lower fixed windows local as well.
        primary_reference = "local"
    cycle_target = next(
        (value for value in chain[1:] if value in cycle.CYCLE_TYPES), None
    )
    cycle_parent = None
    unresolved_streams = set()
    if cycle_target is not None:
        cycle_parent = cycle.get_intervals(
            loader,
            cycle_type=cycle_target,
            user_id=users,
            start_date=start_date,
            end_date=end_date,
            **cycle_kwargs,
        )
    if primary in cycle.CYCLE_TYPES:
        trajectory = _cycle_heart_rate_trajectory(
            loader,
            users,
            statistics,
            primary,
            start_date,
            end_date,
            loader_kwargs,
            cycle_kwargs,
            primary_min_coverage=coverage_thresholds[0],
            unresolved_streams=unresolved_streams,
        )
    elif cycle_parent is not None:
        trajectory = _primary_heart_rate_trajectory_for_cycle_hierarchy(
            loader,
            users,
            statistics,
            primary,
            cycle_parent,
            loader_kwargs,
            primary_reference,
            coverage_thresholds[0],
            unresolved_streams,
        )
    else:
        trajectory = _single_resolution_hr_trajectory(
            loader,
            users,
            statistics,
            start_date,
            end_date,
            primary,
            loader_kwargs,
            primary_reference,
            coverage_thresholds[0],
            unresolved_streams,
        )

    if coverage_thresholds[0] is not None:
        _warn_unknown_primary_coverage(unresolved_streams)

    rename = {feature: f"{feature}__{primary}" for feature in statistics}
    trajectory = trajectory.rename(columns=rename)
    for level_index, target in enumerate(chain[1:]):
        specification = aggregation_levels[level_index]
        if target in cycle.CYCLE_TYPES:
            parent = cycle_parent if target == cycle_target else None
            if parent is None:
                parent = cycle.get_intervals(
                    loader,
                    cycle_type=target,
                    user_id=users,
                    start_date=start_date,
                    end_date=end_date,
                    **cycle_kwargs,
                )
            parent = parent.rename_axis(index={"calendar_date": "window_start"})
            membership = "majority_overlap"
        else:
            bounds = {}
            active_users = []
            lower_resolution = chain[level_index]
            lower_is_cycle = lower_resolution in cycle.CYCLE_TYPES
            for user in users:
                user_rows = (
                    trajectory.xs(user)
                    if user in trajectory.index.get_level_values(0)
                    else trajectory.iloc[0:0]
                )
                if user_rows.empty and (start_date is None or end_date is None):
                    continue
                if start_date is None:
                    if lower_is_cycle:
                        lower = pd.Timestamp(user_rows.index.min()).normalize()
                    else:
                        lower = user_rows["period_start"].min()
                else:
                    lower = pd.Timestamp(start_date)
                    if lower_is_cycle:
                        lower = lower.normalize()
                if end_date is None:
                    if lower_is_cycle:
                        upper = (
                            pd.Timestamp(user_rows.index.max()).normalize()
                            + pd.Timedelta(days=1)
                        )
                    else:
                        upper = user_rows["period_end"].max()
                else:
                    upper = pd.Timestamp(end_date)
                    if lower_is_cycle:
                        upper = upper.normalize()
                if pd.isna(lower) or pd.isna(upper):
                    continue
                bounds[user] = (lower, upper)
                active_users.append(user)
            if not active_users:
                return trajectory.iloc[0:0].rename_axis(
                    index={"window_start": "calendar_date"}
                )
            parent = hierarchy.windows_for_resolution(
                active_users, target, bounds
            )
            membership = (
                "majority_overlap"
                if lower_resolution not in hierarchy.CALENDAR_RESOLUTIONS
                and lower_resolution not in cycle.CYCLE_TYPES
                and lower_resolution != "nightly"
                else "label"
            )
        trajectory, _ = hierarchy.aggregate_level(
            trajectory,
            parent,
            target_resolution=target,
            aggregation=specification,
            membership=membership,
            min_coverage=coverage_thresholds[level_index + 1],
            min_valid_children=valid_child_thresholds[level_index + 1],
        )
    if chain[-1] in cycle.CYCLE_TYPES or chain[-1] in hierarchy.CALENDAR_RESOLUTIONS:
        index_name = "calendar_date"
    else:
        index_name = time_utils.get_resampling_spec(
            chain[-1], time_reference=primary_reference
        ).time_col
        available = getattr(
            loader, "heart_rate_time_columns", heart_rate_schema.TIME_COLUMNS
        )
        if index_name not in available and primary_reference == "auto":
            index_name = heart_rate_schema.LOCAL_TIME_COL
    return trajectory.rename_axis(index={"window_start": index_name})


def get_heart_rate_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    resolution: Union[pd.Timedelta, str, list, tuple] = "day",
    loader_kwargs: dict = {},
    time_reference: str = "auto",
    aggregation=None,
    min_coverage=None,
    cycle_kwargs=None,
    min_valid_children=None,
) -> Union[pd.Series, pd.DataFrame]:
    """Return heart-rate features at one resolution or along a hierarchy.

    A scalar ``resolution`` preserves the ordinary heart-rate extraction API.
    A resolution sequence defines increasingly larger windows: features are
    first calculated from samples at the first resolution, then each upper
    level aggregates the preceding rows with equal child weighting.

    Parameters
    ----------
    loader : BaseLoader
        Loader providing heart-rate samples and, for cycles, sleep sessions.
    statistic : str, sequence of str, or None
        Primary heart-rate features. ``None`` requests every registered
        feature; resting heart rate is available for daily or cycle windows.
    user_id : str or list, default "all"
        Participant identifiers.
    start_date, end_date : timestamp-like or None
        For fixed and calendar windows, the exact half-open analysis interval
        ``[start_date, end_date)``. For cycle resolutions, the bounds select
        cycle calendar labels and selected cycles are never clipped. Without
        bounds, each participant uses only their observed trajectory.
    resolution : str, timedelta, or sequence, default "day"
        One window resolution or a strictly increasing hierarchy, such as
        ``["1h", "cycle_night", "monthly"]``.
    loader_kwargs : dict
        Extra keyword arguments passed to ``loader.load_heart_rate``.
    time_reference : {"auto", "local", "utc"}, default "auto"
        Clock used for fixed windows. ``auto`` selects UTC for purely subdaily
        extraction and local time when calendar or cycle parents require it.
        Explicit UTC is invalid when any sleep-defined cycle is requested.
    aggregation : str, sequence, nested sequence, or None
        Operations for levels above the primary one. ``None`` uses the
        standard numeric operations. With more than one upper level, provide
        one nested specification per level, for example
        ``[["mean"], ["mean", "p90"]]``.
    min_coverage : float, sequence of float or None
        Minimum coverage required at each requested resolution. The first
        value controls raw samples to the primary feature window; later values
        control valid child windows to each parent. A scalar applies to every
        level, while a sequence supplies one value per resolution. An
        individual ``None`` disables filtering at that level. If primary
        cadence cannot be inferred, its coverage is unknown: the feature is
        retained when no primary threshold is requested and excluded when one
        is requested.
    cycle_kwargs : dict or None
        Options forwarded to :func:`pywearable.cycle.get_intervals`, such as
        ``missing_end="midnight"``.
    min_valid_children : int, sequence of int or None
        Minimum valid child windows required at each upper resolution. A
        scalar applies to every upper level. A sequence supplies one value per
        resolution and must use ``None`` for the primary level. When both
        count and coverage thresholds are set, both must pass.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        A ``(user_id, time)`` trajectory. Hierarchical output names retain the
        full calculation path and include immediate child coverage columns.
        A single-resolution request with ``min_coverage`` returns a DataFrame
        containing each requested feature and its ``__coverage`` column.

    Examples
    --------
    Calculate ordinary five-minute features::

        get_heart_rate_statistics(loader, resolution="5min")

    Average hourly mean heart rate within each main-sleep cycle, then return
    its monthly mean and 90th percentile::

        get_heart_rate_statistics(
            loader,
            statistic="heart_rate_mean",
            resolution=["1h", "cycle_night", "monthly"],
            aggregation=[["mean"], ["mean", "p90"]],
            min_coverage=[0.5, 0.7, 0.8],
            min_valid_children=[None, 3, 2],
        )
    """
    chain, hierarchical = hierarchy.normalize_resolution_chain(resolution)
    cycle_kwargs = {} if cycle_kwargs is None else dict(cycle_kwargs)
    if time_reference not in {"auto", "local", "utc"}:
        raise ValueError("time_reference must be 'auto', 'local', or 'utc'")
    if time_reference == "utc" and any(
        value in cycle.CYCLE_TYPES for value in chain
    ):
        raise ValueError("sleep-defined heart-rate cycles require local time")
    if not hierarchical and aggregation is not None:
        raise ValueError("aggregation requires a resolution hierarchy")
    if not hierarchical and min_valid_children is not None:
        raise ValueError("min_valid_children requires a resolution hierarchy")
    if not hierarchical and chain[0] not in cycle.CYCLE_TYPES:
        threshold = hierarchy.normalize_coverage_thresholds(min_coverage, 1)[0]
        unresolved_streams = set()
        result = _get_heart_rate_statistics_single_resolution(
            loader=loader,
            statistic=statistic,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            resolution=chain[0],
            loader_kwargs=loader_kwargs,
            time_reference=time_reference,
            primary_min_coverage=threshold,
            return_primary_coverage=min_coverage is not None,
            unresolved_streams=unresolved_streams,
        )
        if threshold is not None:
            _warn_unknown_primary_coverage(unresolved_streams)
        return result

    users = (
        loader.get_user_ids()
        if user_id == "all" or user_id is None
        else user_id
    )
    if isinstance(users, str):
        users = [users]
    users = list(users)
    statistics = _heart_rate_statistics_list(
        statistic, allow_rest=chain[0] in cycle.CYCLE_TYPES or chain[0] == "daily"
    )
    if len(chain) == 1:
        threshold = hierarchy.normalize_coverage_thresholds(min_coverage, 1)[0]
        unresolved_streams = set()
        trajectory = _cycle_heart_rate_trajectory(
            loader,
            users,
            statistics,
            chain[0],
            start_date,
            end_date,
            loader_kwargs,
            cycle_kwargs,
            primary_min_coverage=threshold,
            return_primary_coverage=min_coverage is not None,
            unresolved_streams=unresolved_streams,
        )
        if threshold is not None:
            _warn_unknown_primary_coverage(unresolved_streams)
        result_columns = list(statistics)
        if min_coverage is not None:
            result_columns.extend(
                f"{statistic_name}__coverage" for statistic_name in statistics
            )
        result = trajectory[result_columns].copy()
        result.index = result.index.set_names(["user_id", "calendar_date"])
        if len(statistics) == 1 and min_coverage is None:
            return result[statistics[0]]
        return result

    return _hierarchical_heart_rate_statistics(
        loader,
        statistics,
        users,
        start_date,
        end_date,
        chain,
        aggregation,
        min_coverage,
        min_valid_children,
        loader_kwargs,
        time_reference,
        cycle_kwargs,
    )
