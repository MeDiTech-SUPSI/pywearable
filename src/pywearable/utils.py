"""
This module contains utility functions that don't directly load/compute metrics.
"""

import datetime
import time
from cmath import phase, rect
from math import degrees, floor, radians
from typing import Union

import dateutil.parser
import numpy as np
import pandas as pd
import scipy.stats

from . import constants

_LABFRONT_LAST_SAMPLE_UNIX_TIMESTAMP_IN_MS_KEY = "lastSampleUnixTimestampInMs"
_LABFRONT_QUESTIONNAIRE_STRING = "questionnaire"
_LABFRONT_TODO_STRING = "todo"
_MS_TO_DAY_CONVERSION = 1000 * 60 * 60 * 24


def check_date(
    date: Union[datetime.datetime, datetime.date, str, None],
) -> datetime.datetime:
    """Convert date to :class:`datetime.datetime` format

    Parameters
    ----------
    date : datetime.datetime or datetime.date or str or None
        Date to be converted

    Returns
    -------
    datetime.datetime
        Converted date.

    Raises
    ------
    ValueError
        If ``date`` is not of valid format.
    """
    # Check dates and times
    if isinstance(date, str):
        return dateutil.parser.parse(date)
    elif type(date) == datetime.date:
        return datetime.datetime.combine(date, datetime.time())
    elif date is None:
        return None
    elif type(date) == datetime.datetime:
        return date
    else:
        raise ValueError(f"{type(date)} is not valid.")


def check_start_and_end_dates(
    start_date: Union[datetime.datetime, None], end_date: Union[datetime.datetime, None]
) -> bool:
    """Check if dates are in correct order.

    This function checks if ``start_date`` is
    lower or equal than ``end_date`` and returns
    `True` if this is the case, `False` otherwise.

    Parameters
    ----------
    start_date : :class:`datetime.datetime` or None
        Start date that needs to be checked.
    end_date : :class:`datetime.datetime` or None
        End date that needs to be checked.

    Returns
    -------
    bool
        `True` if ``start_date`` is equal or lower than ``end_date``,
        `False` otherwise.
    """
    if not ((start_date is None) and (end_date is None)):
        if end_date < start_date:
            return False
    return True


def check_is_df(input, raise_name="input"):
    if not isinstance(input, pd.DataFrame):
        raise ValueError(
            f"{raise_name} must be a pd.DataFrame. {type(input)} is not a valid type."
        )
    return input


def get_user_ids(loader, user_id: Union[str, list] = "all"):
    """Returns user ids in the appropriate format required by pywearable functions

    Parameters
    ----------
    loader : :class:`loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) of the user(s) of interest, by default "all".

    Returns
    -------
    list
        List of all the full user ids of interest.
    """
    if user_id == "all":
        user_id = loader.get_user_ids()

    elif isinstance(user_id, str):
        user_id = [user_id]

    elif not isinstance(user_id, list):
        raise TypeError("user_id has to be a list.")

    return user_id


def return_multiindex_df(
    df: pd.DataFrame, return_df: bool, return_multi_index: bool
) -> Union[dict, pd.DataFrame]:
    # Check if df is pd.DataFrame
    if not (type(df) == pd.DataFrame):
        raise ValueError(f"df must be of type pd.DataFrame, not of {type(df)}")
    # Check if we have multilevel index
    if not (type(df.index) == pd.MultiIndex):
        raise ValueError(
            f"df.index must be of type pd.MultiIndex, not of {type(df.index)}"
        )
    if not return_df:
        # We need to perform some changes to return dictionary with dates
        # and not pd.Timestamp as keys
        if len(df.columns) > 1:
            """
            More than one column, dictionary will be:
            {
                user_id1: {
                    datetime.date1: {
                        metric1: value1,
                        metric2: value2,
                        ...
                    },
                    datetime.date2: {
                        metric1: value1,
                        metric2: value2,
                    }
                },
                user_id2: {
                    datetime.date1: {
                        ...
                    }
                }
            }

            """
            return {
                user: {
                    date.to_pydatetime().date(): df.xs((user, date)).to_dict()
                    for date in df.index.levels[1]
                }
                for user in df.index.levels[0]
            }
        else:
            """
            Single metric, dictionary will be:
            {
                user_id1: {
                    datetime.date1: value1,
                    datetime.date2: value2
                },
                user_id2: {
                    datetime.date1: value1,
                    datetime.date2: value2
                }
            }
            """
            return {
                level: {
                    k.to_pydatetime().date(): v
                    for k, v in df.xs(level).loc[:, df.columns[0]].to_dict().items()
                }
                for level in df.index.levels[0]
            }
    else:
        # We return a dataframe
        if return_multi_index:
            # Standard multi-index dataframe
            return df
        else:
            # Single index dataframe
            return df.reset_index(
                level=[constants.USER_COL, constants.CALENDAR_DATE_COL]
            )


def get_summary(loader, comparison_date=time.time()):
    """Returns a general summary of the latest update of every metric for every participant

    Args:
        loader (:class:`pylabfront.loader.LabfrontLoader`): Instance of `LabfrontLoader`.
        comparison_date (float): unix time in seconds for comparison. Defaults to the current unix time.

    Returns:
        DataFrame: general summary of the number of full days since metrics were updated for all participants.
        Entries are NaN if the metric has never been registered for the participant.
    """

    available_metrics = set(loader.get_available_metrics())
    available_metrics.discard("todo")
    available_metrics.discard("questionnaire")
    available_metrics = sorted(list(available_metrics))
    available_questionnaires = loader.get_available_questionnaires()
    available_todos = loader.get_available_todos()

    features_dictionary = {}

    for participant_id in sorted(loader.get_user_ids()):
        full_participant_id = loader.get_full_id(participant_id)
        features_dictionary[participant_id] = {}
        participant_metrics = loader.get_available_metrics([participant_id])
        participant_questionnaires = loader.get_available_questionnaires(
            [participant_id]
        )
        participant_todos = loader.get_available_todos([participant_id])

        for metric in available_metrics:
            if metric not in participant_metrics:
                features_dictionary[participant_id][metric] = None
            else:  # figure out how many days since the last update
                last_unix_times = [
                    v[_LABFRONT_LAST_SAMPLE_UNIX_TIMESTAMP_IN_MS_KEY]
                    for v in loader.data_dictionary[full_participant_id][
                        metric
                    ].values()
                ]
                number_of_days_since_update = (
                    comparison_date * 1000 - max(last_unix_times)
                ) // _MS_TO_DAY_CONVERSION
                features_dictionary[participant_id][
                    metric
                ] = number_of_days_since_update

        for questionnaire in available_questionnaires:
            if questionnaire not in participant_questionnaires:
                features_dictionary[participant_id][questionnaire] = None
            else:
                last_unix_times = [
                    v[_LABFRONT_LAST_SAMPLE_UNIX_TIMESTAMP_IN_MS_KEY]
                    for v in loader.data_dictionary[full_participant_id][
                        _LABFRONT_QUESTIONNAIRE_STRING
                    ][questionnaire].values()
                ]
                number_of_days_since_update = (
                    comparison_date * 1000 - max(last_unix_times)
                ) // _MS_TO_DAY_CONVERSION
                features_dictionary[participant_id][
                    questionnaire
                ] = number_of_days_since_update

        for todo in available_todos:
            if todo not in participant_todos:
                features_dictionary[participant_id][todo] = None
            else:
                last_unix_times = [
                    v[_LABFRONT_LAST_SAMPLE_UNIX_TIMESTAMP_IN_MS_KEY]
                    for v in loader.data_dictionary[full_participant_id][
                        _LABFRONT_TODO_STRING
                    ][todo].values()
                ]
                number_of_days_since_update = (
                    comparison_date * 1000 - max(last_unix_times)
                ) // _MS_TO_DAY_CONVERSION
                features_dictionary[participant_id][todo] = number_of_days_since_update

    df = pd.DataFrame(features_dictionary)
    return df.T


def is_task_repeatable(file_path):
    """Returns boolean indication of questionnaire/todo repeatability

    Args:
        file_path (Path): path to the folder of the questionnaire/todo csv files

    Returns:
        bool: Indication if the questionnaire/todo is repeatable
    """
    csv_path = list((file_path).iterdir())[0]
    with open(csv_path, "r") as f:
        for _ in range(5):
            line = f.readline().split(",")
    return line[7] == "true"


def is_weekend(day):
    """Indication if the day considered is either a Saturday or Sunday.

    Args:
        day (datetime): date of interest.

    Returns:
        bool: True/False depending if day is a weekend day or not.
    """
    return day.weekday() in [5, 6]


def find_nearest_timestamp(timestamp, timestamp_array):
    """Finds the closest time between a set of timestamps to a given timestamp.

    Args:
        timestamp (datetime): Date of interest, of which to find closest timestamp.
        timestamp_array (datetime): Array of datetimes

    Returns:
        datetime: Closest datetime in timestamp_array to timestamp
    """
    return min(timestamp_array, key=lambda x: abs(x - timestamp))


def trend_analysis(
    data_dict: dict,
    ma_type: str = "MA",
    baseline_periods: int = 7,
    min_periods_baseline: int = 3,
    normal_range_periods: int = 30,
    min_periods_normal_range: int = 20,
    std_multiplier: float = 1.0,
):
    """Perform trend analysis for a pre-loaded metric of interest.

    This function performs trend analysis, i.e. it comptues medium, and long-term trends
    in a metric of interest for a user. The function computes the moving average of the metric
    and the normal range of the metric.

    Parameters
    ----------
    data_dict : :class:`dict`
        Dictionary containing the pre-loaded metric values for the user.
    ma_type : :class:`str`, optional
        Type of moving average to compute, can be either "MA" or "EMA", by default "MA"
    baseline_periods : :class:`int`, optional
        Number of period values to be used for the computation of the rolling baseline, by default 7
    min_periods_baseline : :class:`int`, optional
        Minimum number of period values needed for the baseline to be considered valid, by default 3
    normal_range_periods : :class:`int`, optional
        Number of period values to be used for the computation of the rolling normal range, by default 30
    min_periods_normal_range : :class:`int`, optional
        Minimum number of period values needed for the normal range to be considered valid, by default 20
    std_multiplier : :class:`float`, optional
        Multiplier of the standard deviation for the computation of lower and upper bounds of the normal range, by default 1.0

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing daily metric, baseline, and normal range params.

    Raises
    ------
    ValueError
        If the moving average type is not "MA" or "EMA".
    """

    start_date = min(data_dict.keys())
    end_date = max(data_dict.keys())

    idx = pd.date_range(start_date, end_date)
    s = pd.Series(data_dict)
    s.index = pd.DatetimeIndex(s.index)
    s = s.reindex(idx, fill_value=None)
    df = pd.DataFrame(data=s.values, index=s.index, columns=["metric"])
    if ma_type == "MA":
        df["BASELINE"] = df.metric.rolling(
            window=baseline_periods, min_periods=min_periods_baseline
        ).mean()
        df["NR_MEAN"] = df.metric.rolling(
            window=normal_range_periods, min_periods=min_periods_normal_range
        ).mean()
        df["NR_STD"] = df.metric.rolling(
            window=normal_range_periods, min_periods=min_periods_normal_range
        ).std()
    elif ma_type == "EMA":
        df["BASELINE"] = df.metric.ewm(
            span=baseline_periods, min_periods=min_periods_baseline
        ).mean()
        df["NR_MEAN"] = df.metric.ewm(
            span=normal_range_periods, min_periods=min_periods_normal_range
        ).mean()
        df["NR_STD"] = df.metric.ewm(
            span=normal_range_periods, min_periods=min_periods_normal_range
        ).std()
    else:
        raise ValueError('Moving average kind can only be "normal" or "exponential"')

    df["NR_LOWER_BOUND"] = df["NR_MEAN"] - std_multiplier * df["NR_STD"]
    df["NR_UPPER_BOUND"] = df["NR_MEAN"] + std_multiplier * df["NR_STD"]
    return df


def mean_angle(deg):
    return degrees(phase(sum(rect(1, radians(d)) for d in deg) / len(deg)))


def mean_time(times: Union[str, list]) -> str:
    t = [
        time.split(":") if type(time) is str else (time.hour, time.minute)
        for time in times
        if not (pd.isnull(time))
    ]
    # t = (time.split(":") for time in times)
    seconds = ((int(m) * 60 + int(h) * 3600) for h, m in t)
    day = 24 * 60 * 60
    to_angles = [s * 360.0 / day for s in seconds]
    mean_as_angle = mean_angle(to_angles)
    mean_seconds = mean_as_angle * day / 360.0
    if mean_seconds < 0:
        mean_seconds += day
    h, m = divmod(mean_seconds, 3600)
    m, s = divmod(m, 60)
    if (h == 24) and (m == 0):
        h = 0
    return "%02i:%02i" % (h, m)


def get_earliest_bedtime(times: list) -> str:
    """Get earliest bedtime from list of bedtimes.

    This function returns the earliest bedtime
    from a list of bedtimes. The way in which this function
    returns the earliest bedtime is by computing the
    time difference between the bedtimes and
    "12:00". The time with the lowest difference from 12 o'clock
    (considering times occurring after "00:00" with +1 day) is
    considered as the earliest bedtime.

    Parameters
    ----------
    times : :class:`list`
        List of times from which the earliest bedtime must be determined.
        The times can be either of class :class:`str` or :class:`datetime.datetime`
        or :class:`pandas.Timestamp`

    Returns
    -------
    :class:`str`
        Earliest bedtime in HH:MM format.
    """
    dt_times = []
    for time in times:
        if not pd.isnull(time):
            if type(time) is str:
                hour, minute = [int(x) for x in time.split(":")]
            else:
                hour, minute = time.hour, time.minute
            dt_times.append(datetime.time(hour, minute))

    converted_dt_times = [
        datetime.datetime.combine(datetime.date.today(), dt_time)
        for dt_time in dt_times
    ]
    converted_dt_times = [
        (
            dt_time + datetime.timedelta(days=1)
            if (dt_time.hour >= 0 and dt_time.hour < 12)
            else dt_time
        )
        for dt_time in converted_dt_times
    ]

    dt_diff = np.array(
        [
            (
                dt_time
                - datetime.datetime.combine(datetime.date.today(), datetime.time(12, 0))
            ).total_seconds()
            for dt_time in converted_dt_times
        ]
    )

    return dt_times[dt_diff.argmin()].strftime("%H:%M")


def get_earliest_wakeup_time(times: list) -> str:
    dt_times = []
    for time in times:
        if not pd.isnull(time):
            if type(time) is str:
                hour, minute = [int(x) for x in time.split(":")]
            else:
                hour, minute = time.hour, time.minute
            dt_times.append(datetime.time(hour, minute))

    converted_dt_times = [
        datetime.datetime.combine(datetime.date.today(), dt_time)
        for dt_time in dt_times
    ]
    converted_dt_times = [
        (
            dt_time - datetime.timedelta(days=1)
            if (dt_time.hour >= 12 and dt_time.hour <= 23)
            else dt_time
        )
        for dt_time in converted_dt_times
    ]

    dt_diff = np.array(
        [
            (
                datetime.datetime.combine(datetime.date.today(), datetime.time(12, 0))
                - dt_time
            ).total_seconds()
            for dt_time in converted_dt_times
        ]
    )

    return dt_times[dt_diff.argmax()].strftime("%H:%M")


def get_latest_bedtime(times: list) -> str:
    dt_times = []
    for time in times:
        if not pd.isnull(time):
            if type(time) is str:
                hour, minute = [int(x) for x in time.split(":")]
            else:
                hour, minute = time.hour, time.minute
            dt_times.append(datetime.time(hour, minute))
    converted_dt_times = [
        datetime.datetime.combine(datetime.date.today(), dt_time)
        for dt_time in dt_times
    ]
    converted_dt_times = [
        (
            dt_time + datetime.timedelta(days=1)
            if (dt_time.hour >= 0 and dt_time.hour < 12)
            else dt_time
        )
        for dt_time in converted_dt_times
    ]

    dt_diff = np.array(
        [
            (
                dt_time
                - datetime.datetime.combine(datetime.date.today(), datetime.time(12, 0))
            ).total_seconds()
            for dt_time in converted_dt_times
        ]
    )

    return dt_times[dt_diff.argmax()].strftime("%H:%M")


def get_latest_wakeup_time(times: list) -> str:
    dt_times = []
    for time in times:
        if not pd.isnull(time):
            if type(time) is str:
                hour, minute = [int(x) for x in time.split(":")]
            else:
                hour, minute = time.hour, time.minute
            dt_times.append(datetime.time(hour, minute))
    converted_dt_times = [
        datetime.datetime.combine(datetime.date.today(), dt_time)
        for dt_time in dt_times
    ]
    converted_dt_times = [
        (
            dt_time - datetime.timedelta(days=1)
            if (dt_time.hour >= 12 and dt_time.hour <= 23)
            else dt_time
        )
        for dt_time in converted_dt_times
    ]

    dt_diff = np.array(
        [
            (
                datetime.datetime.combine(datetime.date.today(), datetime.time(12, 0))
                - dt_time
            ).total_seconds()
            for dt_time in converted_dt_times
        ]
    )

    return dt_times[dt_diff.argmin()].strftime("%H:%M")


def std_time(times):
    if type(times) is str:
        times = [times]
    t = [
        time.split(":") if type(time) is str else (time.hour, time.minute)
        for time in times
        if not pd.isnull(time)
    ]

    # Get seconds
    seconds = [(int(m) * 60 + int(h) * 3600) for h, m in t]

    # Get angle in degrees
    day = 24 * 60 * 60
    to_angles = [s * 360.0 / day for s in seconds]
    # Get angles in radiants
    angles = np.deg2rad(to_angles)

    # Apply circular mean and std
    circstd = scipy.stats.circstd(angles)

    circstd_degrees = np.rad2deg(circstd)

    h_std = floor((circstd_degrees / 15) % 24)
    m_std = ((circstd_degrees / 15) % 24 - h_std) * 60
    return "%02i:%02i" % (h_std, m_std)


def filter_out_sleep_stages(
    loader,
    user_id: str,
    df: pd.DataFrame,
    stages_to_filter: Union[list, str],
    # date: datetime.date,
    # resolution: int = 1,
):
    """Filters out night data relative to periods where the user was awake

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        Initialized instance of data loader.
    user : class:`str`
        ID of the user for which data has to be filtered.
    df : class:`pandas.DataFrame`
        DataFrame of the data to filter, must contain an isoDate column
    date : class:`datetime.date`
        Date relative to the night in consideration.
    resolution : class:`int`, optional
        Resolution of the hypnogram used to find awake periods, by default 1

    Returns
    -------
    :class:`pandas.DataFrame`
        Filtered version of the original `df`, including only data of periods when the participant is asleep.
    """
    # Check stages to filter
    if type(stages_to_filter) == str:
        stages_to_filter = [stages_to_filter]
    assert type(stages_to_filter) is list
    for stage in stages_to_filter:
        if not stage in [
            constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
            constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
            constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
            constants.SLEEP_STAGE_REM_STAGE_VALUE,
            constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
        ]:
            raise ValueError(f"Sleep stage {stage} is not valid")

    sleep_summary: pd.DataFrame = loader.load_sleep_summary(
        user_id=user_id,
        start_date=df[constants.ISODATE_COL].min() - pd.Timedelta(hours=48),
        end_date=df[constants.ISODATE_COL].max() + pd.Timedelta(hours=48),
        drop_duplicates=True,
    )

    sleep_stages: pd.DataFrame = loader.load_sleep_stage(
        user_id=user_id,
        start_date=df[constants.ISODATE_COL].min() - pd.Timedelta(hours=24),
        end_date=df[constants.ISODATE_COL].max() + pd.Timedelta(hours=24),
    )
    sleep_stages[constants.SLEEP_STAGE_SLEEP_TYPE_COL] = sleep_stages[
        constants.SLEEP_STAGE_SLEEP_TYPE_COL
    ].str.lower()

    sleep_stages["endIsoDate"] = sleep_stages[constants.ISODATE_COL] + pd.to_timedelta(
        sleep_stages[constants.DURATION_IN_MS_COL] - 1, unit="ms"
    )
    sleep_stages = sleep_stages.loc[
        sleep_stages[constants.SLEEP_SUMMARY_ID_COL].isin(
            sleep_summary[constants.SLEEP_SUMMARY_ID_COL].unique()
        )
    ]

    sleep_stages = sleep_stages.drop_duplicates(
        subset=[
            constants.UNIXTIMESTAMP_IN_MS_COL,
            constants.SLEEP_SUMMARY_ID_COL,
        ],
        keep="last",
    )
    sleep_stages = (
        pd.melt(
            frame=sleep_stages,
            id_vars=[
                constants.UNIXTIMESTAMP_IN_MS_COL,
                constants.SLEEP_SUMMARY_ID_COL,
                constants.SLEEP_STAGE_SLEEP_TYPE_COL,
            ],
            value_vars=[constants.ISODATE_COL, "endIsoDate"],
        )
        .set_index("value")
        .sort_index(ascending=True)
    )
    resampled_sleep_stages = (
        sleep_stages.groupby(constants.UNIXTIMESTAMP_IN_MS_COL)
        .resample(rule="30s", include_groups=False)
        .ffill()
        .reset_index(names=[constants.UNIXTIMESTAMP_IN_MS_COL, constants.ISODATE_COL])
    )
    resampled_sleep_stages = resampled_sleep_stages.drop_duplicates(
        subset=constants.ISODATE_COL, keep="first"
    )
    df = pd.merge_asof(
        left=df.sort_values(by=constants.ISODATE_COL),
        right=resampled_sleep_stages.drop(
            constants.UNIXTIMESTAMP_IN_MS_COL, axis=1
        ).sort_values(constants.ISODATE_COL),
        left_on=constants.ISODATE_COL,
        right_on=constants.ISODATE_COL,
        direction="nearest",
        tolerance=pd.Timedelta(seconds=60),
    )
    df = df.loc[
        ~df[constants.SLEEP_STAGE_SLEEP_TYPE_COL].isin(stages_to_filter),
        :,
    ].rename(columns={constants.SLEEP_STAGE_SLEEP_TYPE_COL: "sleep_stage"})
    return df
