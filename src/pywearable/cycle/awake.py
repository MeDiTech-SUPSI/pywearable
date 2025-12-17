"""
Extraction of cycle awake (wake-up to bedtime) statistics.
"""

from typing import Union

import pandas as pd

from ..loader import BaseLoader
from . import utils


############################################
#               Duration                   #
############################################
def get_cycle_awake_start(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the start of the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily start time of awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_start",
    )


def get_cycle_awake_end(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the end of the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily end time of awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_end",
    )


def get_cycle_awake_duration(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the duration of the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle,
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily duration of awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_duration",
    )


def get_cycle_awake_wearing_time(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the wearing time during the awake cycle.

    The wearing time is computed starting from the
    heart rate samples obtained during the awake
    cycle, and converted in decimal hours.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily wearing time during the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_wearing_time",
    )


############################################
#           Heart Rate Statistics          #
############################################
def get_cycle_awake_heart_rate_count(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the count of HR samples in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily count of heart rate values in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_heart_rate_count",
    )


def get_cycle_awake_heart_rate_max(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the maximum heart rate in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily maximum heart rate in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_heart_rate_max",
    )


def get_cycle_awake_heart_rate_median(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the median heart rate in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily median heart rate in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_heart_rate_median",
    )


def get_cycle_awake_heart_rate_mean(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the mean heart rate in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily mean heart rate value in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_heart_rate_mean",
    )


def get_cycle_awake_heart_rate_min(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the minimum heart rate in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily minimum heart rate in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_heart_rate_min",
    )


def get_cycle_awake_heart_rate_rest(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the rest heart rate in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily rest heart rate value in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_heart_rate_rest",
    )


def get_cycle_awake_heart_rate_std(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the heart rate standard deviation in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily heart rate standard deviation value in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_heart_rate_std",
    )


############################################
#           Steps Statistics              #
############################################
def get_cycle_awake_steps(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the count of steps in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily step count in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_steps",
    )


############################################
#           Stress Statistics              #
############################################
def get_cycle_awake_stress_count(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the count of stress samples in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily count of stress values in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_stress_count",
    )


def get_cycle_awake_stress_max(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the maximum stress in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily maximum stress in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_stress_max",
    )


def get_cycle_awake_stress_median(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the median stress in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily median stress in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_stress_median",
    )


def get_cycle_awake_stress_mean(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the mean stress in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily mean stress value in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_stress_mean",
    )


def get_cycle_awake_stress_min(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the minimum stress in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily minimum stress in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_stress_min",
    )


def get_cycle_awake_stress_std(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the stress standard deviation in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily stress standard deviation value in awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_stress_std",
    )


############################################
#           Stress Ranges Statistics       #
############################################
def get_cycle_awake_high_stress_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the percentage of time spent in the high stress in the awake cycle.

    This function returns the daily percentage of time spent
    in the high stress zone [76-100] during the time spent
    in awake state.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily percentage of high stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_high_stress_percentage",
    )


def get_cycle_awake_high_stress_duration(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the total time spent in the high stress in the awake cycle.

    This function returns the daily time spent
    in the high stress zone [76-100] in the awake cycle.
    The returned value is expressed in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily duration of high stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_high_stress_ms",
    )


def get_cycle_awake_medium_stress_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the percentage of time spent in the medium stress in the awake cycle.

    This function returns the daily percentage of time spent
    in the medium stress zone [51-75] during the time spent
    in awake state.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily percentage of medium stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_medium_stress_percentage",
    )


def get_cycle_awake_medium_stress_duration(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the total time spent in the medium stress in the awake cycle.

    This function returns the daily time spent
    in the medium stress zone [51-75] in the awake cycle.
    The returned value is expressed in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily duration of medium stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_medium_stress_ms",
    )


def get_cycle_awake_low_stress_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the percentage of time spent in the low stress in the awake cycle.

    This function returns the daily percentage of time spent
    in the low stress zone [26-50] during the time spent
    in awake state.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily percentage of low stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_low_stress_percentage",
    )


def get_cycle_awake_low_stress_duration(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the total time spent in the low stress in the awake cycle.

    This function returns the daily time spent
    in the low stress zone [26-50] in the awake cycle.
    The returned value is expressed in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily duration of low stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_low_stress_ms",
    )


def get_cycle_awake_rest_stress_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the percentage of time spent in the rest stress in the awake cycle.

    This function returns the daily percentage of time spent
    in the rest stress zone [0-25] during the time spent
    in awake state.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily percentage of rest stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_rest_stress_percentage",
    )


def get_cycle_awake_rest_stress_duration(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the total time spent in the rest stress in the awake cycle.

    This function returns the daily time spent
    in the rest stress zone [0-25] in the awake cycle.
    The returned value is expressed in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily duration of rest stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_rest_stress_ms",
    )


def get_cycle_awake_unmeasurable_stress_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the percentage of time spent in the unmeasurable stress in the awake cycle.

    This function returns the daily percentage of time spent
    in the unmeasurable stress zone [-2,-1] during the time spent
    in awake state.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily percentage of unmeasurable stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_unmeasurable_stress_percentage",
    )


def get_cycle_awake_unmeasurable_stress_duration(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get the total time spent in the unmeasurable stress in the awake cycle.

    This function returns the daily time spent
    in the unmeasurable stress zone [-2,-1] in the awake cycle.
    The returned value is expressed in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.

    Returns
    -------
    :class:`pandas.Series`
        Daily duration of unmeasurable stress in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_unmeasurable_stress_ms",
    )


############################################
#           Activity percentages           #
############################################
def get_cycle_awake_sedentary_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float = 0.8,
) -> pd.Series:
    """Get the percentage of time spent in the sedentary status in the awake cycle.

    Parameters
    ----------
    loader : :class:`renewal.loader.base.RenewalDataLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.
    hr_coverage : :class:`float`, optional
        Percentage of HR values required to consider an epoch as valid.

    Returns
    -------
    :class:`pandas.Series`
        Daily sedentary percentage in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_sedentary_time_percentage",
        hr_coverage=hr_coverage,
    )


def get_cycle_awake_sedentary_time(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float = 0.8,
) -> pd.Series:
    """Get the time spent in the sedentary status in the awake cycle.

    Sedentary time is reported in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.
    hr_coverage : :class:`float`, optional
        Percentage of HR values required to consider an epoch as valid.

    Returns
    -------
    :class:`pandas.Series`
        Daily sedentary time in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics=f_constants.CYCLE_AWAKE_EPOCHS_SEDENTARY_MS,
        hr_coverage=hr_coverage,
    )


def get_cycle_awake_active_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float = 0.8,
) -> pd.Series:
    """Get the percentage of time spent in the active status in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.

        Voucher identifier, only for compatibility with other functions and unused.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.
    hr_coverage : :class:`float`, optional
        Percentage of HR values required to consider an epoch as valid.

    Returns
    -------
    :class:`pandas.Series`
        Daily active percentage in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics=f_constants.CYCLE_AWAKE_EPOCHS_ACTIVE_PERCENTAGE,
        hr_coverage=hr_coverage,
    )


def get_cycle_awake_active_time(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float = 0.8,
) -> pd.Series:
    """Get the time spent in the active status in the awake cycle.

    Active time is reported in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.

        Voucher identifier, only for compatibility with other functions and unused.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.
    hr_coverage : :class:`float`, optional
        Percentage of HR values required to consider an epoch as valid.

    Returns
    -------
    :class:`pandas.Series`
        Daily active time in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics=f_constants.CYCLE_AWAKE_EPOCHS_ACTIVE_MS,
        hr_coverage=hr_coverage,
    )


def get_cycle_awake_highly_active_percentage(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float = 0.8,
) -> pd.Series:
    """Get the percentage of time spent in the highly active status in the awake cycle.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.

        Voucher identifier, only for compatibility with other functions and unused.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.
    hr_coverage : :class:`float`, optional
        Percentage of HR values required to consider an epoch as valid.

    Returns
    -------
    :class:`pandas.Series`
        Daily highly active percentage in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics=f_constants.CYCLE_AWAKE_EPOCHS_HIGHLY_ACTIVE_PERCENTAGE,
        hr_coverage=hr_coverage,
    )


def get_cycle_awake_highly_active_time(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float = 0.8,
) -> pd.Series:
    """Get the time spent in the highly active status in the awake cycle.

    Active time is reported in milliseconds.

    Parameters
    ----------
    loader : :class:`BaseLoader`
        Initialized data loader.
    user_id : :class:`str`
        Unique user identifier.

        Voucher identifier, only for compatibility with other functions and unused.
    start_date : :class:`pandas.Timestamp`
        Date of the first awake cycle.
    end_date : :class:`pandas.Timestamp`
        Date of the last awake cycle.
    hr_coverage : :class:`float`, optional
        Percentage of HR values required to consider an epoch as valid.

    Returns
    -------
    :class:`pandas.Series`
        Daily highly active time in the awake cycle.
    """
    return get_cycle_awake_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        statistics="cycle_awake_highly_active_time_ms",
        hr_coverage=hr_coverage,
    )


CYCLE_AWAKE_STATISTICS_DICT = {
    "cycle_awake_duration": utils.compute_cycle_duration,
    "cycle_awake_wearing_time": utils.compute_cycle_wearing_time,
    "cycle_awake_start": utils.compute_cycle_start,
    "cycle_awake_end": utils.compute_cycle_end,
    "cycle_awake_heart_rate_count": utils.compute_cycle_heart_rate_count,
    "cycle_awake_heart_rate_max": utils.compute_cycle_heart_rate_max,
    "cycle_awake_heart_rate_mean": utils.compute_cycle_heart_rate_mean,
    "cycle_awake_heart_rate_median": utils.compute_cycle_heart_rate_median,
    "cycle_awake_heart_rate_min": utils.compute_cycle_heart_rate_min,
    "cycle_awake_heart_rate_rest": utils.compute_cycle_heart_rate_rest,
    "cycle_awake_heart_rate_std": utils.compute_cycle_heart_rate_std,
    "cycle_awake_steps": utils.compute_cycle_steps,
    "cycle_awake_stress_count": utils.compute_cycle_stress_count,
    "cycle_awake_stress_max": utils.compute_cycle_stress_max,
    "cycle_awake_stress_mean": utils.compute_cycle_stress_mean,
    "cycle_awake_stress_median": utils.compute_cycle_stress_median,
    "cycle_awake_stress_min": utils.compute_cycle_stress_min,
    "cycle_awake_stress_std": utils.compute_cycle_stress_std,
    "cycle_awake_high_stress_ms": utils.compute_cycle_high_stress_duration,
    "cycle_awake_high_stress_percentage": utils.compute_cycle_high_stress_percentage,
    "cycle_awake_low_stress_ms": utils.compute_cycle_low_stress_duration,
    "cycle_awake_low_stress_percentage": utils.compute_cycle_low_stress_percentage,
    "cycle_awake_medium_stress_ms": utils.compute_cycle_medium_stress_duration,
    "cycle_awake_medium_stress_percentage": utils.compute_cycle_medium_stress_percentage,
    "cycle_awake_rest_stress_ms": utils.compute_cycle_rest_stress_duration,
    "cycle_awake_rest_stress_percentage": utils.compute_cycle_rest_stress_percentage,
    "cycle_awake_unmeasurable_stress_ms": utils.compute_cycle_unmeasurable_stress_duration,
    "cycle_awake_unmeasurable_stress_percentage": utils.compute_cycle_unmeasurable_stress_percentage,
    "cycle_awake_sedentary_time_ms": utils.compute_cycle_epochs_sedentary_duration,
    "cycle_awake_sedentary_time_percentage": utils.compute_cycle_epochs_sedentary_percentage,
    "cycle_awake_active_time_ms": utils.compute_cycle_epochs_active_duration,
    "cycle_awake_active_time_percentage": utils.compute_cycle_epochs_active_percentage,
    "cycle_awake_highly_active_time_ms": utils.compute_cycle_epochs_highly_active_duration,
    "cycle_awake_highly_active_time_percentage": utils.compute_cycle_epochs_highly_active_percentage,
}


def get_cycle_awake_statistics(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float = 0.8,
    statistics: Union[str, list, None] = None,
) -> Union[pd.Series, pd.DataFrame]:
    return utils.get_cycle_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        cycle_statistics=statistics,
        hr_coverage=hr_coverage,
    )
