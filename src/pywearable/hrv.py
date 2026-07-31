"""
This module contains all the functions related to the analysis
of cardiac data, mostly for heart rate (HR) and heart rate
variability (HRV).
"""

import datetime
import math
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy import interpolate, signal

from . import constants  # , sleep, utils
from .loader.base import BaseLoader
from .utils import time as time_utils

# TODO hrv statistics extracted at 5-minute intervals
# TODO possibility of filtering by day/night
# TODO return multiple statistics by night/day
# TODO add parameter for base_window
# TODO add neurokit as backend
# TODO check if loader has hrv_features already
# TODO compute all power statistics at once

HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES = "hrv_rmssd"
HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL = "hrv_sdnn"
HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER = "hrv_lf_power"
HRV_STATISTIC_LOW_FREQUENCY_POWER = "hrv_lf_power"
HRV_STATISTIC_HIGH_FREQUENCY_POWER = "hrv_hf_power"
HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO = "hrv_lfhf_ratio"
HRV_STATISTIC_BBI_MEDIAN = "hrv_bbi_median"
HRV_STATISTIC_BBI_TIME = "hrv_bbi_wearing_time"

HRV_TIME_DOMAIN_STATISTICS = [
    HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES,
    HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL,
]

HRV_FREQUENCY_DOMAIN_STATISTICS = [
    HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER,
    HRV_STATISTIC_LOW_FREQUENCY_POWER,
    HRV_STATISTIC_HIGH_FREQUENCY_POWER,
    HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO,
]

HRV_STATISTICS = HRV_TIME_DOMAIN_STATISTICS + HRV_FREQUENCY_DOMAIN_STATISTICS


def get_night_hrv_rmssd(
    loader: BaseLoader,
    user_id: Union[str, list, None] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, str, None] = None,
    coverage: float = 0.7,
    loader_kwargs: dict = {},
    backend: Literal["pywearable", "neurokit"] = "pywearable",
) -> dict:
    """Computes RMSSD metric considering night data.

    This function computes the root mean squared successive difference (RMSSD),
    taking in consideration only periods where the user was asleep.
    Depending on the value of the ``kind`` parameter, this function
    returns the mean of RMSSD short-term (5 mins) computations overnight
    for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which RMSSD must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.
    kind : :class:`str` or None, optional
        Whether to transform RMSSD over days, or to return the value for each day, by default None.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and RMSSD as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `RMSSD`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """

    return get_hrv_statistic(
        loader=loader,
        user_id=user_id,
        statistic=HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        loader_kwargs=loader_kwargs,
        period="night",
        backend=backend,
    )


def get_night_hrv_sdnn(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    coverage: float = 0.7,
    loader_kwargs: dict = {},
    backend: Literal["pywearable", "neurokit"] = "pywearable",
) -> dict:
    """Computes SDNN metric considering night data.

    This function computes the standard deviation of normal to normal (SDNN) intervals,
    taking in consideration only periods where the user was asleep.
    Depending on the value of the ``kind`` parameter, this function
    returns the mean of SDNN short term computations (5 mins) overnight
    for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which SDNN must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.
    kind : :class:`str` or None, optional
        Whether to transform SDNN over days, or to return the value for each day, by default None.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and SDNN as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `SDNN`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """

    return get_hrv_statistic(
        loader=loader,
        user_id=user_id,
        statistic=HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        loader_kwargs=loader_kwargs,
        period="night",
        backend=backend,
    )


def get_night_hrv_vlf_power(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    coverage: float = 0.7,
    loader_kwargs={},
    backend: Literal["pywearable", "neurokit"] = "pywearable",
) -> Union[dict, pd.DataFrame]:
    """Compute LF power metrics considering night data.

    This function computes the low frequency (LF, 0.04Hz - 0.15Hz) power,
    taking in consideration only periods where the user was asleep.
    Depending on the value of the ``kind`` parameter, this function
    returns the mean of LF short-term (5 mins) computations overnight
    for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which LF must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.
    kind : :class:`str` or None, optional
        Whether to transform LF over days, or to return the value for each day, by default None.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and LF as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `LF`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_hrv_statistic(
        loader=loader,
        user_id=user_id,
        statistic=HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        loader_kwargs=loader_kwargs,
        backend=backend,
        period="night",
    )


def get_night_hrv_lf_power(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    coverage: float = 0.7,
    loader_kwargs={},
    backend: Literal["pywearable", "neurokit"] = "pywearable",
) -> Union[dict, pd.DataFrame]:
    """Compute LF power metrics considering night data.

    This function computes the low frequency (LF, 0.04Hz - 0.15Hz) power,
    taking in consideration only periods where the user was asleep.
    Depending on the value of the ``kind`` parameter, this function
    returns the mean of LF short-term (5 mins) computations overnight
    for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which LF must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.
    kind : :class:`str` or None, optional
        Whether to transform LF over days, or to return the value for each day, by default None.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and LF as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `LF`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """
    return get_hrv_statistic(
        loader=loader,
        user_id=user_id,
        statistic=HRV_STATISTIC_LOW_FREQUENCY_POWER,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        loader_kwargs=loader_kwargs,
        backend=backend,
        period="night",
    )


def get_night_hrv_hf_power(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    coverage: float = 0.7,
    loader_kwargs={},
    backend: Literal["pywearable", "neurokit"] = "pywearable",
) -> Union[dict, pd.DataFrame]:
    """Compute HF power metrics considering night data.

    This function computes the low frequency (HF, 0.15Hz - 0.4Hz) power,
    taking in consideration only periods where the user was asleep.
    Depending on the value of the ``kind`` parameter, this function
    returns the mean of HF short-term (5 mins) computations overnight
    for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which HF must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.
    kind : :class:`str` or None, optional
        Whether to transform HF over days, or to return the value for each day, by default None.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and HF as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `HF`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """

    return get_hrv_statistic(
        loader=loader,
        user_id=user_id,
        statistic=HRV_STATISTIC_HIGH_FREQUENCY_POWER,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        loader_kwargs=loader_kwargs,
        period="night",
        backend=backend,
    )


def get_night_lfhf(
    loader: BaseLoader,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    coverage: float = 0.7,
    loader_kwargs={},
    backend: Literal["pywearable", "neurokit"] = "pywearable",
) -> dict:
    """Compute LF/HF power ratio metrics considering night data.

    This function computes the ratio between low frequency (LF) and high frequency (HF) power,
    taking in consideration only periods where the user was asleep.
    Depending on the value of the ``kind`` parameter, this function
    returns the mean of LFHF short-term (5 mins) computations overnight
    for each calendar day (``kind=None``) from ``start_date`` to
    ``end_date`` or the transformed value across all days.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str` or :class:`list`, optional
        The id(s) for which LFHF must be retrieved, by default "all".
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.
    kind : :class:`str` or None, optional
        Whether to transform LFHF over days, or to return the value for each day, by default None.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        If ``kind==None``, dictionary with ``user_id`` as key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and LFHF as values.
        If ``kind!=None``, dictionary with ``user_id`` as key, and a nested dictionary with `LFHF`
        as key and its transformed value, and an additional `days` keys that contains an array of all
        calendar days over which the transformation was computed.
    """

    return get_hrv_statistic(
        loader=loader,
        user_id=user_id,
        statistic=HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        loader_kwargs=loader_kwargs,
        period="night",
        backend=backend,
    )


def _compute_hrv_statistic(
    bbi_dict: dict, metric: str, coverage: float = 0.7, **kwargs
) -> dict:
    """Computes a HRV statistic at a per night level

    This function computes a specific statistic for
    the BBI data available for each night.

    Parameters
    ----------
    bbi_dict: :class:`dict`
        Dictionary of BBI night data, already filtered to include only sleep periods.
    metric : :class:`str`
        Name of the statistic to be computed
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        dictionary with ``user_id`` as primary key, and a nested dictionary with
        calendar days (:class:`datetime.date`) as keys and the cardiac metric as values.

     Raises
    ------
    ValueError
        if `metric` isn't one of "RMSSD", "SDNN", "LF", "HF", "LFHF"
    """

    daily_means = {}

    for date, bbi_df in bbi_dict.items():
        bbi_df = bbi_df.set_index(constants.ISODATE_COL)
        # we consider the coverage against the mean bbi in that window
        counts = bbi_df.resample("5min").bbi.count()
        means = bbi_df.resample("5min").bbi.mean()
        coverage_filter = (counts > ((300 / (means / 1000)) * coverage)).values

        # time-domain
        if metric == CARDIAC_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES:
            ST_analysis = bbi_df.resample("5min").bbi.apply(
                lambda x: np.sqrt(np.mean(np.diff(x) ** 2)) if x.count() > 5 else 0
            )
        elif metric == CARDIAC_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL:
            ST_analysis = bbi_df.resample("5min").bbi.apply(
                lambda x: np.std(x, ddof=1) if x.count() > 5 else 0
            )
        # frequency domain
        elif metric in CARDIAC_HRV_FREQUENCY_DOMAIN_STATISTICS:
            LF_range = (0.04, 0.15)
            HF_range = (0.15, 0.40)

            def calculate_power(bbi, freq_range):
                """Calculates power in freq band between `f1` and `f2` given an array of beat-to-beat intervals"""

                # Resampling (with 4Hz) and interpolate
                # Because RRi are unevenly spaced we must interpolate it for accurate PSD estimation.

                fs = 4
                t = np.cumsum(bbi)
                t -= t.iloc[0]
                f_interpol = interpolate.interp1d(
                    t, bbi, kind="cubic", fill_value="extrapolate"
                )
                t_interpol = np.arange(t.iloc[0], t.iloc[-1], 1000.0 / fs)
                nn_interpol = f_interpol(t_interpol)

                # Subtract mean value from each sample for suppression of DC-offsets
                nn_interpol = nn_interpol - np.mean(nn_interpol)
                # Calculate power spectral density using Welch method
                nfft = 2**12
                if t.max() < 300000:
                    nperseg = min(nfft, len(nn_interpol))
                else:
                    nperseg = 300
                freq, psd = signal.welch(
                    x=nn_interpol,
                    fs=fs,
                    window="hamming",
                    nfft=nfft,
                    nperseg=nperseg,
                    scaling="density",
                )

                f1, f2 = freq_range
                pow = np.trapz(
                    psd[np.logical_and(freq >= f1, freq < f2)],
                    freq[np.logical_and(freq >= f1, freq < f2)],
                )

                return pow

            if metric == CARDIAC_STATISTIC_HIGH_FREQUENCY:
                ST_analysis = bbi_df.resample("5min").bbi.apply(
                    lambda x: calculate_power(x, HF_range) if x.count() > 5 else 0
                )
            elif metric == CARDIAC_STATISTIC_LOW_FREQUENCY:
                ST_analysis = bbi_df.resample("5min").bbi.apply(
                    lambda x: calculate_power(x, LF_range) if x.count() > 5 else 0
                )
            else:
                # note that here we're computing LFHF ratios over 5 minutes and then getting the average
                # it's not the same as computing the ratio of the mean LF and mean HF over a night
                # if one wants the latter, it should be done as post processing by getting LF and HF overnight metrics
                ST_analysis = bbi_df.resample("5min").bbi.apply(
                    lambda x: (
                        calculate_power(x, LF_range) / calculate_power(x, HF_range)
                        if x.count() > 5
                        else 0
                    )
                )
        else:
            raise ValueError(
                f"{metric} is not a valid value. Select among {CARDIAC_HRV_STATISTICS}"
            )
        # check coverage
        ST_analysis = ST_analysis.iloc[coverage_filter & (ST_analysis != 0).values]
        # compute the mean overnight
        if len(ST_analysis) > 0:
            daily_mean = ST_analysis.values.mean()
            if daily_mean is not None:
                daily_means[date] = round(daily_mean, 2)
            else:
                daily_means[date] = np.nan
        else:
            daily_means[date] = np.nan
    daily_means = pd.Series(daily_means)
    daily_means.index = pd.to_datetime(daily_means.index)
    return daily_means


def _compute_rmssd(bbi_dict: dict, coverage: float = 0.7, **kwargs) -> dict:
    """Compute night RMSSD from BBI data.

    Parameters
    ----------
    bbi_dict: :class:`dict`
        Dictionary of BBI night data, already filtered to include only sleep periods.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        A dictionary with night RMSSD as values, and calendar dates as keys.
    """
    return _compute_hrv_statistic(
        bbi_dict=bbi_dict,
        metric=CARDIAC_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES,
        coverage=coverage,
    )


def _compute_sdnn(bbi_dict: dict, coverage: float = 0.7, **kwargs) -> dict:
    """Compute night SDNN from BBI data.

    Parameters
    ----------
    bbi_dict: :class:`dict`
        Dictionary of BBI night data, already filtered to include only sleep periods.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        A dictionary with night SDNN as values, and calendar dates as keys.
    """
    return _compute_hrv_statistic(
        bbi_dict=bbi_dict,
        metric=CARDIAC_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL,
        coverage=coverage,
    )


def _compute_lf(bbi_dict: dict, coverage: float = 0.7, **kwargs) -> dict:
    """Compute night LF power from BBI data.

    Parameters
    ----------
    bbi_dict: :class:`dict`
        Dictionary of BBI night data, already filtered to include only sleep periods.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        A dictionary with night LF power as values, and calendar dates as keys.
    """
    return _compute_hrv_statistic(
        bbi_dict=bbi_dict, metric=CARDIAC_STATISTIC_LOW_FREQUENCY, coverage=coverage
    )


def _compute_hf(bbi_dict: dict, coverage: float = 0.7, **kwargs) -> dict:
    """Compute night HF power from BBI data.

    Parameters
    ----------
    bbi_dict: :class:`dict`
        Dictionary of BBI night data, already filtered to include only sleep periods.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        A dictionary with night HF power as values, and calendar dates as keys.
    """
    return _compute_hrv_statistic(
        bbi_dict=bbi_dict, metric=CARDIAC_STATISTIC_HIGH_FREQUENCY, coverage=coverage
    )


def _compute_lfhf(bbi_dict: dict, coverage: float = 0.7, **kwargs) -> dict:
    """Compute night LFHF ratios from BBI data.

    Parameters
    ----------
    bbi_dict: :class:`dict`
        Dictionary of BBI night data, already filtered to include only sleep periods.
    coverage : :class:`float`, optional
        The percentage of expected bbi observations for a period to be considered in the analysis, by default 0.7.

    Returns
    -------
    :class:`dict`
        A dictionary with night LFHF ratios as values, and calendar dates as keys.
    """
    return _compute_hrv_statistic(
        bbi_dict=bbi_dict,
        metric=CARDIAC_STATISTIC_LOW_HIGH_FREQUENCY_RATIO,
        coverage=coverage,
    )


def _compute_bbi_wearing_time(bbi_dict: dict, **kwargs) -> pd.Series:
    if len(bbi_dict) > 0:
        bbi_df_list = []
        for k, v in bbi_dict.items():
            tmp_df = pd.DataFrame({"bbi": v["bbi"], "calendarDate": k})
            bbi_df_list.append(tmp_df)
        bbi_df = pd.concat(bbi_df_list, axis=0)
        return pd.to_timedelta(bbi_df.groupby("calendarDate")["bbi"].sum(), unit="ms")

    else:
        return pd.Series()


def _compute_expected_bbi_wearing_time(
    sleep_summary: pd.DataFrame, **kwargs
) -> pd.Series:
    if len(sleep_summary) > 0:
        return sleep_summary["wakeup"] - sleep_summary[constants.ISODATE_COL]
    else:
        return pd.Series()


_HRV_STATISTICS_DICT = {
    HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER: _compute_hrv_power,
    HRV_STATISTIC_LOW_FREQUENCY_POWER: _compute_hrv_power,
    HRV_STATISTIC_HIGH_FREQUENCY_POWER: _compute_hrv_power,
    HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO: _compute_hrv_power,
    HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES: _compute_rmssd,
    HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL: _compute_sdnn,
    "night_bbi_wearing_time": _compute_bbi_wearing_time,
    "night_bbi_expected_wearing_time": _compute_expected_bbi_wearing_time,
}


def get_hrv_statistic(
    loader: BaseLoader,
    statistic: str,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    coverage: float = 0.7,
    kind=None,
    kind_args: list = [],
    kind_kwargs: dict = {},
    loader_kwargs: dict = {},
    return_df: bool = True,
    return_multi_index: bool = True,
):
    return get_hrv_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        kind=kind,
        kind_args=kind_args,
        kind_kwargs=kind_kwargs,
        loader_kwargs=loader_kwargs,
        return_df=return_df,
        return_multi_index=return_multi_index,
    )


def get_hrv_statistics(
    loader: BaseLoader,
    user_id: Union[str, list, None],
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    statistic: Union[str, list, tuple, None] = None,
    coverage: float = 0.7,
    loader_kwargs: dict = {},
    sleep_stages_to_remove: Union[list, None] = ["awake", "unmeasurable"],
    overlap_days: int = 7,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    period: Literal["day", "night", "all"] = "all",
):
    # Get user id from the loader
    if user_id is None:
        user_id = "all"
    user_ids = utils.get_user_ids(loader, user_id)
    # Convert to datetime
    start_date = pd.Timestamp(start_date) if not (start_date is None) else None
    end_date = pd.Timestamp(end_date) if not (end_date is None) else None

    # Check metrics
    if statistic is None:
        # Consider all metrics
        statistics = list(_HRV_STATISTICS_DICT.keys())
    elif type(statistic) == str:
        statistics = [statistic]

    # Check args and kwargs
    if type(loader_kwargs) != dict:
        raise ValueError(
            f"loader_kwargs must be of type dict, not {type(loader_kwargs)}"
        )

    hrv_stats_df_list = []

    for user_id in user_ids:
        # TODO move this at the end only if period is day/night
        sleep_summary = (
            loader.load_sleep_summary(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                drop_duplicates=True,
            )
            .sort_values(constants.CALENDAR_DATE_COL, ascending=True)
            .set_index(constants.CALENDAR_DATE_COL)
        )
        sleep_summary["wakeup"] = sleep_summary[
            constants.ISODATE_COL
        ] + pd.to_timedelta(
            sleep_summary[constants.DURATION_IN_MS_COL]
            + sleep_summary[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL],
            unit="ms",
        )

        # load the whole bbi history required, with some room (+- 12 hrs at start/end)
        # to account for timezones

        intervals = math.ceil(len(sleep_summary) / (overlap_days))
        user_hrv_stats_df_list = []
        for i in range(intervals):
            tmp_start_date = sleep_summary.iloc[
                i * overlap_days : (i + 1) * overlap_days, :
            ][constants.ISODATE_COL].min()
            tmp_end_date = sleep_summary.iloc[
                i * overlap_days : (i + 1) * overlap_days, :
            ]["wakeup"].max() + pd.Timedelta(microseconds=1)

            bbi_df = loader.load_bbi(
                user_id=user_id,
                start_date=tmp_start_date,
                end_date=tmp_end_date,
                **loader_kwargs,
            )
            user_interval_hrv_stats_df_list = []
            if len(bbi_df) > 0:
                filtered_night_bbi = utils.filter_out_sleep_stages(
                    loader=loader,
                    user_id=user_id,
                    df=bbi_df,
                    stages_to_filter=sleep_stages_to_remove,
                )
                bbi_dict = {}
                for idx, row in sleep_summary.iloc[
                    i * overlap_days : (i + 1) * overlap_days, :
                ].iterrows():
                    bbi_dict[idx] = filtered_night_bbi.loc[
                        (
                            filtered_night_bbi[constants.ISODATE_COL]
                            >= row[constants.ISODATE_COL]
                        )
                        & (filtered_night_bbi[constants.ISODATE_COL] <= row["wakeup"])
                    ]

                for statistic in statistics:
                    # Get series with metrics by calendarDate
                    ser: pd.Series = _HRV_STATISTICS_DICT[statistic](
                        bbi_dict=bbi_dict,
                        coverage=coverage,
                        sleep_summary=sleep_summary.iloc[
                            i * overlap_days : (i + 1) * overlap_days, :
                        ],
                    ).rename(statistic)
                    user_interval_hrv_stats_df_list.append(ser)
                if len(user_interval_hrv_stats_df_list) > 0:
                    user_hrv_stats_df_list.append(
                        pd.concat(user_interval_hrv_stats_df_list, axis=1)
                    )
        if len(user_hrv_stats_df_list) > 0:
            user_hrv_stats_df = pd.concat(user_hrv_stats_df_list, axis=0)
        else:
            user_hrv_stats_df = (
                pd.DataFrame(columns=statistics)
                if len(statistics) > 1
                else pd.Series(name=statistics[0])
            )
        user_hrv_stats_df = user_hrv_stats_df.reindex(
            pd.date_range(
                start=start_date, end=end_date, inclusive="left", normalize=True
            )
        )
        hrv_stats_df_list.append(user_hrv_stats_df)
    hrv_stats_df = pd.concat(hrv_stats_df_list, axis=0)

    if len(statistics) == 1:
        return hrv_stats_df[statistic[0]]
    else:
        return hrv_stats_df
