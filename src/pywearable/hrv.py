"""
This module contains all the functions related to the analysis
of cardiac data, mostly for heart rate (HR) and heart rate
variability (HRV).
"""

import datetime
import math
import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger
from scipy import interpolate, signal

from . import constants  # , sleep, utils
from .loader.base import BaseLoader
from .utils import time as time_utils

try:
    import neurokit2 as nk
except:
    warnings.warn("Neurokit was not found")

# TODO possibility of filtering by day/night
# TODO return multiple statistics by night/day
# TODO add parameter for base_window
# TODO check if loader has hrv_features already
# TODO compute all power statistics at once

HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES = "hrv_rmssd"
HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL = "hrv_sdnn"
HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER = "hrv_vlf_power"
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


def compute_rmssd(x):
    return np.sqrt(np.mean(np.diff(x) ** 2)) if len(x) > 5 else np.nan


def compute_sdnn(x):
    return np.std(x, ddof=1) if len(x) > 5 else 0


def compute_pnn20(x):
    x = np.asarray(x)

    if x.size < 2:
        return np.nan

    diff = np.abs(np.diff(x))

    nn20 = np.sum(diff > 20)

    total = diff.size + 1

    return nn20 / total * 100


def compute_pnn50(x):
    x = np.asarray(x)

    if x.size < 2:
        return np.nan

    diff = np.abs(np.diff(x))

    nn50 = np.sum(diff > 50)

    total = diff.size + 1

    return nn50 / total * 100


def compute_hrv_statistics(
    bbi: pd.DataFrame,
    statistics: Union[list, str, None] = None,
    backend: str = "pywearable",
    coverage: float = 0.7,
    window_size: Union[str, pd.Timedelta] = pd.Timedelta("5min"),
) -> pd.DataFrame:
    # We need to extract hrv features with overlapping windows
    # using 5 minute resolution

    time_grid = pd.date_range(
        start=bbi["utc_time"].min().floor("min"),
        end=bbi["utc_time"].max().ceil("min"),
        freq="1min",
    )

    time_features = []
    timezone_info = []
    frequency_features = []

    left = 0
    right = 0

    window_before = pd.Timedelta("5min")
    bbi = bbi.set_index("utc_time")
    bbi_values = bbi["bbi"].values
    times = bbi.index.values.astype("datetime64[ns]")
    timezones = bbi["timezone_offset"].values
    n = len(times)

    compute_time_features = any(
        stat in HRV_TIME_DOMAIN_STATISTICS for stat in statistics
    )
    compute_frequency_features = any(
        stat in HRV_FREQUENCY_DOMAIN_STATISTICS for stat in statistics
    )

    for t in time_grid:
        start = t - window_before
        end = t

        # logger.info(f"Extracting features on window: {start}-{end}")

        # move left pointer
        while left < n and times[left] < start:
            left += 1

        # move right pointer
        while right < n and times[right] < end:
            right += 1
        w = bbi_values[left:right]
        window_times = times[left:right]
        window_timezones = timezones[left:right]
        if len(w) > 0:

            timezone_info.append(
                {
                    "utc_time": t,
                    "timezone_offset": window_timezones[0],
                }
            )
            if compute_time_features:
                if backend == "pywearable":
                    window_time_features = {}
                    window_time_features["utc_time"] = t
                    window_time_features["hrv_min_nn"] = np.nanmin(w)
                    window_time_features["hrv_mean_nn"] = np.nanmean(w)
                    window_time_features["hrv_median_nn"] = np.nanmedian(w)
                    window_time_features["hrv_max_nn"] = np.nanmax(w)

                    window_time_features["hrv_sdnn"] = compute_sdnn(w)
                    window_time_features["hrv_rmssd"] = compute_rmssd(w)
                    window_time_features["hrv_pnn20"] = compute_pnn20(w)
                    window_time_features["hrv_pnn50"] = compute_pnn50(w)

                    window_time_features["hrv_prc20nn"] = np.nanpercentile(w, q=20)
                    window_time_features["hrv_prc80nn"] = np.nanpercentile(w, q=80)

                    window_time_features["hrv_duration"] = (
                        pd.to_timedelta(w.sum(), unit="ms"),
                    )
                    window_time_features["hrv_count"] = len(w)

                    window_time_features["hrv_sdrmmsd"] = (
                        window_time_features["hrv_sdnn"]
                        / window_time_features["hrv_rmssd"]
                    )  # Sollers (2007)

                    window_time_features["hrv_cv_nn"] = (
                        window_time_features["hrv_sdnn"]
                        / window_time_features["hrv_mean_nn"]
                    )
                    window_time_features["hrv_cv_sd"] = (
                        window_time_features["hrv_rmssd"]
                        / window_time_features["hrv_mean_nn"]
                    )
                    window_time_features["hrv_iqr_nn"] = scipy.stats.iqr(w)

                    window_time_features = pd.DataFrame.from_dict(
                        window_time_features, orient="columns"
                    )

                elif backend == "neurokit":
                    window_time_features: pd.DataFrame = nk.hrv_time(
                        {"RRI": w, "RRI_Times": window_times}
                    )

                    window_time_features["utc_time"] = t
                    window_time_features["hrv_duration"] = pd.to_timedelta(
                        w.sum(), unit="ms"
                    )

                time_features.append(window_time_features)

            if compute_frequency_features:
                if backend == "neurokit":
                    window_frequency_features: pd.DataFrame = nk.hrv_frequency(
                        {"RRI": w, "RRI_Times": window_times}
                    )

                    window_frequency_features["utc_time"] = t

                frequency_features.append(window_frequency_features)

    if compute_time_features:
        if len(time_features) > 0:

            hrv_time_features = (
                pd.concat(time_features, axis=0, ignore_index=True)
                .set_index("utc_time")
                .rename_axis(None, axis=1)
            )
            if backend == "neurokit":
                hrv_time_features = hrv_time_features.drop(
                    [
                        "HRV_SDNNI1",
                        "HRV_SDANN1",
                        "HRV_SDANN2",
                        "HRV_SDNNI2",
                        "HRV_SDANN5",
                        "HRV_SDNNI5",
                        "HRV_HTI",
                        "HRV_TINN",
                    ],
                    axis=1,
                )
                hrv_time_features.columns = hrv_time_features.columns.str.lower()
                hrv_time_features = hrv_time_features.rename(
                    columns={
                        "hrv_meannn": "hrv_mean_nn",
                        "hrv_mediannn": "hrv_median_nn",
                        "hrv_minnn": "hrv_min_nn",
                        "hrv_maxnn": "hrv_max_nn",
                    }
                )
    if compute_frequency_features:
        if len(frequency_features) > 0:
            hrv_frequency_features = (
                pd.concat(frequency_features, axis=0, ignore_index=True)
                .set_index("utc_time")
                .rename_axis(None, axis=1)
            )
        # TODO merge with frequency features
        # TODO resample with 1 minute resolution

    hrv_features = pd.concat(
        (
            pd.DataFrame.from_dict(timezone_info)
            .set_index("utc_time")
            .rename_axis(None, axis=1),
            hrv_time_features,
        ),
        axis=1,
    )
    print(hrv_features)

    return hrv_time_features


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


def get_hrv_statistic(
    loader: BaseLoader,
    statistic: str,
    user_id: Union[str, list] = "all",
    start_date: Union[pd.Timestamp, None] = None,
    end_date: Union[pd.Timestamp, None] = None,
    coverage: float = 0.7,
    loader_kwargs: dict = {},
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    period: Literal["awake", "sleep", "all"] = "all",
    resolution: Union[pd.Timedelta, str] = "day",
):
    return get_hrv_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        coverage=coverage,
        loader_kwargs=loader_kwargs,
        backend=backend,
        period=period,
        resolution=resolution,
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
    overlap_days: Union[int, None] = 7,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    period: Literal["day", "night", "all"] = "all",
    resolution: Union[pd.Timedelta, str] = "day",
):
    # Get user id from the loader
    if user_id is None:
        user_ids = loader.get_user_ids()
    elif type(user_id) is str:
        user_ids = [user_id]
    else:
        user_ids = user_id

    # Convert to datetime
    start_date = pd.Timestamp(start_date) if not (start_date is None) else None
    end_date = pd.Timestamp(end_date) if not (end_date is None) else None

    # Check metrics
    if statistic is None:
        # Consider all metrics
        statistics = HRV_STATISTICS
    elif type(statistic) == str:
        statistics = [statistic]

    # Check args and kwargs
    if type(loader_kwargs) != dict:
        raise ValueError(
            f"loader_kwargs must be of type dict, not {type(loader_kwargs)}"
        )

    hrv_stats_df_list = []

    if (start_date is None) and (end_date is None):
        intervals = 1
    else:
        normalized_start_date = pd.Timestamp(start_date).normalize()
        normalized_end_date = pd.Timestamp(end_date).normalize()
        if end_date.hour > 0:
            normalized_end_date += pd.Timedelta(hours=1)
        intervals = int(
            (normalized_end_date - normalized_start_date).days / (overlap_days)
        )

    # TODO do this at the end when filtering (?)
    # Check if we need to filter sleep stages
    filter_sleep_stages = False
    if type(sleep_stages_to_remove) is list:
        if len(sleep_stages_to_remove) > 0:
            filter_sleep_stages = True
    elif type(sleep_stages_to_remove) is str:
        sleep_stages_to_remove = [sleep_stages_to_remove]
        filter_sleep_stages = True

    for user_id in user_ids:

        # load the whole bbi history required, with some room (+- 12 hrs at start/end)
        # to account for timezones
        user_hrv_stats_df_list = []
        for i in range(intervals):

            tmp_start_date = (
                normalized_start_date + pd.Timedelta(i * overlap_days, unit="d")
                if not (start_date is None)
                else None
            )

            tmp_end_date = (
                tmp_start_date + pd.Timedelta((i + 1) * overlap_days, unit="d")
                if not (start_date is None)
                else None
            )

            logger.info(f"Loading BBI from {tmp_start_date} to {tmp_end_date}")

            bbi_df = loader.load_bbi(
                user_id=user_id,
                start_date=tmp_start_date,
                end_date=tmp_end_date,
                **loader_kwargs,
            )
            logger.info(f"Total length of BBI: {len(bbi_df)} rows")
            logger.info(f"First date: {bbi_df["local_time"].min()}")
            logger.info(f"Last date: {bbi_df["local_time"].max()}")

            if len(bbi_df) > 0:
                if filter_sleep_stages:
                    bbi_df = utils.filter_out_sleep_stages(
                        loader=loader,
                        user_id=user_id,
                        df=bbi_df,
                        stages_to_filter=sleep_stages_to_remove,
                    )

                # Get series with metrics computed over 5-minute windows
                # split in time/frequency domain
                user_hrv_stats = compute_hrv_statistics(
                    bbi=bbi_df,
                    coverage=coverage,
                    backend=backend,
                    statistics=statistics,
                )
                if len(user_hrv_stats) > 0:
                    user_hrv_stats_df_list.append(user_hrv_stats)

        if len(user_hrv_stats_df_list) > 0:
            user_hrv_stats_df = pd.concat(user_hrv_stats_df_list, axis=0).assign(
                user_id=user_id
            )
        else:
            user_hrv_stats_df = (
                pd.DataFrame(columns=statistics)
                if len(statistics) > 1
                else pd.Series(name=statistics[0])
            )
        # TODO not always start and end date are populated
        user_hrv_stats_df = user_hrv_stats_df.reindex(
            pd.date_range(
                start=start_date,
                end=end_date,
                inclusive="left",
                # normalize=True,
                freq="1min",
            )
        )

        hrv_stats_df_list.append(user_hrv_stats_df)
    hrv_stats_df = pd.concat(hrv_stats_df_list, axis=0)
    print(hrv_stats_df)
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
    sleep_summary["wakeup"] = sleep_summary[constants.ISODATE_COL] + pd.to_timedelta(
        sleep_summary[constants.DURATION_IN_MS_COL]
        + sleep_summary[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL],
        unit="ms",
    )

    if len(statistics) == 1:
        return hrv_stats_df[statistic[0]]
    else:
        return hrv_stats_df
