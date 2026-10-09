"""Heart-rate-variability features from BBI or provider estimates."""

from __future__ import annotations

import datetime
import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, signal

from . import cycle, hierarchy, schema
from .loader.base import BaseLoader


HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES = "hrv_rmssd"
HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL = "hrv_sdnn"
HRV_STATISTIC_PNN20 = "hrv_pnn20"
HRV_STATISTIC_PNN50 = "hrv_pnn50"
HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER = "hrv_vlf_power"
HRV_STATISTIC_LOW_FREQUENCY_POWER = "hrv_lf_power"
HRV_STATISTIC_HIGH_FREQUENCY_POWER = "hrv_hf_power"
HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO = "hrv_lfhf_ratio"
HRV_STATISTIC_BBI_MEDIAN = "hrv_bbi_median"
HRV_STATISTIC_MIN_NN = "hrv_min_nn"
HRV_STATISTIC_MEAN_NN = "hrv_mean_nn"
HRV_STATISTIC_MAX_NN = "hrv_max_nn"
HRV_STATISTIC_SDNN_RMSSD_RATIO = "hrv_sdnn_rmssd_ratio"
HRV_STATISTIC_CVNN = "hrv_cv_nn"
HRV_STATISTIC_CVSD = "hrv_cv_sd"
HRV_STATISTIC_IQRNN = "hrv_iqr_nn"

HRV_TIME_DOMAIN_STATISTICS = (
    HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES,
    HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL,
    HRV_STATISTIC_PNN20,
    HRV_STATISTIC_PNN50,
    HRV_STATISTIC_BBI_MEDIAN,
    HRV_STATISTIC_MIN_NN,
    HRV_STATISTIC_MEAN_NN,
    HRV_STATISTIC_MAX_NN,
    HRV_STATISTIC_SDNN_RMSSD_RATIO,
    HRV_STATISTIC_CVNN,
    HRV_STATISTIC_CVSD,
    HRV_STATISTIC_IQRNN,
)
HRV_FREQUENCY_DOMAIN_STATISTICS = (
    HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER,
    HRV_STATISTIC_LOW_FREQUENCY_POWER,
    HRV_STATISTIC_HIGH_FREQUENCY_POWER,
    HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO,
)
HRV_STATISTICS = HRV_TIME_DOMAIN_STATISTICS + HRV_FREQUENCY_DOMAIN_STATISTICS

NEUROKIT_TIME_DOMAIN_STATISTICS = HRV_TIME_DOMAIN_STATISTICS + (
    "hrv_sdann_1min",
    "hrv_sdnni_1min",
    "hrv_sdann_2min",
    "hrv_sdnni_2min",
    "hrv_sdann_5min",
    "hrv_sdnni_5min",
    "hrv_sdsd",
    "hrv_mad_nn",
    "hrv_mcv_nn",
    "hrv_prc20_nn",
    "hrv_prc80_nn",
    "hrv_hti",
    "hrv_tinn",
)
NEUROKIT_FREQUENCY_DOMAIN_STATISTICS = HRV_FREQUENCY_DOMAIN_STATISTICS + (
    "hrv_ulf_power",
    "hrv_vhf_power",
    "hrv_total_power",
    "hrv_lf_normalized",
    "hrv_hf_normalized",
    "hrv_log_hf",
)
NEUROKIT_HRV_STATISTICS = (
    NEUROKIT_TIME_DOMAIN_STATISTICS + NEUROKIT_FREQUENCY_DOMAIN_STATISTICS
)

HRV_PRIMARY_RESOLUTION = "5min"
DEFAULT_SLEEP_STAGES_TO_REMOVE = (
    schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
    schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
)
_CONTINUITY_SEGMENT_COL = "_hrv_continuity_segment"
_SLEEP_STAGE_SEGMENT_COL = "_hrv_sleep_stage_segment"
_PROVIDER_COLUMNS = {
    HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES: schema.HRV_RMSSD_COL,
    HRV_STATISTIC_LOW_FREQUENCY_POWER: schema.HRV_LF_POWER_COL,
    HRV_STATISTIC_HIGH_FREQUENCY_POWER: schema.HRV_HF_POWER_COL,
}


def _valid_bbi_inputs(values, segment_ids=None, timestamps=None):
    """Return aligned valid BBIs, segment labels, and timestamps."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0)
    segments = None
    if segment_ids is not None:
        segments = np.asarray(segment_ids)
        if segments.size != values.size:
            raise ValueError("segment_ids must contain one value per BBI interval")
    times = None
    if timestamps is not None:
        times = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        if times.size != values.size:
            raise ValueError("timestamps must contain one value per BBI interval")
        valid &= times.notna().to_numpy()
    return (
        values[valid],
        None if segments is None else segments[valid],
        None if times is None else pd.DatetimeIndex(times[valid]),
    )


def _valid_bbi_values_and_segments(values, segment_ids=None):
    """Return valid positive BBIs and their aligned segment labels."""
    values, segments, _ = _valid_bbi_inputs(values, segment_ids)
    return values, segments


def _valid_bbi_values(values) -> np.ndarray:
    return _valid_bbi_values_and_segments(values)[0]


def _successive_differences(values, segment_ids=None):
    """Return BBIs and differences between adjacent values in the same segment."""
    values, segments = _valid_bbi_values_and_segments(values, segment_ids)
    if values.size < 2:
        return values, np.array([], dtype=float)
    differences = np.diff(values)
    if segments is not None:
        differences = differences[segments[1:] == segments[:-1]]
    return values, differences


def _compute_rmssd(values, segment_ids=None):
    values, differences = _successive_differences(values, segment_ids)
    if values.size <= 5 or differences.size == 0:
        return np.nan
    return np.sqrt(np.mean(differences**2))


def _compute_pnn(values, threshold, segment_ids=None):
    values, differences = _successive_differences(values, segment_ids)
    if values.size < 2 or differences.size == 0:
        return np.nan
    return np.sum(np.abs(differences) > threshold) / differences.size * 100


def compute_rmssd(values):
    """Return RMSSD in milliseconds, or missing for fewer than six intervals."""
    return _compute_rmssd(values)


def compute_sdnn(values):
    """Return sample SDNN in milliseconds, or missing for insufficient data."""
    values = _valid_bbi_values(values)
    return np.std(values, ddof=1) if values.size > 5 else np.nan


def compute_pnn20(values):
    """Return the percentage of successive differences exceeding 20 ms."""
    return _compute_pnn(values, 20)


def compute_pnn50(values):
    """Return the percentage of successive differences exceeding 50 ms."""
    return _compute_pnn(values, 50)


def _power_spectrum(values, timestamps=None) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate irregular BBI at 4 Hz and calculate one Welch spectrum."""
    values, _, timestamps = _valid_bbi_inputs(values, timestamps=timestamps)
    if values.size <= 5:
        return None
    if timestamps is None:
        times = np.cumsum(values) / 1000
        times -= times[0]
    else:
        times = (timestamps - timestamps[0]).total_seconds().to_numpy()
        if np.any(np.diff(times) <= 0):
            return None
    if times[-1] <= 0:
        return None
    interpolator = interpolate.interp1d(
        times,
        values,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    regular_times = np.arange(0, times[-1], 0.25)
    if regular_times.size < 4:
        return None
    regular_values = interpolator(regular_times)
    regular_values -= regular_values.mean()
    frequencies, density = signal.welch(
        regular_values,
        fs=4,
        window="hamming",
        nperseg=min(4096, regular_values.size),
        nfft=4096,
        scaling="density",
    )
    return frequencies, density


def _band_power(spectrum, low: float, high: float) -> float:
    if spectrum is None:
        return np.nan
    frequencies, density = spectrum
    selected = (frequencies >= low) & (frequencies < high)
    if selected.sum() < 2:
        return np.nan
    return float(integrate.trapezoid(density[selected], frequencies[selected]))


def _pywearable_window_statistics(
    values,
    statistics,
    segment_ids=None,
    timestamps=None,
    allow_frequency=True,
) -> dict:
    values, segments, timestamps = _valid_bbi_inputs(
        values, segment_ids, timestamps
    )
    result = {name: np.nan for name in statistics}
    mean_nn = float(np.mean(values)) if values.size else np.nan
    sdnn = compute_sdnn(values)
    rmssd = _compute_rmssd(values, segments)
    operations = {
        HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES: lambda data: (
            rmssd
        ),
        HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL: lambda data: sdnn,
        HRV_STATISTIC_PNN20: lambda data: _compute_pnn(data, 20, segments),
        HRV_STATISTIC_PNN50: lambda data: _compute_pnn(data, 50, segments),
        HRV_STATISTIC_BBI_MEDIAN: lambda data: (
            float(np.median(data)) if data.size else np.nan
        ),
        HRV_STATISTIC_MIN_NN: lambda data: (
            float(np.min(data)) if data.size else np.nan
        ),
        HRV_STATISTIC_MEAN_NN: lambda data: mean_nn,
        HRV_STATISTIC_MAX_NN: lambda data: (
            float(np.max(data)) if data.size else np.nan
        ),
        HRV_STATISTIC_SDNN_RMSSD_RATIO: lambda data: (
            sdnn / rmssd
            if pd.notna(sdnn) and pd.notna(rmssd) and rmssd > 0
            else np.nan
        ),
        HRV_STATISTIC_CVNN: lambda data: (
            sdnn / mean_nn
            if pd.notna(sdnn) and pd.notna(mean_nn) and mean_nn > 0
            else np.nan
        ),
        HRV_STATISTIC_CVSD: lambda data: (
            rmssd / mean_nn
            if pd.notna(rmssd) and pd.notna(mean_nn) and mean_nn > 0
            else np.nan
        ),
        HRV_STATISTIC_IQRNN: lambda data: (
            float(np.percentile(data, 75) - np.percentile(data, 25))
            if data.size
            else np.nan
        ),
    }
    for name in set(statistics).intersection(operations):
        result[name] = operations[name](values)

    requested_frequency = set(statistics).intersection(
        HRV_FREQUENCY_DOMAIN_STATISTICS
    )
    if requested_frequency and allow_frequency:
        spectrum = _power_spectrum(values, timestamps)
        powers = {
            HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER: _band_power(
                spectrum, 0.0033, 0.04
            ),
            HRV_STATISTIC_LOW_FREQUENCY_POWER: _band_power(spectrum, 0.04, 0.15),
            HRV_STATISTIC_HIGH_FREQUENCY_POWER: _band_power(spectrum, 0.15, 0.40),
        }
        for name in requested_frequency.difference(
            {HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO}
        ):
            result[name] = powers[name]
        if HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO in requested_frequency:
            hf = powers[HRV_STATISTIC_HIGH_FREQUENCY_POWER]
            result[HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO] = (
                powers[HRV_STATISTIC_LOW_FREQUENCY_POWER] / hf
                if pd.notna(hf) and hf > 0
                else np.nan
            )
    return result


def _neurokit_window_statistics(
    values,
    statistics,
    segment_ids=None,
    timestamps=None,
    allow_frequency=True,
) -> dict:
    """Calculate NeuroKit features using the observed BBI time axis."""
    try:
        import neurokit2 as nk
    except ImportError as error:
        raise ImportError("backend='neurokit' requires neurokit2") from error

    values, _, timestamps = _valid_bbi_inputs(values, segment_ids, timestamps)
    result = {name: np.nan for name in statistics}
    if values.size <= 5:
        return result
    if timestamps is None:
        interval_times = np.cumsum(values) / 1000
    else:
        interval_times = (timestamps - timestamps[0]).total_seconds().to_numpy()
    intervals = {"RRI": values, "RRI_Time": interval_times}
    time_mapping = {
        HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES: "HRV_RMSSD",
        HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL: "HRV_SDNN",
        HRV_STATISTIC_PNN20: "HRV_pNN20",
        HRV_STATISTIC_PNN50: "HRV_pNN50",
        HRV_STATISTIC_BBI_MEDIAN: "HRV_MedianNN",
        HRV_STATISTIC_MIN_NN: "HRV_MinNN",
        HRV_STATISTIC_MEAN_NN: "HRV_MeanNN",
        HRV_STATISTIC_MAX_NN: "HRV_MaxNN",
        HRV_STATISTIC_SDNN_RMSSD_RATIO: "HRV_SDRMSSD",
        HRV_STATISTIC_CVNN: "HRV_CVNN",
        HRV_STATISTIC_CVSD: "HRV_CVSD",
        HRV_STATISTIC_IQRNN: "HRV_IQRNN",
        "hrv_sdann_1min": "HRV_SDANN1",
        "hrv_sdnni_1min": "HRV_SDNNI1",
        "hrv_sdann_2min": "HRV_SDANN2",
        "hrv_sdnni_2min": "HRV_SDNNI2",
        "hrv_sdann_5min": "HRV_SDANN5",
        "hrv_sdnni_5min": "HRV_SDNNI5",
        "hrv_sdsd": "HRV_SDSD",
        "hrv_mad_nn": "HRV_MadNN",
        "hrv_mcv_nn": "HRV_MCVNN",
        "hrv_prc20_nn": "HRV_Prc20NN",
        "hrv_prc80_nn": "HRV_Prc80NN",
        "hrv_hti": "HRV_HTI",
        "hrv_tinn": "HRV_TINN",
    }
    requested_time = set(statistics).intersection(time_mapping)
    if requested_time:
        frame = nk.hrv_time(intervals, show=False)
        for name in requested_time:
            result[name] = frame.iloc[0].get(time_mapping[name], np.nan)

    frequency_mapping = {
        HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER: "HRV_VLF",
        HRV_STATISTIC_LOW_FREQUENCY_POWER: "HRV_LF",
        HRV_STATISTIC_HIGH_FREQUENCY_POWER: "HRV_HF",
        HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO: "HRV_LFHF",
        "hrv_ulf_power": "HRV_ULF",
        "hrv_vhf_power": "HRV_VHF",
        "hrv_total_power": "HRV_TP",
        "hrv_lf_normalized": "HRV_LFn",
        "hrv_hf_normalized": "HRV_HFn",
        "hrv_log_hf": "HRV_LnHF",
    }
    requested_frequency = set(statistics).intersection(frequency_mapping)
    if (
        requested_frequency
        and allow_frequency
        and np.all(np.diff(interval_times) > 0)
    ):
        frame = nk.hrv_frequency(
            intervals,
            show=False,
            normalize=False,
            interpolation_rate=4,
        )
        for name in requested_frequency:
            result[name] = frame.iloc[0].get(frequency_mapping[name], np.nan)
    return result


def _window_statistics(
    values,
    statistics,
    backend: str,
    segment_ids=None,
    timestamps=None,
    allow_frequency=True,
) -> dict:
    if backend == "pywearable":
        return _pywearable_window_statistics(
            values, statistics, segment_ids, timestamps, allow_frequency
        )
    if backend == "neurokit":
        return _neurokit_window_statistics(
            values, statistics, segment_ids, timestamps, allow_frequency
        )
    raise ValueError("backend must be 'pywearable' or 'neurokit'")


def _statistics_list(statistic, backend: str = "pywearable") -> list[str]:
    available = (
        HRV_STATISTICS if backend == "pywearable" else NEUROKIT_HRV_STATISTICS
    )
    if statistic is None:
        return list(available)
    statistics = [statistic] if isinstance(statistic, str) else list(statistic)
    unknown = set(statistics).difference(available)
    if unknown:
        raise ValueError(
            f"statistic must be one among {list(available)}, "
            f"not {sorted(unknown)}"
        )
    return list(dict.fromkeys(statistics))


def _resolve_time_col(loader, time_reference: str, chain) -> str:
    """Select the available clock appropriate for the requested hierarchy."""
    available = getattr(loader, "hrv_time_columns", schema.BBI_TIME_COLUMNS)
    return hierarchy.resolve_time_col(available, time_reference, chain, "HRV")


def _normalize_window_step(window_step) -> pd.Timedelta:
    """Validate the displacement between primary five-minute windows."""
    duration = pd.Timedelta(HRV_PRIMARY_RESOLUTION)
    step = duration if window_step is None else pd.Timedelta(window_step)
    if step <= pd.Timedelta(0) or step > duration:
        raise ValueError(
            f"window_step must be greater than zero and no longer than {duration}"
        )
    return step


def _timestamp_continuity_segments(frame, time_col) -> np.ndarray:
    """Identify temporally contiguous BBI runs after loader-side filtering."""
    if frame.empty:
        return np.array([], dtype=int)
    continuity_time_col = (
        schema.UTC_TIME_COL
        if schema.UTC_TIME_COL in frame and frame[schema.UTC_TIME_COL].notna().all()
        else time_col
    )
    timestamps = frame[continuity_time_col]
    intervals = frame[schema.BBI_COL].to_numpy(dtype=float)
    elapsed = timestamps.diff().dt.total_seconds().to_numpy() * 1000
    previous = np.roll(intervals, 1)
    tolerance = np.maximum(250.0, 0.25 * np.maximum(intervals, previous))
    error = np.minimum(np.abs(elapsed - intervals), np.abs(elapsed - previous))
    contiguous = (elapsed > 0) & np.isfinite(error) & (error <= tolerance)
    contiguous[0] = True
    return np.cumsum(~contiguous)


def _interval_mask(timestamps: pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    """Return membership in the union of valid half-open intervals."""
    mask = np.zeros(len(timestamps), dtype=bool)
    if intervals.empty or timestamps.empty:
        return mask
    values = pd.DatetimeIndex(timestamps)
    for interval in intervals.itertuples():
        start = getattr(interval, "start_time")
        end = getattr(interval, "end_time")
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        left = values.searchsorted(start, side="left")
        right = values.searchsorted(end, side="left")
        mask[left:right] = True
    return mask


def _primary_windows(user_id, start, end, window_step, cycle_intervals=None):
    """Construct primary windows, keeping cycle windows fully contained."""
    duration = pd.Timedelta(HRV_PRIMARY_RESOLUTION)
    if cycle_intervals is None and window_step == duration:
        return hierarchy.fixed_windows(
            [user_id], HRV_PRIMARY_RESOLUTION, {user_id: (start, end)}
        ).xs(user_id)

    bounds = [(pd.Timestamp(start), pd.Timestamp(end))]
    if cycle_intervals is not None:
        valid = cycle_intervals[cycle_intervals["interval_valid"]]
        bounds = [
            (interval.start_time, interval.end_time)
            for interval in valid.itertuples()
        ]
    records = []
    for interval_start, interval_end in bounds:
        if cycle_intervals is not None and window_step < duration:
            first_start = interval_start.ceil(window_step)
            last_start = interval_end - duration
            starts = (
                pd.date_range(first_start, last_start, freq=window_step)
                if first_start <= last_start
                else pd.DatetimeIndex([])
            )
        else:
            starts = pd.date_range(
                interval_start.floor(window_step),
                interval_end,
                freq=window_step,
                inclusive="left",
            )
        for label in starts:
            if cycle_intervals is None:
                period_start = max(label, interval_start)
                period_end = min(label + duration, interval_end)
            else:
                period_start = label
                period_end = label + duration
            records.append(
                {
                    "window_start": label,
                    "start_time": period_start,
                    "end_time": period_end,
                }
            )
    if not records:
        return pd.DataFrame(columns=["start_time", "end_time"]).rename_axis(
            "window_start"
        )
    return (
        pd.DataFrame(records)
        .drop_duplicates("window_start")
        .set_index("window_start")
        .sort_index()
    )


def _asleep_stage_membership(
    loader,
    user_id,
    timestamps,
    start_date,
    end_date,
    sleep_stages_to_remove,
):
    """Mark retained sleep-stage samples and their contiguous blocks."""
    if not sleep_stages_to_remove:
        return np.ones(len(timestamps), dtype=bool), np.zeros(len(timestamps), dtype=int)
    if isinstance(sleep_stages_to_remove, str):
        sleep_stages_to_remove = [sleep_stages_to_remove]
    stages_to_remove = {str(value).lower() for value in sleep_stages_to_remove}
    unknown = stages_to_remove.difference(schema.SLEEP_STAGE_ALLOWED_VALUES)
    if unknown:
        raise ValueError(
            "sleep_stages_to_remove contains unsupported values: "
            f"{sorted(unknown)}"
        )
    stages = schema.prepare_sleep_stages(
        loader.load_sleep_stage(user_id, start_date, end_date)
    )
    if stages.empty:
        return np.ones(len(timestamps), dtype=bool), np.zeros(len(timestamps), dtype=int)
    if "local_time_uncertain" in stages.columns:
        stages = stages[~stages["local_time_uncertain"].fillna(False)]
    stages = stages[
        ~stages[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(stages_to_remove)
    ]
    intervals = stages.assign(
        start_time=stages[schema.LOCAL_TIME_COL],
        end_time=stages[schema.LOCAL_TIME_COL]
        + pd.to_timedelta(stages[schema.DURATION_IN_MS_COL], unit="ms"),
    )[["start_time", "end_time"]].sort_values("start_time")

    blocks = []
    for interval in intervals.itertuples(index=False):
        if pd.isna(interval.start_time) or pd.isna(interval.end_time):
            continue
        if interval.end_time <= interval.start_time:
            continue
        if not blocks or interval.start_time > blocks[-1][1]:
            blocks.append([interval.start_time, interval.end_time])
        else:
            blocks[-1][1] = max(blocks[-1][1], interval.end_time)

    values = pd.DatetimeIndex(timestamps)
    mask = np.zeros(len(values), dtype=bool)
    segment_ids = np.full(len(values), -1, dtype=int)
    for segment_id, (block_start, block_end) in enumerate(blocks):
        left = values.searchsorted(block_start, side="left")
        right = values.searchsorted(block_end, side="left")
        mask[left:right] = True
        segment_ids[left:right] = segment_id
    return mask, segment_ids


def _raw_bbi_frame_trajectory(
    bbi,
    user_id,
    statistics,
    start_date,
    end_date,
    time_col,
    backend,
    min_coverage,
    return_coverage,
    window_step,
    cycle_intervals=None,
    sleep_stage_loader=None,
    sleep_stages_to_remove=DEFAULT_SLEEP_STAGES_TO_REMOVE,
):
    """Build primary HRV rows from one participant's canonical BBI frame."""
    bbi = schema.prepare_bbi(bbi)
    bbi = bbi[bbi[time_col].notna()].sort_values(time_col, kind="stable")
    bbi[_CONTINUITY_SEGMENT_COL] = _timestamp_continuity_segments(bbi, time_col)
    bbi[_SLEEP_STAGE_SEGMENT_COL] = 0
    bounds = hierarchy.observed_bounds(
        bbi, time_col, start_date, end_date, HRV_PRIMARY_RESOLUTION
    )
    if bounds is None:
        return hierarchy.empty_trajectory(statistics, include_coverage=return_coverage)
    start, end = bounds
    bbi = bbi[bbi[time_col].ge(start) & bbi[time_col].lt(end)].reset_index(drop=True)

    if cycle_intervals is not None and not bbi.empty:
        valid = cycle_intervals[cycle_intervals["interval_valid"]]
        bbi = bbi[_interval_mask(bbi[time_col], valid)].reset_index(drop=True)
        if sleep_stage_loader is not None and not bbi.empty:
            asleep, segments = _asleep_stage_membership(
                sleep_stage_loader,
                user_id,
                bbi[time_col],
                start,
                end,
                sleep_stages_to_remove,
            )
            bbi = bbi[asleep].copy()
            if not bbi.empty:
                bbi[_SLEEP_STAGE_SEGMENT_COL] = segments[asleep]
                # A new calculation segment starts when either the source BBI
                # continuity or the retained sleep-stage block changes.
                combined = list(
                    zip(
                        bbi[_CONTINUITY_SEGMENT_COL].to_numpy(),
                        segments[asleep],
                    )
                )
                bbi[_CONTINUITY_SEGMENT_COL] = pd.factorize(
                    pd.Series(combined, dtype="object")
                )[0]
            bbi = bbi.reset_index(drop=True)

    windows = _primary_windows(user_id, start, end, window_step, cycle_intervals)
    timestamps = pd.DatetimeIndex(bbi[time_col])
    records = []
    for label, window in windows.iterrows():
        # Floor-based grouping works only for disjoint windows. Binary searches
        # let the same sorted BBI rows participate in every overlapping window.
        left = timestamps.searchsorted(window["start_time"], side="left")
        right = timestamps.searchsorted(window["end_time"], side="left")
        rows = bbi.iloc[left:right]
        duration_ms = (window["end_time"] - window["start_time"]).total_seconds() * 1000
        coverage = (
            min(1.0, rows[schema.BBI_COL].sum() / duration_ms)
            if duration_ms > 0
            else np.nan
        )
        accepted = min_coverage is None or coverage >= min_coverage
        values = (
            _window_statistics(
                rows[schema.BBI_COL],
                statistics,
                backend,
                rows[_CONTINUITY_SEGMENT_COL],
                rows[
                    schema.UTC_TIME_COL
                    if schema.UTC_TIME_COL in rows
                    and rows[schema.UTC_TIME_COL].notna().all()
                    else time_col
                ],
                rows[_SLEEP_STAGE_SEGMENT_COL].nunique(dropna=False) <= 1,
            )
            if accepted
            else {name: np.nan for name in statistics}
        )
        records.append(
            {
                schema.USER_COL: user_id,
                "window_start": label,
                "period_start": window["start_time"],
                "period_end": window["end_time"],
                **hierarchy.fixed_window_time_metadata(rows, label, time_col),
                **values,
                **(
                    {f"{name}__coverage": coverage for name in statistics}
                    if return_coverage
                    else {}
                ),
            }
        )
    return (
        pd.DataFrame(records)
        .set_index([schema.USER_COL, "window_start"])
        .sort_index()
    )


def _raw_bbi_primary_trajectory(
    loader,
    user_id,
    statistics,
    start_date,
    end_date,
    time_col,
    loader_kwargs,
    backend,
    min_coverage,
    return_coverage,
    window_step,
    cycle_intervals=None,
    sleep_only=False,
    sleep_stages_to_remove=DEFAULT_SLEEP_STAGES_TO_REMOVE,
):
    padding = pd.Timedelta(days=2)
    load_start = None if start_date is None else pd.Timestamp(start_date) - padding
    load_end = None if end_date is None else pd.Timestamp(end_date) + padding
    bbi = loader.load_bbi(user_id, load_start, load_end, **loader_kwargs)
    return _raw_bbi_frame_trajectory(
        bbi,
        user_id,
        statistics,
        start_date,
        end_date,
        time_col,
        backend,
        min_coverage,
        return_coverage,
        window_step,
        cycle_intervals=cycle_intervals,
        sleep_stage_loader=loader if sleep_only else None,
        sleep_stages_to_remove=sleep_stages_to_remove,
    )


def _prepare_provider_samples(frame) -> pd.DataFrame:
    """Normalize provider HRV samples to the shared column and dtype contract."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("provider HRV samples must be a pandas DataFrame")
    frame = frame.copy().rename(
        columns={
            "RMSSD": schema.HRV_RMSSD_COL,
            "LF": schema.HRV_LF_POWER_COL,
            "HF": schema.HRV_HF_POWER_COL,
            "coverage": schema.HRV_COVERAGE_COL,
            "timezone_offset": schema.TIMEZONE_COL,
        }
    )
    for column in schema.BBI_TIME_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(
            frame[column], errors="coerce", utc=True
        ).dt.tz_localize(None)
    if schema.TIMEZONE_COL not in frame:
        frame[schema.TIMEZONE_COL] = (
            frame[schema.LOCAL_TIME_COL] - frame[schema.UTC_TIME_COL]
        )
    elif pd.api.types.is_datetime64_any_dtype(frame[schema.TIMEZONE_COL]):
        if frame[schema.TIMEZONE_COL].notna().any():
            raise ValueError("provider HRV timezone values must be UTC offsets")
        frame[schema.TIMEZONE_COL] = pd.Series(
            pd.NaT, index=frame.index, dtype="timedelta64[ns]"
        )
    else:
        frame[schema.TIMEZONE_COL] = pd.to_timedelta(
            frame[schema.TIMEZONE_COL], errors="coerce"
        )
    derived_offset = frame[schema.LOCAL_TIME_COL] - frame[schema.UTC_TIME_COL]
    frame[schema.TIMEZONE_COL] = frame[schema.TIMEZONE_COL].fillna(derived_offset)
    for column in {
        schema.HRV_RMSSD_COL,
        schema.HRV_VLF_POWER_COL,
        schema.HRV_LF_POWER_COL,
        schema.HRV_HF_POWER_COL,
        schema.HRV_COVERAGE_COL,
    }:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _provider_primary_trajectory(
    loader,
    user_id,
    statistics,
    start_date,
    end_date,
    time_col,
    loader_kwargs,
    min_coverage,
    return_coverage,
    unknown_coverage,
    window_step,
    cycle_intervals=None,
):
    if window_step != pd.Timedelta(HRV_PRIMARY_RESOLUTION):
        raise ValueError(
            "window_step cannot alter provider-computed HRV samples"
        )
    padding = pd.Timedelta(days=2)
    load_start = None if start_date is None else pd.Timestamp(start_date) - padding
    load_end = None if end_date is None else pd.Timestamp(end_date) + padding
    samples = _prepare_provider_samples(
        loader.load_hrv_samples(user_id, load_start, load_end, **loader_kwargs)
    )
    samples = samples[samples[time_col].notna()].sort_values(time_col, kind="stable")
    bounds = hierarchy.observed_bounds(
        samples, time_col, start_date, end_date, HRV_PRIMARY_RESOLUTION
    )
    if bounds is None:
        return hierarchy.empty_trajectory(statistics, include_coverage=return_coverage)
    start, end = bounds
    samples = samples[
        samples[time_col].ge(start) & samples[time_col].lt(end)
    ].copy()
    if cycle_intervals is not None and not samples.empty:
        valid = cycle_intervals[cycle_intervals["interval_valid"]]
        samples = samples[_interval_mask(samples[time_col], valid)].copy()
    windows = _primary_windows(user_id, start, end, window_step, cycle_intervals)
    samples["window_start"] = samples[time_col].dt.floor(HRV_PRIMARY_RESOLUTION)
    groups = {
        label: rows for label, rows in samples.groupby("window_start", sort=False)
    }
    records = []
    for label, window in windows.iterrows():
        rows = groups.get(label, samples.iloc[0:0])
        provider_coverage = (
            rows[schema.HRV_COVERAGE_COL].mean()
            if schema.HRV_COVERAGE_COL in rows
            and rows[schema.HRV_COVERAGE_COL].notna().any()
            else np.nan
        )
        if pd.notna(provider_coverage):
            provider_coverage = float(np.clip(provider_coverage, 0, 1))
        elif min_coverage is not None and not rows.empty:
            unknown_coverage.add((user_id, getattr(loader, "hrv_source", "provider")))
        accepted = min_coverage is None or (
            pd.notna(provider_coverage) and provider_coverage >= min_coverage
        )
        values = {name: np.nan for name in statistics}
        if accepted and not rows.empty:
            for name, column in _PROVIDER_COLUMNS.items():
                if name in statistics and column in rows:
                    values[name] = rows[column].mean()
            if HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO in statistics:
                lf = values.get(HRV_STATISTIC_LOW_FREQUENCY_POWER)
                hf = values.get(HRV_STATISTIC_HIGH_FREQUENCY_POWER)
                if pd.isna(lf) and schema.HRV_LF_POWER_COL in rows:
                    lf = rows[schema.HRV_LF_POWER_COL].mean()
                if pd.isna(hf) and schema.HRV_HF_POWER_COL in rows:
                    hf = rows[schema.HRV_HF_POWER_COL].mean()
                values[HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO] = (
                    lf / hf if pd.notna(hf) and hf > 0 else np.nan
                )
        records.append(
            {
                schema.USER_COL: user_id,
                "window_start": label,
                "period_start": window["start_time"],
                "period_end": window["end_time"],
                **hierarchy.fixed_window_time_metadata(rows, label, time_col),
                **values,
                **(
                    {f"{name}__coverage": provider_coverage for name in statistics}
                    if return_coverage
                    else {}
                ),
            }
        )
    return (
        pd.DataFrame(records)
        .set_index([schema.USER_COL, "window_start"])
        .sort_index()
    )


def _primary_trajectory(
    loader,
    users,
    statistics,
    start_date,
    end_date,
    time_col,
    loader_kwargs,
    backend,
    min_coverage,
    return_coverage,
    window_step,
    cycle_parent=None,
    sleep_only=False,
    sleep_stages_to_remove=DEFAULT_SLEEP_STAGES_TO_REMOVE,
):
    frames = []
    unknown_coverage = set()
    source = getattr(loader, "hrv_source", "bbi")
    interval_users = (
        set(cycle_parent.index.get_level_values(0))
        if cycle_parent is not None and not cycle_parent.empty
        else set()
    )
    for user in users:
        user_start, user_end = start_date, end_date
        user_intervals = None
        if cycle_parent is not None:
            if user not in interval_users:
                continue
            user_intervals = cycle_parent.xs(user)
            valid = user_intervals[user_intervals["interval_valid"]]
            if valid.empty:
                continue
            user_start = valid["start_time"].min()
            user_end = valid["end_time"].max()
        if source == "provider_samples":
            frame = _provider_primary_trajectory(
                loader,
                user,
                statistics,
                user_start,
                user_end,
                time_col,
                loader_kwargs,
                min_coverage,
                return_coverage,
                unknown_coverage,
                window_step,
                cycle_intervals=user_intervals,
            )
        elif source == "bbi":
            frame = _raw_bbi_primary_trajectory(
                loader,
                user,
                statistics,
                user_start,
                user_end,
                time_col,
                loader_kwargs,
                backend,
                min_coverage,
                return_coverage,
                window_step,
                cycle_intervals=user_intervals,
                sleep_only=sleep_only,
                sleep_stages_to_remove=sleep_stages_to_remove,
            )
        else:
            raise ValueError(
                "loader.hrv_source must be 'bbi' or 'provider_samples'"
            )
        if not frame.empty:
            frames.append(frame)
    if min_coverage is not None and unknown_coverage:
        warnings.warn(
            "Primary HRV coverage could not be determined for "
            f"{len(unknown_coverage)} user/source stream(s); features requiring "
            "a primary coverage threshold were excluded.",
            UserWarning,
            stacklevel=3,
        )
    return (
        pd.concat(frames).sort_index()
        if frames
        else hierarchy.empty_trajectory(statistics, include_coverage=return_coverage)
    )


def compute_hrv_statistics(
    bbi: pd.DataFrame,
    statistics: Union[list, tuple, str, None] = None,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    coverage: float | None = 0.7,
    window_size: Union[str, pd.Timedelta] = HRV_PRIMARY_RESOLUTION,
    window_step: Union[str, pd.Timedelta, None] = None,
    time_col: str = schema.UTC_TIME_COL,
) -> pd.DataFrame:
    """Calculate HRV windows from an already loaded BBI table."""
    statistics = _statistics_list(statistics, backend)
    resolution = hierarchy.canonical_resolution(window_size)
    if resolution != HRV_PRIMARY_RESOLUTION:
        raise ValueError("window_size must currently be five minutes")
    step = _normalize_window_step(window_step)
    frame = schema.prepare_bbi(bbi)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                column
                for name in statistics
                for column in (name, f"{name}__coverage")
            ]
        )
    trajectory = _raw_bbi_frame_trajectory(
        frame,
        "user",
        statistics,
        frame[time_col].min(),
        frame[time_col].max().floor(resolution) + pd.Timedelta(resolution),
        time_col,
        backend,
        coverage,
        True,
        step,
    )
    return trajectory.droplevel(schema.USER_COL)


def _expanded_chain(resolution):
    """Normalize an HRV request and prepend its five-minute primary level."""
    if isinstance(resolution, (str, pd.Timedelta, datetime.timedelta)):
        values = [resolution]
        hierarchical = False
    else:
        values = list(resolution)
        if not values:
            raise ValueError("resolution must be a string or a non-empty sequence")
        hierarchical = len(values) > 1
    chain = [hierarchy.canonical_resolution(value) for value in values]
    chain = [cycle.CYCLE_NIGHT if value == "nightly" else value for value in chain]
    hierarchy.normalize_resolution_chain(chain)
    if chain[0] != HRV_PRIMARY_RESOLUTION:
        if (
            chain[0] not in cycle.CYCLE_TYPES
            and chain[0] not in hierarchy.CALENDAR_RESOLUTIONS
        ):
            raise ValueError("HRV extraction must start with five-minute windows")
        chain = [HRV_PRIMARY_RESOLUTION, *chain]
        hierarchical = True
        hierarchy.normalize_resolution_chain(chain)
    return chain, hierarchical


def _hierarchical_hrv_statistics(
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
    backend,
    window_step,
    sleep_stages_to_remove,
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
    time_col = _resolve_time_col(loader, time_reference, chain)
    start_date = schema.normalize_time_bound(start_date, time_col)
    end_date = schema.normalize_time_bound(end_date, time_col)
    cycle_target = next(
        (value for value in chain[1:] if value in cycle.CYCLE_TYPES), None
    )
    cycle_parent = None
    if cycle_target is not None:
        cycle_parent = cycle.get_intervals(
            loader,
            cycle_type=cycle_target,
            user_id=users,
            start_date=start_date,
            end_date=end_date,
            **cycle_kwargs,
        )
    trajectory = _primary_trajectory(
        loader,
        users,
        statistics,
        start_date,
        end_date,
        time_col,
        loader_kwargs,
        backend,
        coverage_thresholds[0],
        True,
        window_step,
        cycle_parent=cycle_parent,
        sleep_only=cycle_target == cycle.CYCLE_NIGHT
        and getattr(loader, "hrv_source", "bbi") == "bbi",
        sleep_stages_to_remove=sleep_stages_to_remove,
    )
    trajectory = trajectory.rename(
        columns={name: f"{name}__{HRV_PRIMARY_RESOLUTION}" for name in statistics}
    )
    for level_index, target in enumerate(chain[1:]):
        lower = chain[level_index]
        parent, membership = hierarchy.parent_windows_from_trajectory(
            loader,
            users,
            trajectory,
            target,
            lower,
            start_date,
            end_date,
            cycle_kwargs,
            cycle_parent if target == cycle_target else None,
            prefer_explicit_bounds=True,
            cycle_label_bounds=False,
        )
        if parent is None:
            return trajectory.iloc[0:0].rename_axis(
                index={"window_start": schema.CALENDAR_DATE_COL}
            )
        trajectory, _ = hierarchy.aggregate_level(
            trajectory,
            parent,
            target_resolution=target,
            aggregation=aggregation_levels[level_index],
            membership=membership,
            min_coverage=coverage_thresholds[level_index + 1],
            min_valid_children=valid_child_thresholds[level_index + 1],
            time_col=time_col,
        )
    index_name = (
        schema.CALENDAR_DATE_COL
        if chain[-1] in cycle.CYCLE_TYPES
        or chain[-1] in hierarchy.CALENDAR_RESOLUTIONS
        else time_col
    )
    trajectory = trajectory.rename_axis(index={"window_start": index_name})
    if hierarchy.fixed_duration(chain[-1]) is not None:
        return hierarchy.format_fixed_time_output(trajectory, index_name)
    return trajectory


def get_hrv_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    resolution: Union[str, pd.Timedelta, list, tuple] = "day",
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    aggregation=None,
    min_coverage=None,
    cycle_kwargs: dict | None = None,
    min_valid_children=None,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    window_step: Union[str, pd.Timedelta, None] = None,
    sleep_stages_to_remove: Union[str, list, tuple, None] = (
        DEFAULT_SLEEP_STAGES_TO_REMOVE
    ),
) -> pd.DataFrame:
    """Return HRV features at five minutes or along a resolution hierarchy.

    Raw-BBI loaders calculate five-minute features, with optional overlap.
    Provider loaders use their native five-minute estimates and leave
    unsupported statistics missing. A scalar calendar or cycle resolution is
    shorthand for a five-minute-to-parent hierarchy. Source selection is
    fixed by the loader; missing provider values are never filled from a
    different HRV source.

    ``min_coverage`` supplies one threshold per level. At the primary raw-BBI
    level, coverage is represented BBI duration divided by window duration.
    Provider coverage is used when available; an explicit primary threshold
    rejects provider samples whose coverage is unavailable. Higher-level
    coverage is the fraction of valid child features.

    Parameters
    ----------
    loader
        Loader exposing raw beat-to-beat intervals or provider-computed HRV
        samples.
    statistic
        One HRV feature name, a sequence of names, or ``None`` for every
        feature registered for the selected backend.
    user_id
        One participant, several participants, or ``"all"``/``None``.
    start_date, end_date
        Half-open analysis bounds. For cycle resolutions, the bounds select
        cycle calendar labels and selected cycles are not clipped.
    resolution
        A five-minute, calendar, or sleep-cycle resolution, or an increasing
        hierarchy. Calendar and cycle shorthand requests automatically begin
        from the native five-minute HRV grain.
    loader_kwargs
        Extra keyword arguments passed to the loader's HRV method.
    time_reference
        ``"auto"``, ``"local"``, or ``"utc"`` for fixed windows. Calendar
        and sleep-cycle parents require participant-local time.
    aggregation
        Operation or branching operations for each parent level. It is valid
        only when the request forms a hierarchy.
    min_coverage
        Optional minimum coverage at each requested level. A scalar applies
        to every level; a sequence supplies one value per resolution.
    cycle_kwargs
        Extra arguments passed to :func:`pywearable.cycle.get_intervals`.
    min_valid_children
        Optional minimum number of valid child windows at each parent level.
        In sequence form, the primary entry must be ``None``.
    backend
        ``"pywearable"`` for the built-in raw-BBI calculation or
        ``"neurokit"`` for the optional NeuroKit backend. Both backends retain
        the observed BBI time axis when processing irregular intervals.
    window_step
        Displacement between primary five-minute windows. ``None`` preserves
        non-overlapping windows; a shorter duration creates overlapping raw-BBI
        windows. Sleep-cycle overlap retains only fully contained windows.
        Provider-computed HRV cannot be re-windowed.
    sleep_stages_to_remove
        Sleep stages excluded from sleep-cycle extraction. By default,
        ``awake`` and ``unmeasurable`` are removed. ``None`` or an empty
        sequence disables stage filtering.

    Returns
    -------
    pandas.DataFrame
        Fixed-duration output keeps the selected clock in the index and adds
        the alternate clock and ``timezone`` UTC offset as columns.
        A ``(user_id, time)`` trajectory containing each requested feature and
        its ``__coverage`` column. Hierarchical names preserve the complete
        calculation path.

    Examples
    --------
    Five-minute RMSSD::

        get_hrv_statistics(loader, statistic="hrv_rmssd", resolution="5min")

    Nightly RMSSD followed by monthly mean and p90::

        get_hrv_statistics(
            loader,
            statistic="hrv_rmssd",
            resolution=["5min", "cycle_night", "monthly"],
            aggregation=[["mean"], ["mean", "p90"]],
            min_coverage=[0.7, 0.5, 0.8],
            min_valid_children=[None, 3, 10],
        )
    """
    chain, hierarchical = _expanded_chain(resolution)
    if not hierarchical and aggregation is not None:
        raise ValueError("aggregation requires a resolution hierarchy")
    if not hierarchical and min_valid_children is not None:
        raise ValueError("min_valid_children requires a resolution hierarchy")
    if backend not in {"pywearable", "neurokit"}:
        raise ValueError("backend must be 'pywearable' or 'neurokit'")
    step = _normalize_window_step(window_step)
    if (
        getattr(loader, "hrv_source", "bbi") == "provider_samples"
        and step != pd.Timedelta(HRV_PRIMARY_RESOLUTION)
    ):
        raise ValueError("window_step cannot alter provider-computed HRV samples")
    loader_kwargs = {} if loader_kwargs is None else dict(loader_kwargs)
    cycle_kwargs = {} if cycle_kwargs is None else dict(cycle_kwargs)
    statistics = _statistics_list(statistic, backend)
    users = loader.resolve_user_ids(user_id)
    if (
        min_coverage is None
        and cycle.CYCLE_NIGHT in chain
        and getattr(loader, "hrv_source", "bbi") == "bbi"
    ):
        min_coverage = [0.7, *([None] * (len(chain) - 1))]

    if not hierarchical:
        threshold = hierarchy.normalize_coverage_thresholds(min_coverage, 1)[0]
        time_col = _resolve_time_col(loader, time_reference, chain)
        result = _primary_trajectory(
            loader,
            users,
            statistics,
            start_date,
            end_date,
            time_col,
            loader_kwargs,
            backend,
            threshold,
            True,
            step,
        )
        columns = [
            column
            for name in statistics
            for column in (name, f"{name}__coverage")
        ]
        result = result[
            columns
            + [
                column
                for column in hierarchy.TIME_METADATA_COLUMNS
                if column in result
            ]
        ].rename_axis(index={"window_start": time_col})
        return hierarchy.format_fixed_time_output(result, time_col)

    return _hierarchical_hrv_statistics(
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
        backend,
        step,
        sleep_stages_to_remove,
    )


def get_hrv_statistic(
    loader: BaseLoader,
    statistic: str,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    resolution: Union[str, pd.Timedelta, list, tuple] = "day",
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    aggregation=None,
    min_coverage=None,
    cycle_kwargs: dict | None = None,
    min_valid_children=None,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    window_step: Union[str, pd.Timedelta, None] = None,
    sleep_stages_to_remove: Union[str, list, tuple, None] = (
        DEFAULT_SLEEP_STAGES_TO_REMOVE
    ),
):
    """Return one HRV statistic; thin wrapper around :func:`get_hrv_statistics`."""
    return get_hrv_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
        aggregation=aggregation,
        min_coverage=min_coverage,
        cycle_kwargs=cycle_kwargs,
        min_valid_children=min_valid_children,
        backend=backend,
        window_step=window_step,
        sleep_stages_to_remove=sleep_stages_to_remove,
    )


def get_five_minute_hrv_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    min_coverage=None,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    window_step: Union[str, pd.Timedelta, None] = None,
):
    """Return five-minute HRV features, optionally with a shorter step."""
    return get_hrv_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution=HRV_PRIMARY_RESOLUTION,
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
        min_coverage=min_coverage,
        backend=backend,
        window_step=window_step,
    )


def get_night_hrv_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    aggregation="mean",
    min_coverage=None,
    cycle_kwargs: dict | None = None,
    min_valid_children=None,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    window_step: Union[str, pd.Timedelta, None] = None,
    sleep_stages_to_remove: Union[str, list, tuple, None] = (
        DEFAULT_SLEEP_STAGES_TO_REMOVE
    ),
):
    """Return aggregated five-minute HRV features for main-sleep cycles."""
    if min_coverage is None:
        primary = 0.7 if getattr(loader, "hrv_source", "bbi") == "bbi" else None
        min_coverage = [primary, None]
    return get_hrv_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution=[HRV_PRIMARY_RESOLUTION, cycle.CYCLE_NIGHT],
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
        aggregation=aggregation,
        min_coverage=min_coverage,
        cycle_kwargs=cycle_kwargs,
        min_valid_children=min_valid_children,
        backend=backend,
        window_step=window_step,
        sleep_stages_to_remove=sleep_stages_to_remove,
    )


def get_daily_hrv_statistics(
    loader: BaseLoader,
    statistic: Union[str, list, tuple, None] = None,
    user_id: Union[str, list, None] = "all",
    start_date=None,
    end_date=None,
    loader_kwargs: dict | None = None,
    time_reference: str = "auto",
    aggregation=None,
    min_coverage=None,
    min_valid_children=None,
    backend: Literal["pywearable", "neurokit"] = "pywearable",
    window_step: Union[str, pd.Timedelta, None] = None,
):
    """Return HRV features aggregated over participant-local calendar days."""
    return get_hrv_statistics(
        loader=loader,
        statistic=statistic,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        resolution="daily",
        loader_kwargs=loader_kwargs,
        time_reference=time_reference,
        aggregation=aggregation,
        min_coverage=min_coverage,
        min_valid_children=min_valid_children,
        backend=backend,
        window_step=window_step,
    )


def _night_statistic_wrapper(loader, statistic, coverage=None, **kwargs):
    """Implement the named nightly means through the multi-statistic API."""
    aggregation = kwargs.pop("aggregation", "mean")
    if aggregation != "mean":
        raise ValueError(
            "named night HRV wrappers only support aggregation='mean'; "
            "use get_hrv_statistics for other aggregations"
        )
    if coverage is not None:
        kwargs.setdefault("min_coverage", [coverage, None])
    result = get_night_hrv_statistics(
        loader=loader,
        statistic=statistic,
        aggregation="mean",
        **kwargs,
    )
    column = f"{statistic}__{HRV_PRIMARY_RESOLUTION}__{cycle.CYCLE_NIGHT}_mean"
    return result[column].rename(statistic)


def get_night_hrv_rmssd(loader, coverage: float | None = None, **kwargs):
    """Return mean five-minute RMSSD for each main-sleep cycle."""
    return _night_statistic_wrapper(
        loader, HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES, coverage, **kwargs
    )


def get_night_hrv_sdnn(loader, coverage: float | None = None, **kwargs):
    """Return mean five-minute SDNN for each main-sleep cycle."""
    return _night_statistic_wrapper(
        loader, HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL, coverage, **kwargs
    )


def get_night_hrv_vlf_power(loader, coverage: float | None = None, **kwargs):
    """Return mean five-minute VLF power for each main-sleep cycle."""
    return _night_statistic_wrapper(
        loader, HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER, coverage, **kwargs
    )


def get_night_hrv_lf_power(loader, coverage: float | None = None, **kwargs):
    """Return mean five-minute LF power for each main-sleep cycle."""
    return _night_statistic_wrapper(
        loader, HRV_STATISTIC_LOW_FREQUENCY_POWER, coverage, **kwargs
    )


def get_night_hrv_hf_power(loader, coverage: float | None = None, **kwargs):
    """Return mean five-minute HF power for each main-sleep cycle."""
    return _night_statistic_wrapper(
        loader, HRV_STATISTIC_HIGH_FREQUENCY_POWER, coverage, **kwargs
    )


def get_night_lfhf(loader, coverage: float | None = None, **kwargs):
    """Return mean five-minute LF/HF for each main-sleep cycle."""
    return _night_statistic_wrapper(
        loader, HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO, coverage, **kwargs
    )


__all__ = [
    "DEFAULT_SLEEP_STAGES_TO_REMOVE",
    "HRV_FREQUENCY_DOMAIN_STATISTICS",
    "HRV_STATISTIC_BBI_MEDIAN",
    "HRV_STATISTIC_CVNN",
    "HRV_STATISTIC_CVSD",
    "HRV_STATISTIC_HIGH_FREQUENCY_POWER",
    "HRV_STATISTIC_IQRNN",
    "HRV_STATISTIC_LOW_FREQUENCY_POWER",
    "HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO",
    "HRV_STATISTIC_MAX_NN",
    "HRV_STATISTIC_MEAN_NN",
    "HRV_STATISTIC_MIN_NN",
    "HRV_STATISTIC_PNN20",
    "HRV_STATISTIC_PNN50",
    "HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES",
    "HRV_STATISTIC_SDNN_RMSSD_RATIO",
    "HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL",
    "HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER",
    "HRV_STATISTICS",
    "HRV_TIME_DOMAIN_STATISTICS",
    "NEUROKIT_FREQUENCY_DOMAIN_STATISTICS",
    "NEUROKIT_HRV_STATISTICS",
    "NEUROKIT_TIME_DOMAIN_STATISTICS",
    "compute_hrv_statistics",
    "compute_pnn20",
    "compute_pnn50",
    "compute_rmssd",
    "compute_sdnn",
    "get_daily_hrv_statistics",
    "get_five_minute_hrv_statistics",
    "get_hrv_statistic",
    "get_hrv_statistics",
    "get_night_hrv_hf_power",
    "get_night_hrv_lf_power",
    "get_night_hrv_rmssd",
    "get_night_hrv_sdnn",
    "get_night_hrv_statistics",
    "get_night_hrv_vlf_power",
    "get_night_lfhf",
]
