"""Heart-rate-variability features from BBI or provider estimates."""

from __future__ import annotations

import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, signal

from . import cycle, hierarchy, schema
from .loader.base import BaseLoader
from .utils.legacy import get_user_ids


HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES = "hrv_rmssd"
HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL = "hrv_sdnn"
HRV_STATISTIC_PNN20 = "hrv_pnn20"
HRV_STATISTIC_PNN50 = "hrv_pnn50"
HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER = "hrv_vlf_power"
HRV_STATISTIC_LOW_FREQUENCY_POWER = "hrv_lf_power"
HRV_STATISTIC_HIGH_FREQUENCY_POWER = "hrv_hf_power"
HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO = "hrv_lfhf_ratio"
HRV_STATISTIC_BBI_MEDIAN = "hrv_bbi_median"

HRV_TIME_DOMAIN_STATISTICS = (
    HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES,
    HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL,
    HRV_STATISTIC_PNN20,
    HRV_STATISTIC_PNN50,
    HRV_STATISTIC_BBI_MEDIAN,
)
HRV_FREQUENCY_DOMAIN_STATISTICS = (
    HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER,
    HRV_STATISTIC_LOW_FREQUENCY_POWER,
    HRV_STATISTIC_HIGH_FREQUENCY_POWER,
    HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO,
)
HRV_STATISTICS = HRV_TIME_DOMAIN_STATISTICS + HRV_FREQUENCY_DOMAIN_STATISTICS

HRV_PRIMARY_RESOLUTION = "5min"
_PROVIDER_COLUMNS = {
    HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES: schema.HRV_RMSSD_COL,
    HRV_STATISTIC_LOW_FREQUENCY_POWER: schema.HRV_LF_POWER_COL,
    HRV_STATISTIC_HIGH_FREQUENCY_POWER: schema.HRV_HF_POWER_COL,
}


def _valid_bbi_values_and_segments(values, segment_ids=None):
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if segment_ids is None:
        return values[valid], None
    segments = np.asarray(segment_ids)
    if segments.size != values.size:
        raise ValueError("segment_ids must contain one value per BBI interval")
    return values[valid], segments[valid]


def _valid_bbi_values(values) -> np.ndarray:
    return _valid_bbi_values_and_segments(values)[0]


def _successive_differences(values, segment_ids=None):
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
    return np.sum(np.abs(differences) > threshold) / values.size * 100


def compute_rmssd(values):
    """Return RMSSD in milliseconds, or missing for fewer than six intervals."""
    return _compute_rmssd(values)


def compute_sdnn(values):
    """Return sample SDNN in milliseconds, or missing for insufficient data."""
    values = _valid_bbi_values(values)
    return np.std(values, ddof=1) if values.size > 5 else np.nan


def compute_pnn20(values):
    """Return the percentage of supplied intervals followed by a >20 ms change."""
    return _compute_pnn(values, 20)


def compute_pnn50(values):
    """Return the percentage of supplied intervals followed by a >50 ms change."""
    return _compute_pnn(values, 50)


def _power_spectrum(values) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate irregular BBI at 4 Hz and calculate one Welch spectrum."""
    values = _valid_bbi_values(values)
    if values.size <= 5:
        return None
    times = np.cumsum(values) / 1000
    times -= times[0]
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


def _pywearable_window_statistics(values, statistics, segment_ids=None) -> dict:
    values, segments = _valid_bbi_values_and_segments(values, segment_ids)
    has_internal_gap = segments is not None and np.unique(segments).size > 1
    result = {name: np.nan for name in statistics}
    operations = {
        HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES: lambda data: (
            _compute_rmssd(data, segments)
        ),
        HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL: compute_sdnn,
        HRV_STATISTIC_PNN20: lambda data: _compute_pnn(data, 20, segments),
        HRV_STATISTIC_PNN50: lambda data: _compute_pnn(data, 50, segments),
        HRV_STATISTIC_BBI_MEDIAN: lambda data: (
            float(np.median(data)) if data.size else np.nan
        ),
    }
    for name in set(statistics).intersection(operations):
        result[name] = operations[name](values)

    requested_frequency = set(statistics).intersection(
        HRV_FREQUENCY_DOMAIN_STATISTICS
    )
    if requested_frequency and not has_internal_gap:
        spectrum = _power_spectrum(values)
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


def _neurokit_window_statistics(values, statistics, segment_ids=None) -> dict:
    """Calculate requested NeuroKit features with one call per feature domain."""
    try:
        import neurokit2 as nk
    except ImportError as error:
        raise ImportError("backend='neurokit' requires neurokit2") from error

    values, segments = _valid_bbi_values_and_segments(values, segment_ids)
    has_internal_gap = segments is not None and np.unique(segments).size > 1
    result = {name: np.nan for name in statistics}
    if values.size <= 5:
        return result
    intervals = {"RRI": values, "RRI_Time": np.cumsum(values) / 1000}
    time_mapping = {
        HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES: "HRV_RMSSD",
        HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL: "HRV_SDNN",
    }
    requested_time = set(statistics).intersection(time_mapping)
    if requested_time and not has_internal_gap:
        frame = nk.hrv_time(intervals, show=False)
        for name in requested_time:
            result[name] = frame.iloc[0].get(time_mapping[name], np.nan)
    elif requested_time:
        if HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES in requested_time:
            result[HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES] = (
                _compute_rmssd(values, segments)
            )
        if HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL in requested_time:
            result[HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL] = compute_sdnn(
                values
            )
    if HRV_STATISTIC_PNN20 in statistics:
        result[HRV_STATISTIC_PNN20] = _compute_pnn(values, 20, segments)
    if HRV_STATISTIC_PNN50 in statistics:
        result[HRV_STATISTIC_PNN50] = _compute_pnn(values, 50, segments)
    if HRV_STATISTIC_BBI_MEDIAN in statistics:
        result[HRV_STATISTIC_BBI_MEDIAN] = float(np.median(values))

    frequency_mapping = {
        HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER: "HRV_VLF",
        HRV_STATISTIC_LOW_FREQUENCY_POWER: "HRV_LF",
        HRV_STATISTIC_HIGH_FREQUENCY_POWER: "HRV_HF",
        HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO: "HRV_LFHF",
    }
    requested_frequency = set(statistics).intersection(frequency_mapping)
    if requested_frequency and not has_internal_gap:
        frame = nk.hrv_frequency(
            intervals,
            show=False,
            normalize=False,
            interpolation_rate=4,
        )
        for name in requested_frequency:
            result[name] = frame.iloc[0].get(frequency_mapping[name], np.nan)
    return result


def _window_statistics(values, statistics, backend: str, segment_ids=None) -> dict:
    if backend == "pywearable":
        return _pywearable_window_statistics(values, statistics, segment_ids)
    if backend == "neurokit":
        return _neurokit_window_statistics(values, statistics, segment_ids)
    raise ValueError("backend must be 'pywearable' or 'neurokit'")


def _statistics_list(statistic) -> list[str]:
    if statistic is None:
        return list(HRV_STATISTICS)
    statistics = [statistic] if isinstance(statistic, str) else list(statistic)
    unknown = set(statistics).difference(HRV_STATISTICS)
    if unknown:
        raise ValueError(
            f"statistic must be one among {list(HRV_STATISTICS)}, "
            f"not {sorted(unknown)}"
        )
    return list(dict.fromkeys(statistics))


def _resolve_time_col(loader, time_reference: str, chain) -> str:
    """Select the available clock appropriate for the requested hierarchy."""
    available = getattr(loader, "hrv_time_columns", schema.BBI_TIME_COLUMNS)
    return hierarchy.resolve_time_col(available, time_reference, chain, "HRV")


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


def _primary_windows(user_id, start, end, cycle_intervals=None):
    """Construct the required five-minute grid, omitting inter-cycle gaps."""
    if cycle_intervals is None:
        return hierarchy.fixed_windows(
            [user_id], HRV_PRIMARY_RESOLUTION, {user_id: (start, end)}
        ).xs(user_id)
    duration = pd.Timedelta(HRV_PRIMARY_RESOLUTION)
    starts = []
    valid = cycle_intervals[cycle_intervals["interval_valid"]]
    for interval in valid.itertuples():
        starts.extend(
            pd.date_range(
                interval.start_time.floor(duration),
                interval.end_time,
                freq=duration,
                inclusive="left",
            )
        )
    starts = pd.DatetimeIndex(starts).drop_duplicates().sort_values()
    return pd.DataFrame(
        {"start_time": starts, "end_time": starts + duration}, index=starts
    ).rename_axis("window_start")


def _asleep_stage_membership(loader, user_id, timestamps, start_date, end_date):
    """Return asleep membership and contiguous asleep-block identifiers."""
    stages = schema.prepare_sleep_stages(
        loader.load_sleep_stage(user_id, start_date, end_date)
    )
    if stages.empty:
        return np.ones(len(timestamps), dtype=bool), np.zeros(len(timestamps), dtype=int)
    if "local_time_uncertain" in stages.columns:
        stages = stages[~stages["local_time_uncertain"].fillna(False)]
    stages = stages[
        ~stages[schema.SLEEP_STAGE_SLEEP_TYPE_COL].isin(
            {
                schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
            }
        )
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


def _asleep_stage_mask(loader, user_id, timestamps, start_date, end_date):
    """Mark samples in sleep stages, retaining all samples if stages are absent."""
    return _asleep_stage_membership(
        loader, user_id, timestamps, start_date, end_date
    )[0]


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
    cycle_intervals=None,
    sleep_stage_loader=None,
):
    bbi = schema.prepare_bbi(bbi)
    bbi = bbi[bbi[time_col].notna()].sort_values(time_col, kind="stable")
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
                sleep_stage_loader, user_id, bbi[time_col], start, end
            )
            bbi = bbi[asleep].copy()
            bbi["_hrv_segment"] = segments[asleep]
            bbi = bbi.reset_index(drop=True)

    windows = _primary_windows(user_id, start, end, cycle_intervals)
    labels = (
        bbi[time_col].dt.floor(HRV_PRIMARY_RESOLUTION)
        if not bbi.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    groups = {label: rows for label, rows in bbi.groupby(labels, sort=False)}
    records = []
    for label, window in windows.iterrows():
        rows = groups.get(label, bbi.iloc[0:0])
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
                rows["_hrv_segment"] if "_hrv_segment" in rows else None,
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
    cycle_intervals=None,
    sleep_only=False,
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
        cycle_intervals=cycle_intervals,
        sleep_stage_loader=loader if sleep_only else None,
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
    cycle_intervals=None,
):
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
    windows = _primary_windows(user_id, start, end, cycle_intervals)
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
    cycle_parent=None,
    sleep_only=False,
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
                cycle_intervals=user_intervals,
                sleep_only=sleep_only,
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
    time_col: str = schema.UTC_TIME_COL,
) -> pd.DataFrame:
    """Calculate non-overlapping HRV windows from an already loaded BBI table."""
    statistics = _statistics_list(statistics)
    resolution = hierarchy.canonical_resolution(window_size)
    if resolution != HRV_PRIMARY_RESOLUTION:
        raise ValueError("window_size must currently be five minutes")
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
    )
    return trajectory.droplevel(schema.USER_COL)


def _expanded_chain(resolution):
    """Canonicalize HRV resolution shorthand and prepend the five-minute base."""
    chain, hierarchical = hierarchy.normalize_resolution_chain(resolution)
    chain = [cycle.CYCLE_NIGHT if value == "nightly" else value for value in chain]
    if chain[0] != HRV_PRIMARY_RESOLUTION:
        if chain[0] not in cycle.CYCLE_TYPES and chain[0] not in hierarchy.CALENDAR_RESOLUTIONS:
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
        cycle_parent=cycle_parent,
        sleep_only=cycle_target == cycle.CYCLE_NIGHT
        and getattr(loader, "hrv_source", "bbi") == "bbi",
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
        )
    index_name = (
        schema.CALENDAR_DATE_COL
        if chain[-1] in cycle.CYCLE_TYPES
        or chain[-1] in hierarchy.CALENDAR_RESOLUTIONS
        else time_col
    )
    return trajectory.rename_axis(index={"window_start": index_name})


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
) -> pd.DataFrame:
    """Return HRV features at five minutes or along a resolution hierarchy.

    Raw-BBI loaders calculate non-overlapping five-minute features. Provider
    loaders use their five-minute estimates and leave unsupported statistics
    missing. A scalar calendar or cycle resolution is shorthand for a
    five-minute-to-parent hierarchy. Source selection is fixed by the loader;
    missing provider values are never filled from a different HRV source.

    ``min_coverage`` supplies one threshold per level. At the primary raw-BBI
    level, coverage is represented BBI duration divided by window duration.
    Provider coverage is used when available; an explicit primary threshold
    rejects provider samples whose coverage is unavailable. Higher-level
    coverage is the fraction of valid child features.

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
    loader_kwargs = {} if loader_kwargs is None else dict(loader_kwargs)
    cycle_kwargs = {} if cycle_kwargs is None else dict(cycle_kwargs)
    statistics = _statistics_list(statistic)
    users = get_user_ids(loader, user_id)

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
        )
        columns = [
            column
            for name in statistics
            for column in (name, f"{name}__coverage")
        ]
        result = result[columns].rename_axis(index={"window_start": time_col})
        return result

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
):
    """Return non-overlapping five-minute HRV features."""
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
    "HRV_STATISTICS",
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
