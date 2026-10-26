"""Fixed-date seasons and calendar quarters share normal hierarchy semantics."""

import numpy as np
import pandas as pd
import pytest

from pywearable import activity, adherence, heart_rate, hierarchy, hrv, provider_features, respiration, sleep
from pywearable.utils import time as time_utils
from tests.helpers import LoaderBase


RESOLUTIONS = ["seasonal", "seasonal_meteorological", "quarterly"]
BOUNDARIES = ["2024-03-21", "2024-03-01", "2024-04-01"]


class SeasonalLoader(LoaderBase):
    activity_source = "daily_summary"
    activity_primary_resolution = "daily"
    respiratory_rate_source = "provider_summary"
    spo2_source = "provider_summary"
    provider_features = frozenset({"oura_readiness_score"})

    def __init__(self, boundary):
        self.days = pd.date_range(pd.Timestamp(boundary) - pd.Timedelta(days=1), periods=2)
        local = self.days - pd.Timedelta(hours=1)
        self.sleep = pd.DataFrame({
            "calendar_date": self.days, "sleep_summary_id": ["night-1", "night-2"],
            "local_time": local, "duration_in_ms": [360 * 60_000, 480 * 60_000],
            "awake_duration_in_ms": 60 * 60_000, "unmeasurable_sleep_duration_in_ms": 0,
        })
        times = pd.DatetimeIndex([
            day + pd.Timedelta(hours=12, minutes=minute)
            for day in self.days for minute in range(60)
        ])
        self.hr = pd.DataFrame({
            "local_time": times, "utc_time": times - pd.Timedelta(hours=2),
            "timezone": pd.Timedelta(hours=2), "heart_rate": np.repeat([60., 80.], 60),
        })
        intervals = np.tile([1000., 1100.], 143)[:285]
        elapsed = pd.to_timedelta(np.r_[0, np.cumsum(intervals[:-1])], unit="ms")
        times = pd.DatetimeIndex([day + pd.Timedelta(hours=2) + step for day in self.days for step in elapsed])
        self.bbi = pd.DataFrame({
            "local_time": times, "utc_time": times - pd.Timedelta(hours=2),
            "timezone": pd.Timedelta(hours=2), "bbi": np.tile(intervals, 2),
        })

    def get_user_ids(self):
        return ["user"]

    @staticmethod
    def _slice(frame, column, start_date, end_date):
        result = frame.copy()
        if start_date is not None:
            result = result[result[column].ge(pd.Timestamp(start_date))]
        if end_date is not None:
            result = result[result[column].lt(pd.Timestamp(end_date))]
        return result.reset_index(drop=True)

    def load_sleep_summary(self, user_id, start_date=None, end_date=None, **kwargs):
        return self._slice(self.sleep, "calendar_date", start_date, end_date)

    def load_sleep_stage(self, *args, **kwargs):
        return pd.DataFrame()

    def load_heart_rate(self, user_id, start_date=None, end_date=None, time_col="local_time"):
        return self._slice(self.hr, time_col, start_date, end_date)

    def load_bbi(self, user_id, start_date=None, end_date=None):
        return self._slice(self.bbi, "local_time", start_date, end_date)

    def load_activity_summary(self, user_id, start_date=None, end_date=None):
        frame = pd.DataFrame({"calendar_date": self.days, "activity_steps": [1000., 3000.]})
        return self._slice(frame, "calendar_date", start_date, end_date)

    def load_respiration_summary(self, user_id, start_date=None, end_date=None):
        frame = pd.DataFrame({
            "calendar_date": self.days, "respiratory_rate_mean": [12., 16.], "respiratory_rate_coverage": 1.,
        })
        return self._slice(frame, "calendar_date", start_date, end_date)

    def load_respiration(self, user_id, start_date=None, end_date=None):
        frame = self.hr.assign(respiratory_rate=self.hr["heart_rate"] / 5)
        return self._slice(frame, "local_time", start_date, end_date)

    def load_pulse_ox(self, user_id, start_date=None, end_date=None):
        frame = self.hr.assign(spo2=np.where(self.hr["heart_rate"].eq(60), 97., 95.))
        return self._slice(frame, "local_time", start_date, end_date)

    def load_spo2_summary(self, user_id, start_date=None, end_date=None):
        frame = pd.DataFrame({"calendar_date": self.days, "spo2_mean": [97., 95.], "spo2_coverage": 1.})
        return self._slice(frame, "calendar_date", start_date, end_date)

    def load_provider_feature(self, feature, user_id, start_date=None, end_date=None):
        if feature == "garmin_stress_level":
            frame = self.hr.assign(garmin_stress_level=self.hr["heart_rate"])
            return self._slice(frame, "local_time", start_date, end_date)
        frame = pd.DataFrame({"calendar_date": self.days, feature: [70., 90.]})
        return self._slice(frame, "calendar_date", start_date, end_date)


@pytest.mark.parametrize("resolution,rule", [
    ("seasonal", "seasonal"), ("seasonal_meteorological", "QS-DEC"), ("quarterly", "QS-JAN"),
])
def test_calendar_specs_and_quarter_alias(resolution, rule):
    spec = time_utils.get_resampling_spec(resolution)
    assert spec.rule == rule
    assert spec.calendar_based
    assert spec.time_col == "local_time"
    assert hierarchy.fixed_duration(resolution) is None
    assert hierarchy.canonical_resolution("quarter") == "quarterly"
    assert time_utils.get_resampling_spec("quarter").rule == "QS-JAN"


@pytest.mark.parametrize("resolution,starts", [
    ("seasonal", ["2024-03-21", "2024-06-21", "2024-09-21", "2024-12-21"]),
    ("seasonal_meteorological", ["2024-03-01", "2024-06-01", "2024-09-01", "2024-12-01"]),
    ("quarterly", ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]),
])
def test_all_four_midnight_boundaries_are_half_open(resolution, starts):
    rule = time_utils.get_resampling_spec(resolution).rule
    for text in starts:
        boundary = pd.Timestamp(text)
        before = boundary - pd.Timedelta(nanoseconds=1)
        assert time_utils.get_calendar_period_start(boundary, rule) == boundary
        assert time_utils.get_calendar_period_start(before, rule) < boundary
        assert time_utils.get_calendar_period_end(before, rule) == boundary
        assert time_utils.get_calendar_period_end(boundary, rule) == boundary + pd.DateOffset(months=3)


@pytest.mark.parametrize("resolution,expected", [
    ("seasonal", ["2024-02-15 12:00", "2024-03-21", "2024-06-21", "2024-09-21"]),
    ("seasonal_meteorological", ["2024-02-15 12:00", "2024-03-01", "2024-06-01", "2024-09-01"]),
    ("quarterly", ["2024-02-15 12:00", "2024-04-01", "2024-07-01", "2024-10-01"]),
])
def test_partial_windows_keep_exact_requested_bounds(resolution, expected):
    start, end = pd.Timestamp("2024-02-15 12:00"), pd.Timestamp("2024-10-05 06:00")
    windows = hierarchy.calendar_windows(["user"], resolution, {"user": (start, end)}).xs("user")
    pd.testing.assert_index_equal(windows.index, pd.DatetimeIndex(expected, name="window_start"))
    assert windows.iloc[0].start_time == start
    assert windows.iloc[-1].end_time == end
    assert windows.end_time.iloc[:-1].tolist() == windows.start_time.iloc[1:].tolist()


@pytest.mark.parametrize("include_end", [False, True])
def test_seasonal_grid_preserves_exact_endpoints_across_years(include_end):
    start, end = pd.Timestamp("2023-12-22 12:00"), pd.Timestamp("2025-03-21")
    expected = [start, *pd.to_datetime(["2024-03-21", "2024-06-21", "2024-09-21", "2024-12-21"])]
    if include_end:
        expected.append(end)
    actual = time_utils.get_calendar_window_starts(start, end, "seasonal", include_end=include_end)
    pd.testing.assert_index_equal(actual, pd.DatetimeIndex(expected))


@pytest.mark.parametrize("resolution,begin,end", [
    ("seasonal", "2023-12-21", "2024-03-21"),
    ("seasonal_meteorological", "2023-12-01", "2024-03-01"),
    ("quarterly", "2024-01-01", "2024-04-01"),
])
def test_leap_day_and_year_crossing_bounds(resolution, begin, end):
    rule = time_utils.get_resampling_spec(resolution).rule
    assert time_utils.get_calendar_period_start("2024-02-29", rule) == pd.Timestamp(begin)
    assert time_utils.get_calendar_period_end("2024-02-29", rule) == pd.Timestamp(end)
    frame = pd.DataFrame({"local_time": pd.to_datetime(["2024-02-29"])})
    assert hierarchy.observed_bounds(frame, "local_time", None, None, resolution) == (
        pd.Timestamp(begin), pd.Timestamp(end)
    )


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_hierarchy_order_and_non_interchangeable_three_month_windows(resolution):
    for primary in ["daily", "weekly", "monthly", "nightly", "cycle_night", "89D"]:
        assert hierarchy.normalize_resolution_chain([primary, resolution, "yearly"])[1]
    for other in RESOLUTIONS:
        with pytest.raises(ValueError, match="smaller to larger"):
            hierarchy.normalize_resolution_chain([resolution, other])
    with pytest.raises(ValueError, match="smaller to larger"):
        hierarchy.normalize_resolution_chain(["90D", resolution])


@pytest.mark.parametrize("resolution,boundary", list(zip(RESOLUTIONS, BOUNDARIES)))
@pytest.mark.parametrize("domain", ["sleep", "heart_rate", "activity", "respiration", "spo2", "hrv", "provider", "adherence"])
def test_all_domains_split_at_their_selected_boundary(resolution, boundary, domain):
    loader = SeasonalLoader(boundary)
    kwargs = dict(user_id="user", start_date=loader.days[0], end_date=loader.days[-1] + pd.Timedelta(days=1))
    if domain == "sleep":
        result = sleep.get_sleep_statistics(loader, statistic="sleep_TST", resolution=resolution, aggregation="mean", **kwargs)
        prefix, expected = "sleep_TST", [360., 480.]
    elif domain == "heart_rate":
        result = heart_rate.get_heart_rate_statistics(loader, statistic="heart_rate_mean", resolution=resolution, **kwargs)
        prefix, expected = "heart_rate_mean", [60., 80.]
    elif domain == "activity":
        result = activity.get_activity_statistics(
            loader, statistic="activity_steps", resolution=resolution, aggregation="mean", **kwargs
        )
        prefix, expected = "activity_steps", [1000., 3000.]
    elif domain in {"respiration", "spo2"}:
        prefix, expected = ("respiratory_rate_mean", [12., 16.]) if domain == "respiration" else ("spo2_mean", [97., 95.])
        result = respiration.get_respiration_statistics(
            loader, statistic=prefix, resolution=resolution, aggregation="mean", **kwargs
        )
    elif domain == "hrv":
        result = hrv.get_hrv_statistics(loader, statistic="hrv_rmssd", resolution=resolution, aggregation="mean", **kwargs)
        prefix, expected = "hrv_rmssd", [100., 100.]
    elif domain == "provider":
        result = provider_features.get_provider_features(
            loader, feature="oura_readiness_score", resolution=resolution, aggregation="mean", **kwargs
        )
        prefix, expected = "oura_readiness_score", [70., 90.]
    else:
        result = adherence.get_adherence(loader, metric="sleep", resolution=resolution, **kwargs)
        prefix, expected = "sleep_adherence", [1., 1.]
    feature = next(column for column in result if column.startswith(prefix) and not column.endswith("__coverage"))
    np.testing.assert_allclose(result[feature], expected)
    np.testing.assert_array_equal(result.index.get_level_values(1), loader.days)


@pytest.mark.parametrize("resolution,boundary", list(zip(RESOLUTIONS, BOUNDARIES)))
def test_nightly_label_assignment_coverage_and_yearly_parent(resolution, boundary):
    loader = SeasonalLoader(boundary)
    kwargs = dict(user_id="user", start_date=loader.days[0], end_date=loader.days[-1] + pd.Timedelta(days=2))
    result = sleep.get_sleep_statistics(
        loader, statistic="sleep_TST", resolution=["nightly", resolution], aggregation="mean", **kwargs
    )
    column = f"sleep_TST__nightly__{resolution}_mean"
    # The second night begins before the boundary but belongs to its sleep date.
    np.testing.assert_allclose(result[column], [360., 480.])
    np.testing.assert_allclose(result[f"{column}__coverage"], [1., .5])
    filtered = sleep.get_sleep_statistics(
        loader, statistic="sleep_TST", resolution=["nightly", resolution], aggregation="mean",
        min_valid_children=[None, 2], **kwargs
    )
    assert filtered[column].isna().all()
    annual = sleep.get_sleep_statistics(
        loader, statistic="sleep_TST", resolution=["nightly", resolution, "yearly"],
        aggregation="mean", **kwargs
    )
    assert annual[f"{column}__yearly_mean"].iloc[0] == 420.


@pytest.mark.parametrize("resolution,boundary", list(zip(RESOLUTIONS, BOUNDARIES)))
@pytest.mark.parametrize("feature,expected", [
    ("respiratory_rate_mean", [12., 16.]), ("spo2_mean", [97., 95.]),
])
def test_timestamped_respiratory_streams_use_selected_calendar(resolution, boundary, feature, expected):
    loader = SeasonalLoader(boundary)
    loader.respiratory_rate_source = loader.spo2_source = "samples"
    result = respiration.get_respiration_statistics(
        loader, statistic=feature, resolution=resolution, user_id="user",
        start_date=loader.days[0], end_date=loader.days[-1] + pd.Timedelta(days=1),
    )
    np.testing.assert_allclose(result[feature], expected)


@pytest.mark.parametrize("resolution,boundary", list(zip(RESOLUTIONS, BOUNDARIES)))
def test_physical_adherence_and_provider_samples_split_at_boundary(resolution, boundary):
    loader = SeasonalLoader(boundary)
    kwargs = dict(user_id="user", start_date=loader.days[0], end_date=loader.days[-1] + pd.Timedelta(days=1))
    wearing = adherence.get_adherence(loader, metric=["heart_rate", "bbi"], resolution=resolution, **kwargs)
    np.testing.assert_allclose(wearing["heart_rate_adherence"], [60 / 1440] * 2)
    represented_seconds = loader.bbi.groupby(loader.bbi.local_time.dt.normalize()).bbi.sum() / 1000
    np.testing.assert_allclose(wearing["bbi_adherence"], represented_seconds / 86_400)
    loader.provider_features = frozenset({"garmin_stress_level"})
    stress = provider_features.get_provider_features(
        loader, feature="garmin_stress_level", statistic="mean", resolution=resolution, **kwargs
    )
    np.testing.assert_allclose(stress["garmin_stress_level_mean"], [60., 80.])


def test_quarter_alias_matches_canonical_public_output():
    loader = SeasonalLoader("2024-04-01")
    kwargs = dict(user_id="user", start_date=loader.days[0], end_date=loader.days[-1] + pd.Timedelta(days=1))
    for extract, options in [
        (heart_rate.get_heart_rate_statistics, dict(statistic="heart_rate_mean")),
        (sleep.get_sleep_statistics, dict(statistic="sleep_TST", aggregation="mean")),
    ]:
        expected = extract(loader, resolution="quarterly", **options, **kwargs)
        actual = extract(loader, resolution="quarter", **options, **kwargs)
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("resolution,boundary", list(zip(RESOLUTIONS, BOUNDARIES)))
def test_calendar_clock_defaults_to_local_but_honors_utc(resolution, boundary):
    loader = SeasonalLoader(boundary)
    loader.hr = loader.hr.iloc[[0, -1]].copy()
    loader.hr["local_time"] = pd.to_datetime([
        pd.Timestamp(boundary) - pd.Timedelta(minutes=30),
        pd.Timestamp(boundary) + pd.Timedelta(minutes=30),
    ])
    loader.hr["utc_time"] = loader.hr["local_time"] - pd.Timedelta(hours=2)
    kwargs = dict(user_id="user", start_date=loader.days[0], end_date=loader.days[-1] + pd.Timedelta(days=1))
    local = heart_rate.get_heart_rate_statistics(
        loader, statistic="heart_rate_mean", resolution=resolution, time_reference="local", **kwargs
    )
    automatic = heart_rate.get_heart_rate_statistics(
        loader, statistic="heart_rate_mean", resolution=resolution, **kwargs
    )
    utc = heart_rate.get_heart_rate_statistics(
        loader, statistic="heart_rate_mean", resolution=resolution, time_reference="utc", **kwargs
    )
    pd.testing.assert_frame_equal(automatic, local)
    np.testing.assert_allclose(local["heart_rate_mean"], [60., 80.])
    np.testing.assert_allclose(utc["heart_rate_mean"], [70., np.nan], equal_nan=True)


@pytest.mark.parametrize("resolution,boundary", list(zip(RESOLUTIONS, BOUNDARIES)))
def test_nightly_hrv_can_feed_each_calendar_parent(resolution, boundary):
    loader = SeasonalLoader(boundary)
    result = hrv.get_hrv_statistics(
        loader, statistic="hrv_rmssd", resolution=["5min", "cycle_night", resolution],
        aggregation="mean", sleep_stages_to_remove=None, user_id="user",
        start_date=loader.days[0], end_date=loader.days[-1] + pd.Timedelta(days=1),
    )
    column = f"hrv_rmssd__5min__cycle_night_mean__{resolution}_mean"
    np.testing.assert_allclose(result[column], [100., 100.])


def test_seasonal_heart_rate_retains_empty_intermediate_periods_without_bounds():
    loader = SeasonalLoader("2024-03-21")
    loader.hr = loader.hr.iloc[[0, -1]].copy()
    loader.hr["local_time"] = pd.to_datetime(["2023-12-22", "2024-09-22"])
    loader.hr["utc_time"] = loader.hr["local_time"] - pd.Timedelta(hours=2)
    result = heart_rate.get_heart_rate_statistics(
        loader, statistic=["heart_rate_mean", "heart_rate_count"], user_id="user", resolution="seasonal"
    )
    assert len(result) == 4
    assert result["heart_rate_mean"].iloc[1:3].isna().all()
    np.testing.assert_array_equal(result["heart_rate_count"], [1, 0, 0, 1])
