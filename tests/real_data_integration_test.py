"""Real-export regression checks, independent of private data and network drives."""

import csv
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pywearable import (
    activity,
    adherence,
    cycle,
    heart_rate,
    hrv,
    provider_features,
    respiration,
    sleep,
    trajectory,
    visualization,
)
from pywearable.loader import LabfrontLoader, OuraLoader


FIXTURES = Path(__file__).parent / "data"
USER = "participant-01_00000001"
START, END = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01")
DATES = pd.date_range(START, END, inclusive="left")
BOUNDS = dict(user_id=USER, start_date=START, end_date=END)
IDENTITY_METADATA = {
    "projectId": "00000000-0000-4000-8000-000000000001",
    "projectTitle": "Wearable regression fixtures",
    "participantId": "00000001-0000-4000-8000-000000000001",
    "participantInsignia": "participant-01",
    "taskId": "00000000-0000-4000-8000-000000000002",
    "taskScheduleRepeat": "false",
    "taskScheduleDescription": "",
    "garminDeviceId": "00000000-0000-4000-8000-000000000003",
}


@pytest.fixture(scope="module")
def loaders(tmp_path_factory):
    # Labfront builds an absolute-path index; never cache it inside the fixtures.
    result = {}
    for source, loader_type in [("labfront", LabfrontLoader), ("oura", OuraLoader)]:
        root = tmp_path_factory.mktemp(f"real-data-{source}") / source
        shutil.copytree(
            FIXTURES / source, root,
            ignore=shutil.ignore_patterns(".pywearable_file_index.csv"),
        )
        result[source] = loader_type(root)
    return result


@pytest.fixture(scope="module", params=["labfront", "oura"])
def wearable(request, loaders):
    return request.param, loaders[request.param]


@pytest.fixture(scope="module")
def labfront(loaders):
    return loaders["labfront"]


@lru_cache
def _csv(metric):
    """Read original-format rows without going through a pywearable loader."""
    frames = []
    for path in sorted((FIXTURES / "labfront" / USER / metric).glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            header_length = int(next(csv.reader(stream))[1])
        frames.append(pd.read_csv(path, skiprows=header_length + 1))
    return pd.concat(frames, ignore_index=True)


@lru_cache
def _json(metric):
    rows = []
    for path in sorted((FIXTURES / "oura" / USER / metric).glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["data"])
    return rows


def _april(frame, column):
    dates = pd.to_datetime(frame[column])
    return frame.loc[dates.ge(START) & dates.lt(END)].copy()


def _raw_nights(source):
    """Select main nights using native source fields and snapshot priorities."""
    if source == "oura":
        rows = pd.DataFrame([r for r in _json("sleep") if r["type"] == "long_sleep"])
        rows = _april(rows, "day")
        return rows.set_index(pd.to_datetime(rows["day"]))
    rows = _april(_csv("garmin-connect-sleep-summary"), "calendarDate")
    rows["rank"] = rows["validation"].map(
        {"AUTO_TENTATIVE": 1, "AUTO_FINAL": 2, "ENHANCED_TENTATIVE": 3, "ENHANCED_FINAL": 4}
    )
    rows = rows.sort_values(["calendarDate", "rank", "durationInMs"])
    rows = rows.drop_duplicates("calendarDate", keep="last")
    return rows.set_index(pd.to_datetime(rows["calendarDate"]))


def _assert_clock(frame):
    known = frame[["utc_time", "local_time", "timezone"]].dropna()
    assert not known.empty
    pd.testing.assert_series_equal(
        known["local_time"] - known["utc_time"], known["timezone"], check_names=False
    )


def _assert_json_identity_fields(value):
    """Reject identifying fields and non-pseudonymous IDs in fixture JSON."""
    if isinstance(value, list):
        for item in value:
            _assert_json_identity_fields(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            assert key.lower() not in {
                "name", "email", "first_name", "last_name", "user_id", "serial_number",
                "latitude", "longitude", "access_token", "refresh_token",
            }
            if key == "id":
                assert re.fullmatch(r"oura-record-\d{5}", str(item))
            _assert_json_identity_fields(item)


def test_fixture_identity_and_period_are_bounded(wearable):
    source, loader = wearable
    assert loader.get_user_ids() == [USER]
    for path in (FIXTURES / source / USER).rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(
            r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}|(?<![a-z0-9])[a-z]:[\\/]|/(?:home|Users)/|"
            r"(?:access|refresh)[_-]?token|client[_-]?secret|password|api[_-]?key",
            text, flags=re.IGNORECASE,
        )
        uuids = {
            value.lower()
            for value in re.findall(
                r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", text, flags=re.IGNORECASE
            )
        }
        assert uuids.issubset(IDENTITY_METADATA.values())
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream)
                header = [next(reader) for _ in range(5)]
                next(reader)
                fields = next(reader)
                id_positions = [i for i, field in enumerate(fields) if field.endswith("Id")]
                for row in reader:
                    assert all(re.fullmatch(r"record-\d{5}", row[i]) for i in id_positions)
            identity = dict(zip(header[3], header[4]))
            assert {"projectId", "projectTitle", "participantId", "participantInsignia"}.issubset(identity)
            assert identity == {key: IDENTITY_METADATA[key] for key in identity}
            assert USER in path.name
        elif path.suffix == ".json":
            page = json.loads(text)
            assert page["next_token"] is None
            _assert_json_identity_fields(page["data"])
            for row in page["data"]:
                if "id" in row:
                    assert row["id"].startswith("oura-record-")
                if "day" in row:
                    assert "2024-03-31" <= row["day"] < "2024-05-02"
    if source == "labfront":
        bbi = _csv("garmin-device-bbi")
        assert len(bbi) == 76_077
        assert bbi["isoDate"].str[:10].unique().tolist() == ["2024-04-11"]
        assert not (FIXTURES / source / USER / "garmin-device-stress").exists()


def test_sleep_features_match_native_summaries_and_missing_nights(wearable):
    source, loader = wearable
    native = _raw_nights(source)
    assert len(native) == (23 if source == "labfront" else 25)
    result = sleep.get_sleep_statistics(loader, **BOUNDS).xs(USER)
    pd.testing.assert_index_equal(result.index, DATES.rename("calendar_date"))
    if source == "oura":
        tst = native["total_sleep_duration"] / 60
        tib = (native["total_sleep_duration"] + native["awake_time"]) / 60
        scores = pd.DataFrame(_json("daily_sleep")).set_index("day")["score"]
        scores.index = pd.to_datetime(scores.index)
    else:
        tst = (native["durationInMs"] - native["unmeasurableSleepInMs"]) / 60_000
        tib = (native["durationInMs"] + native["awakeDurationInMs"]) / 60_000
        scores = native["overallSleepScore"]
    np.testing.assert_allclose(result["sleep_TST"], tst.reindex(DATES), equal_nan=True)
    np.testing.assert_allclose(result["sleep_SE"], (100 * tst / tib).reindex(DATES), equal_nan=True)
    np.testing.assert_allclose(result["sleep_score"], scores.reindex(DATES), equal_nan=True)
    assert result.loc[~result.index.isin(native.index), "sleep_TST__coverage"].eq(0).all()
    assert result["sleep_bedtime"].notna().sum() == len(native)
    assert result["sleep_stage_coverage"].notna().sum() == len(native)
    stages = loader.load_sleep_stage(USER, START, END)
    assert len(stages) == (473 if source == "labfront" else 633)
    assert set(stages["sleep_summary_id"]).issubset(
        loader.load_sleep_summary(USER, START, END)["sleep_summary_id"]
    )


def test_24h_sleep_totals_match_native_oura_stage_grid(loaders):
    expected = {}
    nap_sleep = {}
    for record in _json("sleep"):
        start = pd.Timestamp(record["bedtime_start"]).tz_localize(None)
        end = pd.Timestamp(record["bedtime_end"]).tz_localize(None)
        day = start.normalize()
        if start.hour >= 18:
            day += pd.Timedelta(days=1)
        if day not in DATES:
            continue
        asleep, classified, duration = expected.get(day, (0.0, 0.0, 0.0))
        duration += (end - start).total_seconds()
        for position, phase in enumerate(record["sleep_phase_5_min"]):
            epoch_start = start + pd.Timedelta(minutes=5 * position)
            epoch_end = min(epoch_start + pd.Timedelta(minutes=5), end)
            seconds = max(0.0, (epoch_end - epoch_start).total_seconds())
            if phase in "1234":
                classified += seconds
            if phase in "123":
                asleep += seconds
                if record["type"] != "long_sleep":
                    nap_sleep[day] = nap_sleep.get(day, 0.0) + seconds
        expected[day] = asleep, classified, duration
    result = sleep.get_total_sleep_time_24h(loaders["oura"], **BOUNDS).xs(USER)
    pd.testing.assert_index_equal(result.index, DATES.rename("calendar_date"))
    for day, row in result.iterrows():
        asleep, classified, duration = expected.get(day, (0.0, 0.0, 0.0))
        if classified:
            assert row["sleep_TST_24h"] == pytest.approx(asleep / 60)
            assert row["sleep_TST_24h__coverage"] == pytest.approx(classified / duration)
        else:
            assert pd.isna(row["sleep_TST_24h"])
            assert row["sleep_TST_24h__coverage"] == 0
    assert nap_sleep[pd.Timestamp("2024-04-06")] > 0
    with pytest.raises(ValueError, match="nightly-only"):
        sleep.get_total_sleep_time_24h(loaders["labfront"], **BOUNDS)


def test_monthly_heart_rate_timestamps_and_unrecoverable_local_offsets(wearable):
    source, loader = wearable
    if source == "oura":
        with pytest.warns(UserWarning, match="no unambiguous local offset"):
            samples = loader.load_heart_rate(USER, START, END)
        assert len(samples) == 7356
        utc_samples = loader.load_heart_rate(USER, time_col="utc_time")
        unknown = utc_samples["local_time_uncertain"]
        assert unknown.any()
        assert utc_samples.loc[unknown, "local_time"].isna().all()
        assert utc_samples.loc[unknown, "utc_time"].notna().all()
    else:
        samples = loader.load_heart_rate(USER, START, END)
        assert len(samples) == 104_512
        assert samples["local_time"].notna().all()
    _assert_clock(samples)


def test_sleep_regularity_has_real_comparable_epochs(wearable):
    _, loader = wearable
    regularity = sleep.get_sleep_regularity(
        loader, user_id=USER, start_date="2024-04-10", end_date="2024-04-17"
    )
    values = regularity["sleep_SRI"].dropna()
    assert not values.empty
    assert values.between(-100, 100).all()
    coverage = regularity["sleep_sri_coverage"].dropna()
    assert coverage.between(0, 100).all()
    assert coverage.gt(0).any()


def test_weekly_sleep_uses_equal_night_weights_and_calendar_coverage(wearable):
    _, loader = wearable
    nightly = sleep.get_sleep_statistics(loader, statistic="sleep_TST", **BOUNDS).xs(USER)
    weekly = sleep.get_sleep_statistics(
        loader, statistic="sleep_TST", resolution=("nightly", "weekly"),
        aggregation="mean", min_valid_children=(None, 2), **BOUNDS,
    ).xs(USER)
    first_week = nightly.loc["2024-04-01":"2024-04-07", "sleep_TST"]
    feature = "sleep_TST__nightly__weekly_mean"
    assert weekly.loc[START, feature] == pytest.approx(first_week.mean())
    assert weekly.loc[START, f"{feature}__coverage"] == pytest.approx(first_week.notna().sum() / 7)


@pytest.mark.parametrize("resolution", ["seasonal", "seasonal_meteorological", "quarterly"])
def test_three_month_sleep_resolutions_keep_real_april_values_and_coverage(wearable, resolution):
    source, loader = wearable
    native = _raw_nights(source)
    expected = (
        native["total_sleep_duration"] / 60
        if source == "oura"
        else (native["durationInMs"] - native["unmeasurableSleepInMs"]) / 60_000
    )
    result = sleep.get_sleep_statistics(
        loader, statistic="sleep_TST", resolution=resolution, aggregation="mean", **BOUNDS
    )
    column = f"sleep_TST__nightly__{resolution}_mean"
    assert len(result) == 1
    assert result.iloc[0][column] == pytest.approx(expected.mean())
    assert result.iloc[0][f"{column}__coverage"] == pytest.approx(len(native) / 30)
    assert result.iloc[0]["period_start"] == START
    assert result.iloc[0]["period_end"] == END
    filtered = sleep.get_sleep_statistics(
        loader, statistic="sleep_TST", resolution=resolution, aggregation="mean", min_coverage=.9, **BOUNDS
    )
    assert filtered[column].isna().all()
    assert filtered.iloc[0][f"{column}__coverage"] == pytest.approx(len(native) / 30)


def test_sleep_cycles_match_night_boundaries_and_retain_context(wearable):
    source, loader = wearable
    summaries = loader.load_sleep_summary(USER, START, END)
    nights = cycle.get_intervals(loader, cycle.CYCLE_NIGHT, **BOUNDS).xs(USER)
    assert nights["interval_valid"].sum() == len(summaries)
    for row in summaries.itertuples():
        night = nights.loc[pd.Timestamp(row.calendar_date)]
        assert night["start_time"] == row.local_time
        duration = row.duration_in_ms + row.awake_duration_in_ms
        assert night["end_time"] == row.local_time + pd.Timedelta(milliseconds=duration)
    for name in (cycle.CYCLE_AWAKE, cycle.CYCLE_PHYSIOLOGICAL):
        intervals = cycle.get_intervals(loader, name, **BOUNDS).xs(USER)
        valid = intervals[intervals["interval_valid"]]
        assert not valid.empty
        assert valid["end_time"].gt(valid["start_time"]).all()
    assert loader.load_sleep_summary(USER, "2024-03-31", START).shape[0] == (
        0 if source == "labfront" else 1
    )


def test_heart_rate_hourly_reductions_match_physical_samples(wearable):
    source, loader = wearable
    if source == "labfront":
        raw = _csv("garmin-connect-daily-heart-rate").copy()
        raw["utc"] = pd.to_datetime(raw["unixTimestampInMs"], unit="ms")
        raw["value"] = raw["beatsPerMinute"]
    else:
        raw = pd.DataFrame(_json("heartrate"))
        raw["utc"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_localize(None)
        raw["value"] = raw["bpm"]
    raw = raw[raw["utc"].ge("2024-04-11") & raw["utc"].lt("2024-04-12") & raw["value"].gt(0)]
    conflicts = raw.groupby("utc")["value"].transform("nunique").gt(1)
    raw = raw[~conflicts].drop_duplicates("utc")
    grouped = raw.groupby(raw["utc"].dt.floor("1h"))["value"]
    result = heart_rate.get_heart_rate_statistics(
        loader, statistic=list(heart_rate.HEART_RATE_REDUCTIONS) + ["heart_rate_p10", "heart_rate_p90"],
        user_id=USER, start_date="2024-04-11", end_date="2024-04-12",
        resolution="1h", time_reference="utc",
    ).xs(USER)
    for feature, operation in heart_rate.HEART_RATE_REDUCTIONS.items():
        expected = getattr(grouped, operation)()
        np.testing.assert_allclose(result.loc[expected.index, feature], expected, equal_nan=True)
    for quantile in (10, 90):
        expected = grouped.quantile(quantile / 100)
        np.testing.assert_allclose(result.loc[expected.index, f"heart_rate_p{quantile}"], expected)
    _assert_clock(result.reset_index())


def test_activity_totals_match_native_steps(wearable):
    source, loader = wearable
    result = activity.get_activity_statistics(loader, statistic="activity_steps", **BOUNDS).xs(USER)
    if source == "oura":
        raw = _april(pd.DataFrame(_json("daily_activity")), "day")
        expected = raw.set_index(pd.to_datetime(raw["day"]))["steps"].reindex(DATES)
        assert expected.notna().sum() == 29
        assert pd.isna(result.loc[pd.Timestamp("2024-04-30"), "activity_steps"])
    else:
        raw = _csv("garmin-connect-epoch").copy()
        raw["local"] = pd.to_datetime(raw["isoDate"].str[:19])
        raw = _april(raw, "local")
        raw.loc[raw["steps"].lt(0), "steps"] = np.nan
        identity = ["unixTimestampInMs", "intensity", "activityType"]
        raw = raw.drop_duplicates(identity + ["steps", "durationInMs"])
        versions = raw.groupby(identity)[["steps", "durationInMs"]].nunique()
        conflicts = versions.index[versions.gt(1).any(axis=1)].get_level_values(0)
        epochs = raw.groupby("unixTimestampInMs").agg(local=("local", "first"), steps=("steps", "sum"))
        epochs.loc[epochs.index.isin(conflicts), "steps"] = np.nan
        expected = epochs.groupby(epochs["local"].dt.normalize())["steps"].agg(lambda v: v.sum(skipna=False))
        expected = expected.reindex(DATES)
        assert len(loader.load_activity_samples(USER, START, END)) == 2751
        peaks = activity.get_activity_statistics(loader, **BOUNDS).xs(USER)
        valid = peaks[["activity_steps", "activity_peak_60min_steps"]].dropna()
        assert not valid.empty
        assert valid["activity_peak_60min_steps"].le(valid["activity_steps"]).all()
    np.testing.assert_allclose(result["activity_steps"], expected, equal_nan=True)


def test_respiration_matches_native_rates(wearable):
    source, loader = wearable
    if source == "oura":
        native = _raw_nights(source)
        expected = native["average_breath"].reindex(DATES)
        result = respiration.get_respiration_statistics(
            loader, statistic="respiratory_rate_mean", resolution="nightly", **BOUNDS
        ).xs(USER)
        assert expected.notna().sum() == 25
        np.testing.assert_allclose(result["respiratory_rate_mean"], expected, equal_nan=True)
    else:
        raw = _csv("garmin-connect-respiration").copy()
        raw["local"] = pd.to_datetime(raw["isoDate"].str[:19])
        raw = _april(raw, "local").drop_duplicates("unixTimestampInMs")
        raw = raw[raw["breathsPerMinute"].gt(0)]
        grouped = raw.groupby(raw["local"].dt.normalize())["breathsPerMinute"]
        result = respiration.get_respiration_statistics(
            loader, statistic=list(respiration.RESPIRATORY_RATE_STATISTICS), resolution="daily", **BOUNDS
        ).xs(USER)
        for suffix, expected in [("mean", grouped.mean()), ("median", grouped.median()),
                                 ("std", grouped.std()), ("count", grouped.count()),
                                 ("p10", grouped.quantile(.1)), ("p90", grouped.quantile(.9))]:
            np.testing.assert_allclose(result.loc[expected.index, f"respiratory_rate_{suffix}"], expected)


def test_absent_spo2_is_not_invented(wearable):
    _, loader = wearable
    assert loader.load_pulse_ox(USER, START, END).empty
    assert loader.load_spo2_summary(USER, START, END).empty
    result = respiration.get_respiration_statistics(
        loader, statistic="spo2_mean", resolution="nightly", **BOUNDS
    )
    assert result["spo2_mean"].isna().all()
    assert not result["spo2_mean__coverage"].gt(0).any()


def test_provider_features_match_native_scores_and_clean_sentinels(wearable):
    source, loader = wearable
    available = provider_features.get_available_provider_features(loader)
    if source == "oura":
        result = provider_features.get_provider_features(loader, feature=list(loader.provider_features), **BOUNDS).xs(USER)
        raw = _april(pd.json_normalize(_json("daily_readiness")), "day")
        raw = raw.set_index(pd.to_datetime(raw["day"]))
        for feature in loader.provider_features:
            suffix = feature.removeprefix("oura_readiness_")
            field = suffix if suffix.startswith("temperature") or suffix == "score" else f"contributors.{suffix}"
            np.testing.assert_allclose(result[feature], raw[field].reindex(DATES), equal_nan=True)
        assert set(loader.provider_features).issubset(available)
        assert len(loader.load_resilience(USER, START, END)) == 22
    else:
        raw = _csv("garmin-connect-stress").copy()
        raw["local"] = pd.to_datetime(raw["isoDate"].str[:19])
        raw = _april(raw, "local").drop_duplicates("unixTimestampInMs")
        for feature, field in [("garmin_stress_level", "stressLevel"), ("garmin_body_battery", "bodyBattery")]:
            assert feature in available
            expected_rows = raw.loc[raw[field].between(0, 100)]
            expected = expected_rows.groupby(expected_rows["local"].dt.normalize())[field].mean()
            result = provider_features.get_provider_features(
                loader, feature=feature, statistic="mean", resolution="daily", **BOUNDS
            ).xs(USER)
            np.testing.assert_allclose(result[f"{feature}_mean"], expected.reindex(DATES), equal_nan=True)
        raw = _april(_csv("garmin-connect-daily-summary"), "calendarDate")
        result = provider_features.get_provider_features(
            loader, feature="garmin_daily_stress_duration_seconds", **BOUNDS
        ).xs(USER)
        for day, versions in raw.groupby("calendarDate"):
            seconds = result.loc[pd.Timestamp(day), "garmin_daily_stress_duration_seconds"]
            assert versions["unixTimestampInMs"].nunique() == 1
            complete = versions[versions["durationInMs"].eq(versions["durationInMs"].max())]
            candidates = complete["stressDurationInMs"].drop_duplicates()
            assert len(candidates) == 1
            expected = candidates.iloc[0] / 1000
            assert pd.isna(expected) and pd.isna(seconds) or seconds == expected


def test_body_battery_sleep_recovery_matches_observed_boundaries(labfront):
    features = ["garmin_body_battery_bedtime", "garmin_body_battery_getup",
                "garmin_body_battery_sleep_gain", "garmin_body_battery_sleep_recovery_fraction"]
    result = provider_features.get_provider_features(labfront, feature=features, **BOUNDS).xs(USER)
    raw = _csv("garmin-connect-stress").copy()
    raw["local"] = pd.to_datetime(raw["isoDate"].str[:19])
    raw = raw[raw["bodyBattery"].between(0, 100)].sort_values("local")
    nights = _raw_nights("labfront")
    compared = 0
    for day, night in nights.iterrows():
        start = pd.Timestamp(night["isoDate"]).tz_localize(None)
        end = start + pd.Timedelta(milliseconds=night["durationInMs"] + night["awakeDurationInMs"])
        before = raw[raw["local"].between(start - pd.Timedelta(minutes=5), start)]
        after = raw[raw["local"].between(end, end + pd.Timedelta(minutes=5))]
        if before.empty or after.empty:
            assert pd.isna(result.loc[day, "garmin_body_battery_sleep_gain"])
            continue
        bedtime, getup = before.iloc[-1]["bodyBattery"], after.iloc[0]["bodyBattery"]
        assert result.loc[day, "garmin_body_battery_bedtime"] == bedtime
        assert result.loc[day, "garmin_body_battery_getup"] == getup
        assert result.loc[day, "garmin_body_battery_sleep_gain"] == getup - bedtime
        if bedtime < 100:
            assert result.loc[day, "garmin_body_battery_sleep_recovery_fraction"] == pytest.approx(
                max((getup - bedtime) / (100 - bedtime), 0)
            )
        compared += 1
    assert compared > 10


def test_daily_stress_revision_retains_full_export_reference(labfront):
    # The original full export resolves this day to its complete daily summary.
    # Removing unrelated months must not change the selected measurement.
    raw = _csv("garmin-connect-daily-summary")
    complete = raw[raw["calendarDate"].eq("2024-04-12") & raw["durationInMs"].eq(86_400_000)]
    expected = complete["stressDurationInMs"].drop_duplicates().to_numpy() / 1000
    np.testing.assert_array_equal(expected, [21_120])
    result = provider_features.get_provider_features(
        labfront, feature="garmin_daily_stress_duration_seconds", **BOUNDS
    )
    assert result.loc[(USER, pd.Timestamp("2024-04-12")), "garmin_daily_stress_duration_seconds"] == expected[0]


def test_adherence_matches_observed_nights_and_heart_rate_minutes(wearable):
    source, loader = wearable
    result = adherence.get_adherence(loader, metric="sleep", **BOUNDS).xs(USER)
    expected = DATES.isin(_raw_nights(source).index).astype(float)
    np.testing.assert_array_equal(result["sleep_adherence"], expected)
    samples = loader.load_heart_rate(USER, START, END)
    minutes = samples["local_time"].dropna().dt.floor("1min").drop_duplicates()
    counts = minutes.groupby(minutes.dt.normalize()).size().reindex(DATES, fill_value=0)
    wearing = adherence.get_adherence(loader, metric="heart_rate", time_reference="local", **BOUNDS).xs(USER)
    np.testing.assert_allclose(wearing["heart_rate_adherence"], counts / 1440)


def test_hrv_uses_native_source_and_keeps_missing_features_missing(wearable):
    source, loader = wearable
    if source == "oura":
        raw = []
        for night in _json("sleep"):
            if night["type"] != "long_sleep" or not night.get("hrv"):
                continue
            series = night["hrv"]
            start = pd.Timestamp(series["timestamp"]).tz_convert("UTC").tz_localize(None)
            raw.extend((start + pd.Timedelta(seconds=i * series["interval"]), value)
                       for i, value in enumerate(series["items"]) if value is not None)
        raw = pd.DataFrame(raw, columns=["utc", "rmssd"])
        raw = _april(raw, "utc")
        expected = raw.groupby(raw["utc"].dt.floor("5min"))["rmssd"].mean()
        result = hrv.get_hrv_statistics(
            loader, statistic=["hrv_rmssd", "hrv_sdnn"], resolution="5min", time_reference="utc", **BOUNDS
        ).xs(USER)
        np.testing.assert_allclose(result.loc[expected.index, "hrv_rmssd"], expected)
        assert result["hrv_sdnn"].isna().all()
        with pytest.raises(ValueError, match="provider-computed"):
            hrv.get_hrv_statistics(loader, statistic="hrv_rmssd", window_step="1min", **BOUNDS)
    else:
        samples = loader.load_bbi(USER)
        assert len(samples) == 76_077
        assert samples["local_time"].dt.normalize().unique().tolist() == [pd.Timestamp("2024-04-11")]
        assert loader.load_bbi(USER, "2024-04-10", "2024-04-11").empty
        assert loader.load_bbi(USER, "2024-04-12", "2024-04-13").empty
        result = hrv.get_hrv_statistics(
            loader, resolution="5min", user_id=USER, start_date="2024-04-11",
            end_date="2024-04-12", min_coverage=.7, time_reference="local",
        ).xs(USER)
        assert len(result) == 288
        assert result["hrv_rmssd"].notna().sum() == 209
        assert set(hrv.HRV_STATISTICS).issubset(result)
        assert result[list(hrv.HRV_FREQUENCY_DOMAIN_STATISTICS)].notna().any().all()
        _assert_clock(result.reset_index())


def test_raw_bbi_metrics_and_loader_side_gap_do_not_bridge_differences(labfront):
    raw = _csv("garmin-device-bbi").copy()
    raw["utc_time"] = pd.to_datetime(raw["unixTimestampInMs"], unit="ms")
    raw["local_time"] = pd.to_datetime(raw["isoDate"].str[:-6], format="ISO8601")
    # Use a genuinely contiguous real run, not a presumed gap-free whole window.
    candidates = raw.groupby(raw["local_time"].dt.floor("5min"))
    reference = None
    for label, rows in candidates:
        rows = rows.sort_values("utc_time").drop_duplicates(["utc_time", "bbi"]).reset_index(drop=True)
        values = rows["bbi"].to_numpy(dtype=float)
        elapsed = rows["utc_time"].diff().dt.total_seconds().to_numpy() * 1000
        previous = np.r_[np.nan, values[:-1]]
        error = np.minimum(np.abs(elapsed - values), np.abs(elapsed - previous))
        breaks = ~((elapsed > 0) & (error < 100) & (values > 0))
        groups = rows.groupby(np.cumsum(breaks))
        reference = next((run for _, run in groups if len(run) > 30), None)
        if reference is not None:
            rows = reference.reset_index(drop=True)
            values = rows["bbi"].to_numpy(dtype=float)
            break
    else:
        pytest.fail("The real BBI fixture lost its continuous reference window")
    result = hrv.compute_hrv_statistics(
        rows, statistics=list(hrv.HRV_STATISTICS), coverage=None, time_col="local_time"
    ).loc[label]
    differences = np.diff(values)
    rmssd, sdnn, mean = np.sqrt(np.mean(differences ** 2)), np.std(values, ddof=1), values.mean()
    expected = {
        "hrv_rmssd": rmssd, "hrv_sdnn": sdnn,
        "hrv_pnn20": 100 * np.mean(np.abs(differences) > 20),
        "hrv_pnn50": 100 * np.mean(np.abs(differences) > 50),
        "hrv_min_nn": values.min(), "hrv_mean_nn": mean, "hrv_max_nn": values.max(),
        "hrv_bbi_median": np.median(values), "hrv_iqr_nn": np.quantile(values, .75) - np.quantile(values, .25),
        "hrv_sdnn_rmssd_ratio": sdnn / rmssd, "hrv_cv_nn": sdnn / mean, "hrv_cv_sd": rmssd / mean,
    }
    for name, value in expected.items():
        assert result[name] == pytest.approx(value)
    assert result["hrv_lfhf_ratio"] == pytest.approx(result["hrv_lf_power"] / result["hrv_hf_power"])
    position = len(rows) // 2
    filtered = rows.drop(index=position)
    result = hrv.compute_hrv_statistics(
        filtered, statistics=["hrv_rmssd", "hrv_pnn50"], coverage=None, time_col="local_time"
    ).loc[label]
    original_positions = filtered.index.to_numpy()
    valid_differences = np.diff(filtered["bbi"])[np.diff(original_positions) == 1]
    assert result["hrv_rmssd"] == pytest.approx(np.sqrt(np.mean(valid_differences ** 2)))
    assert result["hrv_pnn50"] == pytest.approx(100 * np.mean(np.abs(valid_differences) > 50))


def test_raw_hrv_overlap_and_sleep_filtering(labfront):
    loader = labfront
    kwargs = dict(user_id=USER, start_date="2024-04-11", end_date="2024-04-12",
                  statistic="hrv_rmssd", time_reference="local", min_coverage=.7)
    ordinary = hrv.get_hrv_statistics(loader, resolution="5min", **kwargs)
    rolling = hrv.get_hrv_statistics(loader, resolution="5min", window_step="1min", **kwargs)
    assert len(rolling) > len(ordinary)
    assert rolling["hrv_rmssd"].notna().sum() > ordinary["hrv_rmssd"].notna().sum()
    filtered = hrv.get_hrv_statistics(
        loader, resolution=("5min", "nightly"), window_step="1min", aggregation="mean", **kwargs
    )
    explicit = hrv.get_hrv_statistics(
        loader, resolution=("5min", "cycle_night"), window_step="1min", aggregation="mean",
        sleep_stages_to_remove=["awake", "unmeasurable"], **kwargs,
    )
    pd.testing.assert_frame_equal(filtered, explicit)
    unfiltered = hrv.get_hrv_statistics(
        loader, resolution=("5min", "nightly"), window_step="1min", aggregation="mean",
        sleep_stages_to_remove=None, **kwargs,
    )
    feature = "hrv_rmssd__5min__cycle_night_mean"
    assert filtered[feature].notna().any()
    assert unfiltered[feature].notna().any()
    assert not np.allclose(filtered[feature], unfiltered[feature], equal_nan=True)


def test_optional_neurokit_matches_direct_backend_on_real_timestamps(labfront):
    loader = labfront
    nk = pytest.importorskip("neurokit2")
    kwargs = dict(user_id=USER, start_date="2024-04-11", end_date="2024-04-12",
                  statistic=["hrv_rmssd", "hrv_sdnn", "hrv_mean_nn"], resolution="5min", min_coverage=.7)
    builtin = hrv.get_hrv_statistics(loader, backend="pywearable", **kwargs)
    neurokit = hrv.get_hrv_statistics(loader, backend="neurokit", **kwargs)
    for feature in ["hrv_sdnn", "hrv_mean_nn"]:
        paired = pd.concat([builtin[feature], neurokit[feature]], axis=1).dropna()
        assert len(paired) > 100
        np.testing.assert_allclose(paired.iloc[:, 0], paired.iloc[:, 1], rtol=1e-8)
    # Successive-difference metrics can differ because each backend applies
    # its own timestamp-gap policy. NeuroKit must still match NeuroKit itself.
    samples = loader.load_bbi(USER, "2024-04-10", "2024-04-13")
    samples = samples.drop_duplicates(["utc_time", "bbi"]).sort_values("utc_time")
    valid = neurokit["hrv_rmssd"].dropna()
    assert len(valid) > 100
    for _, label in valid.index[:3]:
        rows = samples[samples["utc_time"].ge(label) & samples["utc_time"].lt(label + pd.Timedelta(minutes=5))]
        intervals = {
            "RRI": rows["bbi"].to_numpy(),
            "RRI_Time": (rows["utc_time"] - rows["utc_time"].iloc[0]).dt.total_seconds().to_numpy(),
        }
        expected = nk.hrv_time(intervals, show=False).iloc[0]["HRV_RMSSD"]
        assert neurokit.loc[(USER, label), "hrv_rmssd"] == pytest.approx(expected)


def test_trajectory_history_and_visualizations_on_real_outputs(wearable):
    _, loader = wearable
    nightly = sleep.get_sleep_statistics(loader, statistic="sleep_TST", **BOUNDS)
    original = nightly.copy(deep=True)
    filled = trajectory.impute_trajectory(nightly, "sleep_TST", method="ma", window=7, min_periods=3)
    observed = nightly["sleep_TST"].notna()
    pd.testing.assert_series_equal(filled.loc[observed, "sleep_TST"], nightly.loc[observed, "sleep_TST"])
    pd.testing.assert_series_equal(filled["sleep_TST__coverage"], nightly["sleep_TST__coverage"])
    assert filled["sleep_TST__is_imputed"].any()
    enriched = trajectory.compute_trajectory_trend(
        filled, "sleep_TST", baseline_window=7, min_periods_baseline=3,
        normal_range_window=30, min_periods_normal_range=10,
    )
    expected_baseline = nightly["sleep_TST"].shift().rolling(7, min_periods=3).mean()
    np.testing.assert_allclose(enriched["sleep_TST__trend_baseline"], expected_baseline, equal_nan=True)
    try:
        trend = visualization.plot_trajectory_trend(enriched, "sleep_TST", show_coverage=True, show=False)
        assert set(trend) == {"trajectory", "coverage"}
        hypnogram = visualization.plot_sleep_summary(loader, USER, "2024-04-10", "2024-04-17", show=False)
        assert len(hypnogram["sleep"].collections) > 0
        night_hr = visualization.plot_sleep_heart_rate(loader, USER, "2024-04-11", show=False)
        assert len(night_hr.lines) > 0
        for number in plt.get_fignums():
            plt.figure(number).canvas.draw()
    finally:
        plt.close("all")
    pd.testing.assert_frame_equal(nightly, original)
