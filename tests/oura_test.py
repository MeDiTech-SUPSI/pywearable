import json

import pandas as pd
import pytest

from pywearable import heart_rate, provider_features, sleep
from pywearable.loader import OuraLoader


def _page(root, metric, name, records):
    folder = root / "person-1" / metric
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps({"data": records, "next_token": None}), encoding="utf-8")


def test_oura_reader_combines_pages_and_deduplicates_identical_records(tmp_path):
    first = {"id": "sleep-1", "day": "2024-01-02", "type": "long_sleep"}
    second = {"id": "sleep-2", "day": "2024-01-03", "type": "long_sleep"}
    _page(tmp_path, "sleep", "data_0.json", [first])
    _page(tmp_path, "sleep", "data_1.json", [first, second])
    loader = OuraLoader(tmp_path)
    assert loader.get_user_ids() == ["person-1"]
    assert len(loader._read_records("person-1", "sleep")) == 2
    assert loader._day_filter(loader._read_records("person-1", "sleep"), "2024-01-03", "2024-01-04") == [second]


def test_oura_reader_rejects_conflicting_export_snapshots(tmp_path):
    _page(tmp_path, "sleep", "data_0.json", [{"id": "sleep-1", "day": "2024-01-02"}])
    _page(tmp_path, "sleep", "data_1.json", [{"id": "sleep-1", "day": "2024-01-03"}])
    with pytest.raises(ValueError, match="Conflicting"):
        OuraLoader(tmp_path)._read_records("person-1", "sleep")


def test_oura_reader_skips_numbered_placeholder_pages(tmp_path):
    folder = tmp_path / "person-1" / "heartrate"
    folder.mkdir(parents=True)
    (folder / "data_0.json").write_text('"0"', encoding="utf-8")
    _page(tmp_path, "heartrate", "data_1.json", [{"timestamp": "2024-01-01T00:00:00+00:00", "bpm": 60}])
    assert len(OuraLoader(tmp_path)._read_records("person-1", "heartrate")) == 1


def test_missing_oura_metric_returns_typed_empty_tables(tmp_path):
    _page(tmp_path, "daily_activity", "data.json", [])
    loader = OuraLoader(tmp_path)
    stages = loader.load_sleep_stage("person-1")
    assert stages.empty
    assert {"sleep_summary_id", "utc_time", "local_time_uncertain"} <= set(stages)
    assert loader.load_heart_rate("person-1").empty


def test_subdaily_time_uses_embedded_offset_not_local_clock():
    assert OuraLoader._utc_time("2024-01-01T01:00:00+01:00") == pd.Timestamp("2024-01-01 00:00:00")
    with pytest.raises(ValueError, match="lacks an offset"):
        OuraLoader._utc_time("2024-01-01T01:00:00")


def _sleep_record(session_id="sleep-1", day="2024-01-02", sleep_type="long_sleep"):
    return {
        "id": session_id,
        "day": day,
        "type": sleep_type,
        "bedtime_start": "2024-01-01T23:00:00+01:00",
        "bedtime_end": "2024-01-02T00:20:00+01:00",
        "time_in_bed": 4800,
        "total_sleep_duration": 3900,
        "awake_time": 900,
        "deep_sleep_duration": 1200,
        "light_sleep_duration": 1800,
        "rem_sleep_duration": 900,
        "sleep_phase_5_min": "2211433333333333",
    }


def test_oura_sleep_maps_long_sleep_and_keeps_naps_separate(tmp_path):
    _page(tmp_path, "sleep", "data.json", [_sleep_record(), _sleep_record("nap-1", sleep_type="sleep")])
    loader = OuraLoader(tmp_path)
    summaries = loader.load_sleep_summary("person-1", "2024-01-02", "2024-01-03")
    assert len(summaries) == 1
    assert summaries.loc[0, "sleep_summary_id"] == "sleep-1"
    assert summaries.loc[0, "duration_in_ms"] == 3_900_000
    assert summaries.loc[0, "utc_time"] == pd.Timestamp("2024-01-01 22:00")
    assert summaries.loc[0, "local_time"] == pd.Timestamp("2024-01-01 23:00")
    assert pd.isna(summaries.loc[0, "overall_sleep_score"])
    assert len(loader.load_sleep_sessions("person-1")) == 2
    assert loader.load_sleep_sessions("person-1")["main_sleep"].tolist() == [True, False]
    assert len(loader.load_sleep_sessions("person-1", sleep_type="sleep")) == 1
    assert set(loader.load_sleep_stage("person-1")["sleep_summary_id"]) == {"sleep-1"}
    assert set(loader.load_sleep_session_stages("person-1")["sleep_summary_id"]) == {
        "sleep-1",
        "nap-1",
    }
    assert loader.load_sleep_summary("person-1", "2024-01-03", "2024-01-04").empty


def test_default_oura_sleep_extraction_returns_shared_features(tmp_path):
    _page(tmp_path, "sleep", "data.json", [_sleep_record()])
    result = sleep.get_sleep_statistics(
        OuraLoader(tmp_path), user_id="person-1", start_date=None, end_date=None
    )
    assert result.shape == (1, 39)
    assert set(result) == {
        "sleep_TIB", "sleep_TST", "sleep_SE", "sleep_SME", "sleep_SPT", "sleep_WASO", "sleep_SOL", "sleep_deep", "sleep_light",
        "sleep_REM", "sleep_NREM", "sleep_awake", "sleep_unmeasurable", "sleep_%_deep", "sleep_%_light",
        "sleep_%_REM", "sleep_%_NREM", "sleep_lat_deep", "sleep_lat_light", "sleep_lat_REM",
        "sleep_score", "sleep_count_awake", "sleep_count_deep", "sleep_count_light", "sleep_count_REM",
        "sleep_awakening_count", "sleep_awakening_index", "sleep_median_awakening_duration",
        "sleep_stage_transition_count", "sleep_stage_transition_index", "sleep_stage_coverage",
        "sleep_bedtime", "sleep_sleep_onset", "sleep_wakeup", "sleep_getup", "sleep_getup_latency",
        "sleep_midpoint", "sleep_cpd_midpoint", "sleep_cpd_duration",
    }
    assert not {"sleep_validation", "sleep_SRI"} & set(result)
    night = result.iloc[0]
    assert night["sleep_TIB"] == 80
    assert night["sleep_TST"] == 65
    assert night["sleep_SPT"] == 80
    assert night["sleep_WASO"] == 5
    assert night["sleep_SOL"] == 0
    assert night["sleep_SME"] == 93.75
    assert night["sleep_lat_deep"] == 10
    assert night["sleep_lat_light"] == 0
    assert night["sleep_lat_REM"] == 25
    assert night["sleep_count_awake"] == 1
    assert night["sleep_count_deep"] == 1
    assert night["sleep_count_light"] == 1
    assert night["sleep_count_REM"] == 1
    assert night["sleep_bedtime"] == pd.Timestamp("2024-01-01 23:00")
    assert night["sleep_sleep_onset"] == pd.Timestamp("2024-01-01 23:00")
    assert night["sleep_wakeup"] == pd.Timestamp("2024-01-02 00:20")
    assert night["sleep_getup"] == pd.Timestamp("2024-01-02 00:20")
    assert night["sleep_getup_latency"] == 0
    assert night["sleep_midpoint"] == pd.Timestamp("2024-01-01 23:40")
    assert night["sleep_awakening_count"] == 1
    assert night["sleep_awakening_index"] == pytest.approx(60 / 65)
    assert night["sleep_median_awakening_duration"] == 5
    assert night["sleep_stage_transition_count"] == 1
    assert night["sleep_stage_transition_index"] == pytest.approx(60 / 65)
    assert night["sleep_stage_coverage"] == 100


def test_oura_stages_decode_runs_and_filter_by_local_instant(tmp_path):
    record = _sleep_record()
    record["sleep_phase_5_min"] = "2214"
    _page(tmp_path, "sleep", "data.json", [record])
    stages = OuraLoader(tmp_path).load_sleep_stage("person-1")
    assert list(stages["type"]) == ["light", "deep", "awake"]
    assert list(stages["duration_in_ms"]) == [600_000, 300_000, 300_000]
    assert stages.loc[0, "utc_time"] == pd.Timestamp("2024-01-01 22:00")
    filtered = OuraLoader(tmp_path).load_sleep_stage("person-1", "2024-01-01 23:10", "2024-01-02 00:00")
    assert list(filtered["type"]) == ["deep", "awake"]


def test_oura_hr_preserves_source_conflict_but_excludes_it_from_aggregation(tmp_path):
    _page(
        tmp_path, "heartrate", "data.json",
        [
            {"timestamp": "2024-01-01T01:00:00+01:00", "bpm": 80, "source": "awake"},
            {"timestamp": "2024-01-01T00:00:00+00:00", "bpm": 78, "source": "live"},
            {"timestamp": "2024-01-01T00:30:00+00:00", "bpm": 60, "source": "rest"},
        ],
    )
    _page(
        tmp_path,
        "daily_activity",
        "data.json",
        [{"id": "activity-1", "day": "2024-01-01", "timestamp": "2024-01-01T04:00:00+01:00"}],
    )
    loader = OuraLoader(tmp_path)
    samples = loader.load_heart_rate("person-1")
    assert len(samples) == 3
    assert samples["timestamp_conflict"].sum() == 2
    local_samples = loader.load_heart_rate("person-1", time_col="local_time")
    assert local_samples.loc[0, "local_time"] == pd.Timestamp("2024-01-01 01:00")
    assert local_samples.loc[0, "timezone"] == pd.Timedelta(hours=1)
    assert set(local_samples["local_time_source"]) == {"daily_activity"}
    with pytest.warns(UserWarning, match="ambiguous"):
        trajectory = heart_rate.get_heart_rate_statistics(
            loader, user_id="person-1", resolution="1h",
            start_date="2024-01-01 00:00", end_date="2024-01-01 01:00",
            statistic=["heart_rate_mean", "heart_rate_count"],
        )
    assert trajectory.iloc[0]["heart_rate_mean"] == 60
    assert trajectory.iloc[0]["heart_rate_count"] == 1


def test_oura_hr_does_not_fill_days_without_an_offset_anchor(tmp_path):
    _page(
        tmp_path,
        "heartrate",
        "data.json",
        [
            {"timestamp": "2024-01-01T12:00:00+00:00", "bpm": 60},
            {"timestamp": "2024-01-03T12:00:00+00:00", "bpm": 70},
        ],
    )
    _page(
        tmp_path,
        "daily_activity",
        "data.json",
        [{"id": "activity-1", "day": "2024-01-01", "timestamp": "2024-01-01T04:00:00+01:00"}],
    )
    with pytest.warns(UserWarning, match="no unambiguous local offset"):
        samples = OuraLoader(tmp_path).load_heart_rate(
            "person-1", time_col="local_time"
        )
    assert samples["local_time"].notna().sum() == 1
    assert samples.loc[1, "local_time_uncertain"]


def test_oura_hr_uses_exact_event_offset_before_daily_activity(tmp_path):
    _page(
        tmp_path,
        "heartrate",
        "data.json",
        [{"timestamp": "2024-01-01T22:30:00+00:00", "bpm": 60}],
    )
    _page(
        tmp_path,
        "daily_activity",
        "data.json",
        [{"id": "activity-1", "day": "2024-01-02", "timestamp": "2024-01-02T04:00:00+01:00"}],
    )
    record = _sleep_record()
    record["bedtime_start"] = "2024-01-01T23:00:00+02:00"
    record["bedtime_end"] = "2024-01-02T01:00:00+02:00"
    _page(tmp_path, "sleep", "data.json", [record])
    sample = OuraLoader(tmp_path).load_heart_rate("person-1").iloc[0]
    assert sample["local_time"] == pd.Timestamp("2024-01-02 00:30")
    assert sample["timezone"] == pd.Timedelta(hours=2)
    assert sample["local_time_source"] == "sleep"


def test_hourly_hr_uses_utc_hour_boundaries_and_requested_columns(tmp_path):
    _page(tmp_path, "heartrate", "data.json", [
        {"timestamp": "2024-01-01T00:17:00+00:00", "bpm": 60},
        {"timestamp": "2024-01-01T00:49:00+00:00", "bpm": 80},
    ])
    result = heart_rate.get_heart_rate_statistics(
        OuraLoader(tmp_path), user_id="person-1", resolution="1h",
        start_date="2024-01-01 00:17", end_date="2024-01-01 01:17",
        statistic=["heart_rate_mean", "heart_rate_max"],
    )
    assert list(result) == ["heart_rate_mean", "heart_rate_max"]
    assert result.index.get_level_values("utc_time")[0] == pd.Timestamp("2024-01-01 00:00")
    assert result.iloc[0].to_dict() == {"heart_rate_mean": 70.0, "heart_rate_max": 80.0}
    offset_aware = heart_rate.get_heart_rate_statistics(
        OuraLoader(tmp_path), user_id="person-1", resolution="1h",
        start_date="2024-01-01T01:17:00+01:00",
        end_date="2024-01-01T02:17:00+01:00",
        statistic=["heart_rate_mean", "heart_rate_max"],
    )
    pd.testing.assert_frame_equal(result, offset_aware)


def test_oura_sleep_stage_flags_unknown_local_clock_across_offset_change(tmp_path):
    record = _sleep_record()
    record["bedtime_start"] = "2024-03-31T01:55:00+01:00"
    record["bedtime_end"] = "2024-03-31T03:10:00+02:00"
    record["sleep_phase_5_min"] = "24"
    _page(tmp_path, "sleep", "data.json", [record])
    loader = OuraLoader(tmp_path)
    stage = loader.load_sleep_stage("person-1")
    assert stage.loc[0, "local_time"] == pd.Timestamp("2024-03-31 01:55")
    assert stage.loc[1, "utc_time"] == pd.Timestamp("2024-03-31 01:00")
    assert stage["local_time_uncertain"].all()
    assert list(stage["timezone_offset_in_ms"]) == [3_600_000, 3_600_000]
    summary = loader.load_sleep_summary("person-1")
    assert summary.loc[0, "local_end_time"] == pd.Timestamp("2024-03-31 03:10")
    assert pd.isna(sleep._compute_wakeup_time(summary, stage).loc[0])
    assert sleep._compute_getup_time(summary).loc[0] == pd.Timestamp(
        "2024-03-31 03:10"
    )
    assert sleep._compute_getup_latency(summary, stage).loc[0] == 10
    assert pd.isna(sleep._compute_sleep_midpoint(summary, stage).loc[0])


def test_offset_change_does_not_truncate_autumn_sleep_stages(tmp_path):
    record = _sleep_record(day="2024-10-27")
    record["bedtime_start"] = "2024-10-27T01:55:00+02:00"
    record["bedtime_end"] = "2024-10-27T02:10:00+01:00"
    record["sleep_phase_5_min"] = "2" * 8 + "4" + "2" * 6
    _page(tmp_path, "sleep", "data.json", [record])
    loader = OuraLoader(tmp_path)
    stages = loader.load_sleep_stage(
        "person-1", start_date="2024-10-27 01:55", end_date="2024-10-27 02:10"
    )
    assert list(stages["type"]) == ["light", "awake", "light"]
    assert stages["local_time_uncertain"].all()
    assert stages["duration_in_ms"].sum() == 75 * 60_000


def test_oura_vendor_hrv_and_respiration_keep_night_and_sample_grains(tmp_path):
    record = _sleep_record()
    record["average_hrv"] = 42
    record["average_breath"] = 14.5
    record["hrv"] = {"timestamp": "2024-01-01T23:00:00+01:00", "interval": 300.0, "items": [None, 40.0, 44.0]}
    _page(tmp_path, "sleep", "data.json", [record])
    loader = OuraLoader(tmp_path)
    night = loader.load_hrv_summary("person-1")
    samples = loader.load_hrv_samples("person-1")
    breaths = loader.load_sleep_respiration_summary("person-1")
    assert night.loc[0, "oura_average_rmssd_ms"] == 42
    assert list(samples["oura_rmssd_ms"]) == [40.0, 44.0]
    assert samples.loc[0, "utc_time"] == pd.Timestamp("2024-01-01 22:05")
    assert breaths.loc[0, "oura_average_breaths_per_minute"] == 14.5
    assert breaths.loc[0, "respiratory_rate_mean"] == 14.5


def test_oura_daily_provider_tables_use_day_and_keep_vendor_labels(tmp_path):
    _page(tmp_path, "daily_activity", "data.json", [{"id": "activity-1", "day": "2024-01-02", "steps": 8021}])
    _page(tmp_path, "daily_spo2", "data.json", [{"id": "spo2-1", "day": "2024-01-02", "spo2_percentage": {"average": 97.6}}])
    _page(tmp_path, "daily_readiness", "data.json", [{"id": "ready-1", "day": "2024-01-02", "score": 82, "contributors": {"hrv_balance": 71}}])
    _page(tmp_path, "daily_resilience", "data.json", [{"id": "resilient-1", "day": "2024-01-02", "level": "solid", "contributors": {"stress": 65}}])
    loader = OuraLoader(tmp_path)
    assert loader.load_steps_summary("person-1").loc[0, "steps"] == 8021
    assert loader.load_pulse_ox_summary("person-1").loc[0, "oura_daily_spo2_percentage"] == 97.6
    assert loader.load_spo2_summary("person-1").loc[0, "spo2_mean"] == 97.6
    assert loader.load_readiness("person-1").loc[0, "readiness_hrv_balance"] == 71
    assert provider_features.get_available_provider_features(loader) == sorted(
        loader.provider_features
    )
    readiness = provider_features.get_provider_features(
        loader,
        feature=[
            provider_features.OURA_READINESS_SCORE,
            provider_features.OURA_READINESS_HRV_BALANCE,
            provider_features.OURA_READINESS_SLEEP_BALANCE,
        ],
        user_id="person-1",
    )
    assert readiness.loc[
        ("person-1", pd.Timestamp("2024-01-02")),
        provider_features.OURA_READINESS_SCORE,
    ] == 82
    assert readiness.loc[
        ("person-1", pd.Timestamp("2024-01-02")),
        provider_features.OURA_READINESS_HRV_BALANCE,
    ] == 71
    assert pd.isna(
        readiness.loc[
            ("person-1", pd.Timestamp("2024-01-02")),
            provider_features.OURA_READINESS_SLEEP_BALANCE,
        ]
    )
    monthly_readiness = provider_features.get_provider_features(
        loader,
        feature=provider_features.OURA_READINESS_SCORE,
        user_id="person-1",
        resolution="monthly",
        aggregation="mean",
    )
    assert monthly_readiness[
        "oura_readiness_score__daily__monthly_mean"
    ].iloc[0] == 82
    assert loader.load_resilience("person-1").loc[0, "oura_resilience_level"] == "solid"
    assert loader.load_steps_summary("person-1", "2024-01-03", "2024-01-04").empty
    with pytest.raises(NotImplementedError, match="vendor-specific"):
        loader.load_recovery("person-1")


def test_oura_zero_steps_and_spo2_are_missing(tmp_path):
    _page(
        tmp_path,
        "daily_activity",
        "data.json",
        [{"id": "activity-1", "day": "2024-01-02", "steps": 0}],
    )
    _page(
        tmp_path,
        "daily_spo2",
        "data.json",
        [
            {
                "id": "spo2-1",
                "day": "2024-01-02",
                "spo2_percentage": {"average": 0.0},
            }
        ],
    )
    loader = OuraLoader(tmp_path)

    steps = loader.load_steps_summary("person-1")
    pulse_ox = loader.load_pulse_ox_summary("person-1")

    assert pd.isna(steps.loc[0, "steps"])
    assert str(steps["steps"].dtype) == "Int64"
    assert pd.isna(
        pulse_ox.loc[0, "oura_daily_spo2_percentage"]
    )
    assert str(pulse_ox["oura_daily_spo2_percentage"].dtype) == "Float64"
