from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pywearable import schema, trajectory, visualization


def _trajectory(values, *, feature="feature", frequency="D"):
    times = pd.date_range("2024-01-01", periods=len(values), freq=frequency)
    index = pd.MultiIndex.from_arrays(
        [["person-1"] * len(times), times],
        names=[schema.USER_COL, schema.CALENDAR_DATE_COL],
    )
    return pd.DataFrame(
        {
            feature: values,
            f"{feature}__coverage": [float(pd.notna(value)) for value in values],
        },
        index=index,
    )


def _sleep_loader():
    summary_rows = []
    stage_rows = []
    for offset in range(4):
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=offset)
        start = date - pd.Timedelta(hours=1) + pd.Timedelta(minutes=10 * offset)
        summary_id = f"night-{offset}"
        summary_rows.append(
            {
                schema.SLEEP_SUMMARY_ID_COL: summary_id,
                schema.CALENDAR_DATE_COL: date,
                schema.LOCAL_TIME_COL: start,
                schema.LOCAL_END_TIME_COL: start + pd.Timedelta(hours=7),
                schema.DURATION_IN_MS_COL: 6.5 * 3_600_000,
                schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL: 30 * 60_000,
                schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL: 90 * 60_000,
                schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL: 210 * 60_000,
                schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL: 90 * 60_000,
                schema.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL: 0,
                schema.SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL: 75 + offset,
            }
        )
        stage_start = start
        for stage_type, minutes in (
            (schema.SLEEP_STAGE_AWAKE_STAGE_VALUE, 15),
            (schema.SLEEP_STAGE_LIGHT_STAGE_VALUE, 180),
            (schema.SLEEP_STAGE_DEEP_STAGE_VALUE, 90),
            (schema.SLEEP_STAGE_REM_STAGE_VALUE, 120),
            (schema.SLEEP_STAGE_AWAKE_STAGE_VALUE, 15),
        ):
            stage_rows.append(
                {
                    schema.SLEEP_SUMMARY_ID_COL: summary_id,
                    schema.LOCAL_TIME_COL: stage_start,
                    schema.DURATION_IN_MS_COL: minutes * 60_000,
                    schema.SLEEP_STAGE_SLEEP_TYPE_COL: stage_type,
                }
            )
            stage_start += pd.Timedelta(minutes=minutes)

    summaries = pd.DataFrame(summary_rows)
    stages = pd.DataFrame(stage_rows)
    heart_times = pd.date_range(
        summaries.iloc[0][schema.LOCAL_TIME_COL],
        summaries.iloc[0][schema.LOCAL_END_TIME_COL],
        freq="1min",
        inclusive="left",
    )
    heart_rates = pd.DataFrame(
        {
            schema.UTC_TIME_COL: heart_times - pd.Timedelta(hours=1),
            schema.LOCAL_TIME_COL: heart_times,
            schema.TIMEZONE_COL: pd.Timedelta(hours=1),
            schema.HEART_RATE_COL: 58
            + 5 * np.sin(np.linspace(0, 4 * np.pi, len(heart_times))),
        }
    )
    activity_times = pd.date_range(
        "2024-01-01 12:00", "2024-01-09 12:00", freq="15min", inclusive="left"
    )
    activity_samples = pd.DataFrame(
        {
            schema.UTC_TIME_COL: activity_times - pd.Timedelta(hours=1),
            schema.LOCAL_TIME_COL: activity_times,
            schema.TIMEZONE_COL: pd.Timedelta(hours=1),
            schema.DURATION_IN_MS_COL: 15 * 60_000,
            schema.ACTIVITY_STEPS_COL: np.where(
                (activity_times.hour >= 8) & (activity_times.hour <= 21), 20, 0
            ),
        }
    )

    def load_sleep_summary(user_id, start_date=None, end_date=None, **kwargs):
        frame = summaries.copy()
        if start_date is not None:
            frame = frame[
                frame[schema.CALENDAR_DATE_COL] >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            frame = frame[
                frame[schema.CALENDAR_DATE_COL] < pd.Timestamp(end_date).normalize()
            ]
        return frame.reset_index(drop=True)

    def load_sleep_stage(user_id, start_date=None, end_date=None, **kwargs):
        return stages.copy()

    def load_heart_rate(
        user_id, start_date=None, end_date=None, time_col=schema.LOCAL_TIME_COL
    ):
        frame = heart_rates.copy()
        if start_date is not None:
            frame = frame[frame[time_col] >= pd.Timestamp(start_date)]
        if end_date is not None:
            frame = frame[frame[time_col] < pd.Timestamp(end_date)]
        return frame.reset_index(drop=True)

    def load_activity_samples(user_id, start_date=None, end_date=None, **kwargs):
        frame = activity_samples.copy()
        if start_date is not None:
            frame = frame[frame[schema.LOCAL_TIME_COL] >= pd.Timestamp(start_date)]
        if end_date is not None:
            frame = frame[frame[schema.LOCAL_TIME_COL] < pd.Timestamp(end_date)]
        return frame.reset_index(drop=True)

    return SimpleNamespace(
        activity_source="samples",
        activity_primary_resolution="15min",
        activity_time_columns=(schema.UTC_TIME_COL, schema.LOCAL_TIME_COL),
        heart_rate_time_columns=(schema.UTC_TIME_COL, schema.LOCAL_TIME_COL),
        get_user_ids=lambda: ["person-1"],
        load_sleep_summary=load_sleep_summary,
        load_sleep_stage=load_sleep_stage,
        load_heart_rate=load_heart_rate,
        load_activity_samples=load_activity_samples,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_trajectory_distinguishes_imputed_values_and_preserves_input(tmp_path):
    frame = _trajectory([0.0, np.nan, 20.0, 30.0, np.nan])
    frame.loc[frame.index[1], "feature"] = 10.0
    frame["feature__is_imputed"] = [False, True, False, False, False]
    before = frame.copy(deep=True)
    output = tmp_path / "trajectory.png"

    axes = visualization.plot_trajectory(
        frame,
        "feature",
        show_coverage=True,
        show=False,
        save_to=output,
    )

    assert set(axes) == {"trajectory", "coverage"}
    assert output.is_file()
    pd.testing.assert_frame_equal(frame, before)
    labels = axes["trajectory"].get_legend_handles_labels()[1]
    assert "feature (imputed)" in labels


def test_plot_trajectory_requires_user_for_multiuser_input():
    frame = _trajectory([1.0, 2.0])
    second = frame.copy()
    second.index = pd.MultiIndex.from_arrays(
        [["person-2"] * len(second), second.index.get_level_values(-1)],
        names=frame.index.names,
    )

    with pytest.raises(ValueError, match="user_id is required"):
        visualization.plot_trajectory(pd.concat([frame, second]), "feature", show=False)


def test_plot_trajectory_trend_handles_numeric_and_clock_features():
    numeric = _trajectory([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    numeric = trajectory.impute_trajectory(
        numeric, "feature", method="ma", window=3, min_periods=2
    )
    numeric = trajectory.compute_trajectory_trend(
        numeric,
        "feature",
        baseline_window=3,
        min_periods_baseline=2,
        normal_range_window=4,
        min_periods_normal_range=2,
    )
    axes = visualization.plot_trajectory_trend(
        numeric,
        "feature",
        show_deviation=True,
        show_coverage=True,
        show=False,
    )
    assert set(axes) == {"trajectory", "coverage"}

    clock = _trajectory(
        pd.to_datetime(
            [
                "2024-01-01 23:30",
                "2024-01-02 00:15",
                "2024-01-02 23:45",
                "2024-01-04 00:20",
                "2024-01-04 23:50",
                "2024-01-06 00:10",
            ]
        ),
        feature="sleep_bedtime",
    )
    clock = trajectory.compute_trajectory_trend(
        clock,
        "sleep_bedtime",
        baseline_window=3,
        min_periods_baseline=2,
        normal_range_window=3,
        min_periods_normal_range=2,
    )
    axis = visualization.plot_trajectory_trend(clock, "sleep_bedtime", show=False)
    assert axis.yaxis.get_major_formatter()(24, None) == "00:00"


def test_hierarchy_calendar_and_period_plots_accept_canonical_outputs(tmp_path):
    children = _trajectory([0.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    parent_index = pd.MultiIndex.from_arrays(
        [["person-1"], [pd.Timestamp("2024-01-01")]],
        names=[schema.USER_COL, schema.CALENDAR_DATE_COL],
    )
    parents = pd.DataFrame(
        {
            "period_start": [pd.Timestamp("2024-01-01")],
            "period_end": [pd.Timestamp("2024-01-08")],
            "weekly_mean": [4.0],
            "weekly_p10": [1.0],
            "weekly_p90": [7.0],
            "weekly_mean__coverage": [6 / 7],
        },
        index=parent_index,
    )

    hierarchy_axis = visualization.plot_hierarchical_trajectory(
        _sleep_loader(),
        "sleep_TST",
        ("nightly", "weekly"),
        user_id="person-1",
        start_date="2024-01-02",
        end_date="2024-01-06",
        show=False,
    )
    assert hierarchy_axis.collections

    activity_hierarchy_axis = visualization.plot_hierarchical_trajectory(
        _sleep_loader(),
        "activity_steps",
        ("15min", "daily"),
        user_id="person-1",
        start_date="2024-01-01",
        end_date="2024-01-03",
        statistics_kwargs={"time_reference": "local"},
        show=False,
    )
    assert activity_hierarchy_axis.collections

    calendar_axis = visualization.plot_calendar_heatmap(children, "feature", show=False)
    assert len(calendar_axis.patches) == 1
    assert calendar_axis.lines

    output = tmp_path / "period.png"
    period_axis = visualization.plot_period_comparison(
        parents,
        "weekly_mean",
        favorable_direction="higher",
        show=False,
        save_to=output,
    )
    assert period_axis.containers
    assert output.is_file()


def test_subdaily_coverage_and_weekly_profile_keep_expected_shape():
    times = pd.date_range("2024-01-01", periods=14 * 24, freq="1h")
    index = pd.MultiIndex.from_arrays(
        [["person-1"] * len(times), times],
        names=[schema.USER_COL, schema.LOCAL_TIME_COL],
    )
    frame = pd.DataFrame(
        {
            "activity_steps": 10 + np.arange(len(times)) % 24,
            "activity_steps__coverage": np.where(times.hour < 6, 0.5, 1.0),
        },
        index=index,
    )

    coverage_axis = visualization.plot_coverage_heatmap(
        frame, "activity_steps", show=False
    )
    profile_axes = visualization.plot_weekly_profile(
        frame,
        "activity_steps",
        loader=_sleep_loader(),
        aggregation="median",
        show=False,
    )

    assert coverage_axis.images[0].get_array().shape == (14, 24)
    assert coverage_axis.images[0].get_cmap().name == "RdYlGn"
    assert profile_axes.shape == (7,)
    assert all(axis.lines for axis in profile_axes)
    assert any(axis.patches for axis in profile_axes)


def test_weekday_sleep_windows_use_actual_local_days(monkeypatch):
    dates = pd.to_datetime(
        ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-17"]
    )
    index = pd.MultiIndex.from_arrays(
        [["person-1"] * len(dates), dates],
        names=[schema.USER_COL, schema.CALENDAR_DATE_COL],
    )
    nightly = pd.DataFrame(
        {
            "sleep_bedtime": pd.to_datetime(
                [
                    "2024-01-08 00:30",
                    "2024-01-08 23:00",
                    "2024-01-09 23:00",
                    "2024-01-17 00:30",
                ]
            ),
            "sleep_wakeup": pd.to_datetime(
                [
                    "2024-01-08 07:00",
                    "2024-01-09 07:00",
                    "2024-01-10 07:00",
                    "2024-01-17 07:00",
                ]
            ),
            "sleep_bedtime__coverage": [1.0] * len(dates),
            "sleep_wakeup__coverage": [1.0] * len(dates),
        },
        index=index,
    )
    monkeypatch.setattr(
        visualization.sleep,
        "get_sleep_statistics",
        lambda *args, **kwargs: nightly,
    )

    windows = visualization._weekday_sleep_windows(
        SimpleNamespace(), "person-1", dates.min(), dates.max()
    )

    assert windows["Monday"] == [(0.5, 7.0), (23.0, 24.0)]
    assert windows["Tuesday"] == [(0.0, 7.0), (23.75, 24.0)]
    assert windows["Wednesday"] == [(0.0, 7.0)]


def test_period_comparison_labels_only_consecutive_percent_changes():
    frame = _trajectory([100.0, 150.0, np.nan, 75.0])

    axis = visualization.plot_period_comparison(
        frame,
        "feature",
        favorable_direction="higher",
        show=False,
    )

    labels = [text.get_text() for text in axis.texts]
    assert labels == ["↑ 50.0%"]
    assert axis.get_legend() is None


def test_sleep_views_combine_loader_tables_without_expanding_stages(monkeypatch):
    loader = _sleep_loader()
    summary_axes = visualization.plot_sleep_summary(
        loader,
        "person-1",
        "2024-01-02",
        "2024-01-06",
        cpd="duration",
        show=False,
    )
    assert set(summary_axes) == {"sleep", "sleep_cpd_duration"}
    assert summary_axes["sleep"].collections

    boundary_index = pd.MultiIndex.from_arrays(
        [["person-1"], [pd.Timestamp("2024-01-02")]],
        names=[schema.USER_COL, schema.CALENDAR_DATE_COL],
    )
    boundaries = pd.DataFrame(
        {
            "sleep_bedtime": [pd.Timestamp("2024-01-01 23:00")],
            "sleep_sleep_onset": [pd.Timestamp("2024-01-01 23:00")],
            "sleep_wakeup": [pd.Timestamp("2024-01-02 06:00")],
            "sleep_getup": [pd.Timestamp("2024-01-02 06:00")],
        },
        index=boundary_index,
    )
    monkeypatch.setattr(
        visualization.sleep,
        "get_sleep_statistics",
        lambda *args, **kwargs: boundaries,
    )

    heart_axes = visualization.plot_sleep_heart_rate(
        loader,
        "person-1",
        "2024-01-02",
        show_coverage=True,
        show=False,
    )
    assert set(heart_axes) == {"heart_rate", "coverage"}
    assert heart_axes["heart_rate"].patches
    labels = heart_axes["heart_rate"].get_legend_handles_labels()[1]
    assert "Bedtime" in labels
    assert "Wake-up" in labels
    assert "Sleep onset" not in labels
    assert "Get-up" not in labels
    assert {patch.get_alpha() for patch in heart_axes["heart_rate"].patches} == {
        0.175,
        0.45,
    }


def test_weekly_overview_combines_activity_heart_rate_sleep_and_missingness():
    axes = visualization.plot_weekly_overview(
        _sleep_loader(),
        "person-1",
        "2024-01-01",
        "2024-01-05",
        show=False,
    )

    assert set(axes) == {"activity", "heart_rate"}
    assert axes["activity"].shape == (4,)
    assert axes["heart_rate"].shape == (4,)
    assert not axes["activity"][0].figure.legends
    assert any(axis.patches for axis in axes["activity"])
    assert any(axis.lines for axis in axes["activity"])
    for axis in axes["activity"]:
        x, y = axis.lines[0].get_data()
        sloped = (np.diff(x) > 1e-9) & (np.abs(np.diff(y)) > 1e-9)
        assert not sloped.any()
        artificial_wake_spike = [
            x[index - 1] == x[index] == x[index + 1]
            and y[index] == pytest.approx(0.89)
            and y[index - 1] != pytest.approx(0.89)
            and y[index + 1] != pytest.approx(0.89)
            for index in range(1, len(x) - 1)
        ]
        assert not any(artificial_wake_spike)
        assert axis.get_ylabel() == "Steps / 15 min"


def test_adherence_summary_builds_four_cohort_panels(monkeypatch):
    def fake_adherence(
        loader,
        metric,
        user_id,
        start_date,
        end_date,
        resolution,
        time_reference,
    ):
        frequency = "1h" if resolution == "1h" else "1D"
        times = pd.date_range(start_date, end_date, freq=frequency, inclusive="left")
        users = ["person-1", "person-2", "outside-study"]
        index = pd.MultiIndex.from_product(
            [users, times],
            names=[
                schema.USER_COL,
                (
                    schema.LOCAL_TIME_COL
                    if resolution == "1h"
                    else schema.CALENDAR_DATE_COL
                ),
            ],
        )
        values = []
        for participant, timestamp in index:
            if participant == "person-1":
                value = float(
                    pd.Timestamp("2024-01-01")
                    <= timestamp
                    < pd.Timestamp("2024-01-15")
                )
            elif participant == "person-2":
                value = 0.5 * float(
                    pd.Timestamp("2024-01-08")
                    <= timestamp
                    < pd.Timestamp("2024-01-22")
                )
            else:
                value = 0.0
            values.append(value)
        return pd.DataFrame({f"{metric}_adherence": values}, index=index)

    monkeypatch.setattr(visualization.adherence, "get_adherence", fake_adherence)
    full_period_axes = visualization.plot_adherence_summary(
        SimpleNamespace(),
        "2024-01-01",
        "2024-01-22",
        show=False,
    )
    full_period_categories = full_period_axes["participant_weeks"].images[
        0
    ].get_array()
    assert full_period_categories.shape == (3, 3)
    assert not np.ma.getmaskarray(full_period_categories).any()

    inferred_axes = visualization.plot_adherence_summary(
        SimpleNamespace(),
        "2024-01-01",
        "2024-01-22",
        infer_participant_periods=True,
        show=False,
    )

    assert set(inferred_axes) == {
        "participant_weeks",
        "weekly_distribution",
        "weekday_pattern",
        "intraday",
    }
    assert inferred_axes["participant_weeks"].images[0].get_array().shape == (2, 2)
    weekday = inferred_axes["weekday_pattern"].images[0].get_array()
    intraday = inferred_axes["intraday"].images[0].get_array()
    assert weekday.shape == (7, 2)
    assert np.asarray(weekday).mean() == pytest.approx(0.75)
    assert intraday.shape == (24, 14)
    assert np.asarray(intraday).mean() == pytest.approx(0.75)


def test_empty_trajectory_reports_a_clear_error():
    index = pd.MultiIndex.from_arrays(
        [[], []], names=[schema.USER_COL, schema.CALENDAR_DATE_COL]
    )
    frame = pd.DataFrame(columns=["feature", "feature__coverage"], index=index)

    with pytest.raises(ValueError, match="at least one row"):
        visualization.plot_calendar_heatmap(frame, "feature", show=False)
