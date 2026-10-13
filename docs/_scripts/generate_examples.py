"""Generate the real-data figures embedded in the user guide."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pywearable import (  # noqa: E402
    activity,
    cycle,
    heart_rate,
    schema,
    sleep,
    trajectory,
    visualization,
)
from pywearable.loader import LabfrontLoader, OuraLoader  # noqa: E402


OUTPUT = ROOT / "docs" / "_static" / "figures"
DATA_ROOT = Path(
    os.environ.get(
        "PYWEARABLE_DOCS_OURA_ROOT",
        ROOT / "data_wearable" / "oura_20240822",
    )
)
LABFRONT_ROOT = os.environ.get("PYWEARABLE_DOCS_LABFRONT_ROOT")
USER_PREFIX = os.environ.get("PYWEARABLE_DOCS_USER_PREFIX", "A_")
PUBLIC_USER = "example-participant"
PUBLIC_MONDAY = pd.Timestamp("2026-01-05")


def _save(fig, name):
    """Save one guide figure as compact vector artwork."""
    path = OUTPUT / name
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    svg = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def _load_example():
    """Return the configured Oura loader and participant without printing the ID."""
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            "Set PYWEARABLE_DOCS_OURA_ROOT to the local Oura export root."
        )
    loader = OuraLoader(DATA_ROOT)
    matches = [user for user in loader.get_user_ids() if user.startswith(USER_PREFIX)]
    if not matches:
        raise ValueError(
            f"No participant directory starts with {USER_PREFIX!r} in {DATA_ROOT}."
        )
    return loader, matches[0]


def _load_labfront_example():
    """Return the configured Labfront loader without exposing participant IDs."""
    if not LABFRONT_ROOT:
        raise FileNotFoundError(
            "Set PYWEARABLE_DOCS_LABFRONT_ROOT to the local Labfront export root."
        )
    data_root = Path(LABFRONT_ROOT)
    if not data_root.is_dir():
        raise FileNotFoundError(
            "PYWEARABLE_DOCS_LABFRONT_ROOT does not identify a directory."
        )
    return LabfrontLoader(data_root)


def _analysis_data(loader, user, days=70):
    """Select the best-supported fixed-length activity interval."""
    source = loader.load_activity_summary(user)
    start = pd.to_datetime(source[schema.CALENDAR_DATE_COL]).min().normalize()
    end = pd.to_datetime(source[schema.CALENDAR_DATE_COL]).max().normalize()
    steps = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        user_id=user,
        start_date=start,
        end_date=end + pd.Timedelta(days=1),
        resolution="daily",
    )
    user_steps = steps.xs(user, level=schema.USER_COL)
    support = user_steps[activity.ACTIVITY_STATISTIC_STEPS].notna().astype(int)
    rolling = support.rolling(days, min_periods=days).sum()
    if rolling.notna().sum() == 0:
        raise ValueError(f"The example participant has fewer than {days} activity days.")
    final_day = rolling.idxmax()
    interval_start = final_day - pd.Timedelta(days=days - 1)
    interval_end = final_day + pd.Timedelta(days=1)
    times = steps.index.get_level_values(schema.CALENDAR_DATE_COL)
    selected = steps[(times >= interval_start) & (times < interval_end)]
    return selected, interval_start, interval_end


def _shift_trajectory(frame):
    """Replace participant and calendar labels while preserving measured values."""
    rows = frame.reset_index()
    time_levels = [
        name
        for name in frame.index.names
        if name not in {schema.USER_COL, schema.TIMEZONE_COL}
    ]
    if len(time_levels) != 1:
        raise ValueError("Expected one trajectory time level")
    time_level = time_levels[0]
    times = pd.to_datetime(rows[time_level])
    source_start = times.min().normalize()
    public_start = PUBLIC_MONDAY + pd.Timedelta(days=source_start.weekday())
    rows[time_level] = public_start + (times - source_start)
    rows[schema.USER_COL] = PUBLIC_USER
    return rows.set_index(frame.index.names)[frame.columns]


def _relative_week_axis(ax, origin):
    """Format a date axis as participant-relative study weeks."""
    origin = pd.Timestamp(origin)

    def label(value, _):
        timestamp = pd.Timestamp(mdates.num2date(value)).tz_localize(None)
        return f"Week {(timestamp.normalize() - origin).days // 7 + 1}"

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(FuncFormatter(label))
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Study week")


def _densest_sleep_window(loader, user, start, end, days=14):
    """Return the inclusive bounds of the densest requested sleep span."""
    summaries = loader.load_sleep_summary(user, start, end)
    observed = pd.Series(
        1,
        index=pd.DatetimeIndex(summaries[schema.CALENDAR_DATE_COL]).normalize(),
    )
    grid = pd.date_range(start, end, inclusive="left")
    support = observed.reindex(grid, fill_value=0).rolling(days, min_periods=days).sum()
    final_day = support.idxmax()
    return final_day - pd.Timedelta(days=days - 1), final_day


def make_cycle_figure(loader, user, start, end):
    """Compare cycle definitions over one real multi-day heart-rate trace."""
    cycle_types = (
        cycle.CYCLE_NIGHT,
        cycle.CYCLE_AWAKE,
        cycle.CYCLE_PHYSIOLOGICAL,
    )
    intervals = {
        kind: cycle.get_intervals(
            loader,
            kind,
            user_id=user,
            start_date=start,
            end_date=end,
        ).xs(user, level=schema.USER_COL)
        for kind in cycle_types
    }
    valid_sets = {
        kind: set(frame.index[frame["interval_valid"]])
        for kind, frame in intervals.items()
    }
    common_dates = sorted(set.intersection(*valid_sets.values()))
    common_set = set(common_dates)
    runs = [
        pd.DatetimeIndex([date + pd.Timedelta(days=offset) for offset in range(3)])
        for date in common_dates
        if all(
            date + pd.Timedelta(days=offset) in common_set for offset in range(3)
        )
    ]
    if not runs:
        raise ValueError("No three consecutive complete sleep-wake cycles are available.")

    heart_rate_all = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
        user_id=user,
        start_date=start,
        end_date=end,
        resolution="15min",
        time_reference="local",
    ).xs(user, level=schema.USER_COL)
    feature = heart_rate.HEART_RATE_STATISTIC_MEAN

    def bounds(run):
        rows = intervals[cycle.CYCLE_PHYSIOLOGICAL].loc[run]
        return pd.Timestamp(rows.iloc[0]["start_time"]), pd.Timestamp(
            rows.iloc[-1]["end_time"]
        )

    selected = max(
        runs,
        key=lambda run: heart_rate_all.loc[
            (heart_rate_all.index >= bounds(run)[0])
            & (heart_rate_all.index < bounds(run)[1]),
            feature,
        ].notna().sum(),
    )
    plot_start, plot_end = bounds(selected)
    heart_rate_window = heart_rate_all.loc[
        (heart_rate_all.index >= plot_start)
        & (heart_rate_all.index < plot_end)
    ]

    def elapsed_days(values):
        return (pd.to_datetime(values) - plot_start).total_seconds() / 86_400

    colors = {
        cycle.CYCLE_NIGHT: visualization.BLUE,
        cycle.CYCLE_AWAKE: visualization.ORANGE,
        cycle.CYCLE_PHYSIOLOGICAL: visualization.GREEN,
    }
    labels = {
        cycle.CYCLE_NIGHT: "Night cycles: bedtime to sleep-session end",
        cycle.CYCLE_AWAKE: "Awake cycles: session end to the next bedtime",
        cycle.CYCLE_PHYSIOLOGICAL: "Physiological cycles: bedtime to bedtime",
    }
    x = elapsed_days(heart_rate_window.index)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, kind in zip(axes, cycle_types):
        for calendar_date in selected:
            row = intervals[kind].loc[calendar_date]
            axis.axvspan(
                elapsed_days(pd.DatetimeIndex([row["start_time"]]))[0],
                elapsed_days(pd.DatetimeIndex([row["end_time"]]))[0],
                color=colors[kind],
                alpha=0.20,
                linewidth=0,
            )
        if kind == cycle.CYCLE_PHYSIOLOGICAL:
            for boundary in intervals[kind].loc[selected, "start_time"].iloc[1:]:
                axis.axvline(
                    elapsed_days(pd.DatetimeIndex([boundary]))[0],
                    color=visualization.BLACK,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.65,
                )
        axis.plot(
            x,
            heart_rate_window[feature],
            color=visualization.BLACK,
            linewidth=1.1,
        )
        axis.set_title(labels[kind], loc="left", fontsize=10)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(0, (plot_end - plot_start).total_seconds() / 86_400)
    axes[-1].set_xlabel("Elapsed days from the first bedtime")
    fig.supylabel("Heart rate (bpm)")
    fig.suptitle("One heart-rate trajectory under three cycle definitions")
    _save(fig, "cycles.svg")


def make_hierarchy_figure(loader, user, start, end):
    """Plot nightly sleep duration against its weekly hierarchy."""
    axis = visualization.plot_hierarchical_trajectory(
        loader,
        sleep.SLEEP_STATISTIC_TST,
        ("nightly", "weekly"),
        user_id=user,
        start_date=start,
        end_date=end,
        child_label="Nightly TST",
        center_label="Weekly mean",
        range_label="Weekly mean ± SD",
        title="Nightly total sleep time and its weekly hierarchy",
        ylabel="Total sleep time (minutes)",
        show=False,
    )
    _relative_week_axis(axis, start)
    _save(axis.figure, "hierarchy.svg")


def make_trajectory_figure(steps):
    """Render imputation, trend, normal range, and coverage together."""
    source = _shift_trajectory(steps)
    axes = visualization.plot_trajectory(
        source,
        activity.ACTIVITY_STATISTIC_STEPS,
        kind="bar",
        show_imputed=False,
        show_coverage=True,
        title="Daily steps and source coverage",
        ylabel="Steps",
        show=False,
    )
    _save(axes["trajectory"].figure, "trajectory.svg")

    frame = trajectory.impute_trajectory(
        source,
        activity.ACTIVITY_STATISTIC_STEPS,
        method="rolling_median",
        window=7,
        min_periods=4,
    )
    frame = trajectory.compute_trajectory_trend(
        frame,
        activity.ACTIVITY_STATISTIC_STEPS,
        baseline_window=7,
        min_periods_baseline=4,
        normal_range_window=21,
        min_periods_normal_range=10,
    )
    axes = visualization.plot_trajectory_trend(
        frame,
        activity.ACTIVITY_STATISTIC_STEPS,
        show_coverage=True,
        title="Daily steps with imputation and historical context",
        ylabel="Steps",
        show=False,
    )
    _save(axes["trajectory"].figure, "trajectory-trend.svg")


def make_calendar_figure(steps):
    """Render a calendar view of measured daily steps."""
    axis = visualization.plot_calendar_heatmap(
        _shift_trajectory(steps),
        activity.ACTIVITY_STATISTIC_STEPS,
        title="Daily steps",
        colorbar_label="Steps",
        show=False,
    )
    _save(axis.figure, "calendar-heatmap.svg")


def make_coverage_figure(loader):
    """Render one de-identified Labfront heart-rate coverage heatmap."""
    start = pd.Timestamp("2024-02-12")
    end = pd.Timestamp("2024-03-01")
    user = next(
        (
            candidate
            for candidate in loader.get_user_ids()
            if not loader.load_heart_rate(
                candidate,
                start_date=start,
                end_date=end,
                time_col=schema.LOCAL_TIME_COL,
            ).empty
        ),
        None,
    )
    if user is None:
        raise ValueError("No Labfront participant has heart-rate data in the window.")

    five_minute = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
        user_id=user,
        start_date=start,
        end_date=end,
        resolution="5min",
        time_reference="local",
    )
    axis = visualization.plot_coverage_heatmap(
        _shift_trajectory(five_minute),
        heart_rate.HEART_RATE_STATISTIC_MEAN,
        cmap="RdYlGn",
        title="Heart-rate coverage by local time",
        show=False,
    )
    _save(axis.figure, "coverage-heatmap.svg")


def make_weekly_profile_figure(loader, user, start, end):
    """Render seven real within-day heart-rate profiles."""
    hourly = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
        user_id=user,
        start_date=start,
        end_date=end,
        resolution="1h",
        time_reference="local",
    )
    axes = visualization.plot_weekly_profile(
        hourly,
        heart_rate.HEART_RATE_STATISTIC_MEAN,
        user_id=user,
        loader=loader,
        aggregation="median",
        title="Median heart-rate profile by weekday",
        ylabel="Heart rate (bpm)",
        figsize=(9, 8),
        show=False,
    )
    _save(axes[0].figure, "weekly-profile.svg")


def make_sleep_summary_figure(loader, user, start, end):
    """Render a de-identified multi-night sleep summary."""
    first_night, final_night = _densest_sleep_window(loader, user, start, end)
    axes = visualization.plot_sleep_summary(
        loader,
        user,
        first_night,
        final_night,
        cpd=("midpoint", "duration"),
        title="Sleep stages, timing, score, and composite phase deviation",
        figsize=(10, 6),
        fontsize=9,
        show=False,
    )
    main_ax = axes["sleep"]
    main_ax.set_yticklabels(
        [f"Night {index + 1}" for index in range(len(main_ax.get_yticks()))]
    )
    main_ax.set_ylabel("Observed night")
    _save(main_ax.figure, "sleep-summary.svg")


def make_sleep_heart_rate_figure(loader, user, start, end):
    """Render heart rate and stage context for one observed sleep session."""
    summaries = loader.load_sleep_summary(user, start, end)
    selected = None
    for summary in summaries.itertuples(index=False):
        session_hr = loader.load_heart_rate(
            user,
            start_date=pd.Timestamp(summary.local_time),
            end_date=pd.Timestamp(summary.local_end_time),
            time_col=schema.LOCAL_TIME_COL,
        )
        if session_hr[schema.HEART_RATE_COL].notna().any():
            selected = pd.Timestamp(summary.calendar_date)
            break
    if selected is None:
        raise ValueError("No sleep session overlaps local heart-rate samples.")
    axes = visualization.plot_sleep_heart_rate(
        loader,
        user,
        selected,
        show_coverage=True,
        title="Heart rate through one observed sleep session",
        show=False,
    )
    _save(axes["heart_rate"].figure, "sleep-heart-rate.svg")


def make_adherence_figure(loader):
    """Render a de-identified Labfront cohort adherence summary."""
    axes = visualization.plot_adherence_summary(
        loader,
        metric="heart_rate",
        user_id="all",
        follow_up_days=70,
        worn_day_threshold=0.7,
        include_without_evidence=False,
        title="Cohort heart-rate and sleep adherence",
        show=False,
    )
    participants = len(axes["wearing_days"].get_yticks())
    labels = [f"P{index + 1:02d}" for index in range(participants)]
    for name in ("wearing_days", "available_nights"):
        axes[name].set_yticklabels(labels)
    _save(axes["wearing_days"].figure, "adherence-summary.svg")


def main():
    """Generate every figure referenced by the Sphinx pages."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": "pywearable-docs",
        }
    )
    loader, user = _load_example()
    labfront_loader = _load_labfront_example()
    steps, start, end = _analysis_data(loader, user)
    make_cycle_figure(loader, user, start, end)
    make_hierarchy_figure(loader, user, start, end)
    make_trajectory_figure(steps)
    make_calendar_figure(steps)
    make_coverage_figure(labfront_loader)
    make_weekly_profile_figure(loader, user, start, end)
    make_sleep_summary_figure(loader, user, start, end)
    make_sleep_heart_rate_figure(loader, user, start, end)
    make_adherence_figure(labfront_loader)


if __name__ == "__main__":
    main()
