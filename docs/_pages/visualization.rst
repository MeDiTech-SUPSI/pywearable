Visualization guide
===================

The visualization module contains two kinds of plots. Trajectory plots accept
an analysis-ready DataFrame, while composite plots accept a loader because
they coordinate several streams in one figure. Every function returns its
Matplotlib axes, so labels, annotations, and layout can be customized after
the plot is created.

The figures below use de-identified measurements from one Oura export. Dates
are shifted or replaced by relative study labels; measurements and temporal
patterns are retained.

Choose a plot
-------------

.. list-table:: Plot chooser
   :header-rows: 1
   :widths: 30 42 28

   * - Function
     - Best used for
     - Input
   * - :func:`~pywearable.visualization.plot_trajectory`
     - One or more features over time, optionally with coverage
     - Trajectory
   * - :func:`~pywearable.visualization.plot_trajectory_trend`
     - Values, imputation, historical baseline, and normal range
     - Trend-enriched trajectory
   * - :func:`~pywearable.visualization.plot_hierarchical_trajectory`
     - Child observations against a parent mean and standard deviation
     - Loader and feature request
   * - :func:`~pywearable.visualization.plot_calendar_heatmap`
     - One daily value in a familiar calendar layout
     - Daily trajectory
   * - :func:`~pywearable.visualization.plot_coverage_heatmap`
     - Subdaily data availability by date and local time
     - Subdaily trajectory
   * - :func:`~pywearable.visualization.plot_weekly_profile`
     - Typical within-day pattern for each weekday
     - Subdaily trajectory
   * - :func:`~pywearable.visualization.plot_period_comparison`
     - Consecutive weekly, monthly, or yearly values and percentage changes
     - Aggregated trajectory
   * - :func:`~pywearable.visualization.plot_sleep_summary`
     - Sleep stages, timing, score, and optional CPD panels across nights
     - Loader
   * - :func:`~pywearable.visualization.plot_sleep_heart_rate`
     - Raw and aggregated heart rate through one sleep session
     - Loader
   * - :func:`~pywearable.visualization.plot_weekly_overview`
     - Sleep, activity, heart rate, and missingness in aligned 24-hour rows
     - Loader
   * - :func:`~pywearable.visualization.plot_adherence_summary`
     - Cohort adherence by participant, week, weekday, and local hour
     - Loader

Trajectory and trend
--------------------

Use :func:`~pywearable.visualization.plot_trajectory` for a direct view of a
feature. Add a coverage panel when the support behind each row matters::

    from pywearable import activity, visualization

    steps = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        user_id="participant-01",
        start_date="2026-01-01",
        end_date="2026-02-01",
        resolution="daily",
    )

    axes = visualization.plot_trajectory(
        steps,
        activity.ACTIVITY_STATISTIC_STEPS,
        kind="bar",
        show_coverage=True,
        ylabel="Steps",
    )

For historical context, enrich the trajectory first and pass the result to
:func:`~pywearable.visualization.plot_trajectory_trend`::

    from pywearable import trajectory

    enriched = trajectory.compute_trajectory_trend(
        steps,
        activity.ACTIVITY_STATISTIC_STEPS,
    )
    visualization.plot_trajectory_trend(
        enriched,
        activity.ACTIVITY_STATISTIC_STEPS,
        show_deviation=True,
        ylabel="Steps",
    )

.. image:: /_static/figures/trajectory-trend.svg
   :alt: Daily step trajectory with observed and imputed values, historical baseline, normal range, and source coverage.
   :width: 96%
   :align: center

Imputed values have a distinct marker or hatch. Clock features such as bedtime
are displayed on a continuous clock axis across midnight.

Calendar views
--------------

A calendar heatmap is useful for daily features such as steps, total sleep
time, resting heart rate, or a provider score::

    visualization.plot_calendar_heatmap(
        steps,
        activity.ACTIVITY_STATISTIC_STEPS,
        title="Daily steps",
        colorbar_label="Steps",
    )

.. image:: /_static/figures/calendar-heatmap.svg
   :alt: Calendar heatmap of daily steps with missing dates shown in grey.
   :width: 96%
   :align: center

:func:`~pywearable.visualization.plot_coverage_heatmap` uses rows for dates
and columns for local time. It is designed for questions such as “when during
the day is heart-rate coverage usually available?”

Typical within-day profiles
---------------------------

:func:`~pywearable.visualization.plot_weekly_profile` groups a subdaily
trajectory by weekday and local-time slot. The default line view creates seven
aligned panels; ``kind="heatmap"`` provides a compact matrix.

.. image:: /_static/figures/weekly-profile.svg
   :alt: Seven aligned weekday panels showing the median hourly heart-rate profile, with observed sleep windows shaded.
   :width: 82%
   :align: center

The profile describes repeated weekdays across the supplied interval. It does
not replace chronological weekly aggregation, which is covered in
:doc:`resolutions`.

Sleep views
-----------

:func:`~pywearable.visualization.plot_sleep_summary` shows native stage
intervals across consecutive sleep calendar dates. It can add provider sleep
scores, average bedtime and wake-up boundaries, and narrow CPD panels.

.. image:: /_static/figures/sleep-summary.svg
   :alt: Multi-night sleep summary with stage intervals, average bedtime and wake-up, sleep scores, and composite phase deviation.
   :width: 90%
   :align: center

::

    visualization.plot_sleep_summary(
        loader,
        user_id="participant-01",
        start_date="2026-01-01",
        end_date="2026-01-07",
        cpd=("midpoint", "duration"),
    )

For this figure, ``start_date`` and ``end_date`` are inclusive sleep-calendar
labels. A selected night retains its complete stage intervals, including the
portion before midnight.

:func:`~pywearable.visualization.plot_sleep_heart_rate` focuses on one night
and overlays raw heart-rate samples, a fixed-window mean, sleep-stage
backgrounds, and optional coverage.

.. image:: /_static/figures/sleep-heart-rate.svg
   :alt: Raw and aggregated heart rate over one observed sleep session, with sleep-stage backgrounds and coverage.
   :width: 96%
   :align: center

Use
:func:`~pywearable.visualization.plot_weekly_overview` when the goal is an
aligned multi-day view of sleep stages, steps, heart rate, and missingness.

Adherence monitoring
--------------------

:func:`~pywearable.visualization.plot_adherence_summary` is a cohort view. It
shows worn days, available nights, cohort summaries, weekday patterns, and
local-hour patterns. Participants are aligned at their first date with
wearable or sleep evidence; ``follow_up_days`` can set a shared study window.

Save and customize
------------------

Pass ``show=False`` when building figures in a script and ``save_to`` to write
the output. The returned axes can be edited with normal Matplotlib methods::

    axis = visualization.plot_calendar_heatmap(
        steps,
        activity.ACTIVITY_STATISTIC_STEPS,
        show=False,
        save_to="daily-steps.svg",
    )
    axis.set_title("Participant 01 — daily steps")

The generation script is ``docs/_scripts/generate_examples.py``.
