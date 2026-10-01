Working with trajectories
=========================

A trajectory is the common output of pywearable's feature modules. It is a
:class:`pandas.DataFrame` indexed by ``(user_id, time)`` and contains feature
values beside their coverage. This common shape allows the same
post-processing tools to work across activity, sleep, cardiac, and respiratory
features.

.. list-table:: Base trajectory columns
   :header-rows: 1

   * - Column pattern
     - Meaning
   * - ``<feature>``
     - Observed or calculated feature value
   * - ``<feature>__coverage``
     - Evidence supporting that row

Post-processing adds its own clearly named columns. The sections below
introduce those columns together with the operation that creates them.

Impute selected gaps
--------------------

:func:`pywearable.trajectory.impute_trajectory` fills eligible missing values
and adds ``<feature>__is_imputed``. This Boolean column is ``True`` only where
that feature was filled. Coverage continues to describe the source data that
supported the original row.

Historical methods use observations preceding the missing row:

``ma``
    Trailing moving mean.

``rolling_median``
    Trailing moving median; useful when occasional extremes should have less
    influence.

``ema``
    Exponentially weighted history, giving greater weight to recent values.

``cumulative_mean``
    Mean of all earlier observations for that participant.

Retrospective methods use the complete supplied trajectory:
``user_mean``, ``user_median``, ``population_mean``, and
``population_median``. Choose these only when using later observations is
consistent with the study design.

::

    from pywearable import trajectory

    filled = trajectory.impute_trajectory(
        daily_steps,
        features="activity_steps",
        method="ma",
        window=7,
        min_periods=3,
    )

Clock-valued features such as bedtime use circular time calculations, so
23:50 and 00:10 are treated as nearby times.

Add historical context
----------------------

:func:`pywearable.trajectory.compute_trajectory_trend` adds a baseline and a
normal range independently for each participant. By default, a row's trend is
calculated from earlier observed values and excludes imputed values.

.. list-table:: Columns added for each trended feature
   :header-rows: 1

   * - Column suffix
     - Meaning
   * - ``__trend_baseline``
     - Participant-specific historical baseline
   * - ``__trend_normal_center`` / ``__trend_normal_std``
     - Center and standard deviation of the historical reference distribution
   * - ``__trend_normal_lower`` / ``__trend_normal_upper``
     - Historical normal-range bounds
   * - ``__trend_deviation``
     - Difference between the current value and its baseline
   * - ``__trend_baseline_support`` / ``__trend_normal_support``
     - Number of historical observations supporting each estimate

.. image:: /_static/figures/trajectory-trend.svg
   :alt: Daily step trajectory showing observed and imputed values, a historical baseline, normal range, and source coverage.
   :width: 96%
   :align: center

The baseline and normal range can use different histories. A short baseline
can follow recent change, while a longer normal-range window describes the
participant's broader variability::

    enriched = trajectory.compute_trajectory_trend(
        filled,
        features="activity_steps",
        baseline_method="ma",
        baseline_window=7,
        min_periods_baseline=4,
        normal_range_method="ma",
        normal_range_window=30,
        min_periods_normal_range=14,
        std_multiplier=1.0,
    )

Available trend methods are ``ma``, ``ema``, and ``cumulative_mean``. Support
columns record how many historical observations contributed to each estimate.
For clock-valued features, deviations and standard deviations are expressed
in minutes.

Summarize a typical week
------------------------

:func:`pywearable.trajectory.aggregate_trajectory_by_weekday` combines all
Mondays, all Tuesdays, and so on within each participant. It supports mean,
median, min, max, standard deviation, and percentiles such as ``p90``. This is
a descriptive weekday profile, not a chronological parent hierarchy.

::

    weekday_steps = trajectory.aggregate_trajectory_by_weekday(
        daily_steps,
        features="activity_steps",
        aggregation="median",
    )

Use :func:`pywearable.visualization.plot_weekly_profile` when the source is a
subdaily trajectory and the goal is to compare within-day patterns across the
seven weekdays.
