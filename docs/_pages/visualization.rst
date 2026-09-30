Visualization
=============

Most plotting functions consume canonical trajectories, keeping feature
calculation separate from presentation. Composite views accept a loader only
when they must synchronize several raw streams.

Trajectory plots
----------------

Use :func:`pywearable.visualization.plot_trajectory` for one or more compatible
features::

    from pywearable import activity, visualization

    steps = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        user_id="participant-01",
        start_date="2026-01-01",
        end_date="2026-02-01",
    )

    axes = visualization.plot_trajectory(
        steps,
        activity.ACTIVITY_STATISTIC_STEPS,
        show_coverage=True,
        ylabel="Steps",
    )

Missing observations remain gaps. Numeric zero remains visible. If a
trajectory contains ``__is_imputed`` flags, imputed observations use distinct
markers or hatching.

Trend plots
-----------

Calculate a trend first, then plot it::

    from pywearable import trajectory

    enriched = trajectory.compute_trajectory_trend(
        steps,
        activity.ACTIVITY_STATISTIC_STEPS,
    )
    visualization.plot_trajectory_trend(
        enriched,
        activity.ACTIVITY_STATISTIC_STEPS,
        show_deviation=True,
    )

Calendar and profile views
--------------------------

The generic calendar, coverage, weekly-profile, hierarchical, and
period-comparison functions accept prepared trajectories. Clock-valued sleep
features are unwrapped across midnight automatically where documented.

Loader-aware composite views
----------------------------

The following plots load several streams internally because the composition is
the feature:

* :func:`pywearable.visualization.plot_sleep_summary`
* :func:`pywearable.visualization.plot_sleep_heart_rate`
* :func:`pywearable.visualization.plot_weekly_overview`
* :func:`pywearable.visualization.plot_adherence_summary`

Every public plotting function returns its Matplotlib axes for further
customization. ``show=False`` suppresses display, while ``save_to`` writes the
figure without changing its analytical content.
