Feature extraction
==================

Each domain owns one canonical extraction function:

.. list-table:: Domain entry points
   :header-rows: 1

   * - Domain
     - Function
   * - Activity
     - :func:`pywearable.activity.get_activity_statistics`
   * - Adherence
     - :func:`pywearable.adherence.get_adherence`
   * - Heart rate
     - :func:`pywearable.heart_rate.get_heart_rate_statistics`
   * - HRV
     - :func:`pywearable.hrv.get_hrv_statistics`
   * - Respiratory rate and SpO2
     - :func:`pywearable.respiration.get_respiration_statistics`
   * - Sleep
     - :func:`pywearable.sleep.get_sleep_statistics`
   * - Provider-native features
     - :func:`pywearable.provider_features.get_provider_features`

Selecting participants and dates
--------------------------------

``user_id`` accepts one participant, a sequence, ``"all"``, or ``None`` where
documented. Date bounds are half-open: ``[start_date, end_date)``. Calendar and
sleep-cycle bounds select participant-local labels. They do not clip a sleep
session at midnight or at the requested label boundary.

Resolutions and clocks
----------------------

A scalar ``resolution`` returns one row per requested fixed window, calendar
period, or sleep-based cycle. Examples include ``"5min"``, ``"daily"``,
``"weekly"``, ``"nightly"``, and ``"cycle_awake"`` when supported by the
domain.

Timestamped streams use UTC for fixed subdaily windows with
``time_reference="auto"`` when UTC data are available. Calendar windows are
participant-local. Set ``time_reference="local"`` explicitly when a local
clock analysis is intended.

Resolution chains perform hierarchical aggregation. For example::

    from pywearable import heart_rate

    weekly = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
        resolution=("daily", "weekly"),
        aggregation="mean",
        min_coverage=(0.7, 5 / 7),
    )

This first validates daily heart-rate means and then aggregates valid days into
weeks. Valid children have equal weight at the parent level. Parent coverage is
the fraction of expected children that passed the child requirement.

Coverage and missingness
------------------------

Coverage always describes the source appropriate to the domain. Examples
include occupied heart-rate cadence, represented BBI duration, valid expected
nights, and provider-supplied summary availability. Consult the canonical
function's docstring for the exact denominator.

``min_coverage`` filters weak windows without converting them into observed
zeros. ``min_valid_children`` can impose an additional count requirement on a
parent hierarchy. Unsupported combinations remain missing or raise a clear
validation error; they are not approximated from unrelated fields.

Convenience getters
-------------------

Named getters such as :func:`pywearable.sleep.get_total_sleep_time`,
:func:`pywearable.sleep.get_bedtime`, and
:func:`pywearable.heart_rate.get_daily_mean_heart_rate` are permanent public
conveniences. They forward to the owning canonical function and preserve its
date, coverage, and missingness semantics.

Provider-native features
------------------------

Provider scores retain their provider prefix because similarly named scores
are not assumed to measure the same construct. Inspect a loader first::

    from pywearable import provider_features

    available = provider_features.get_available_provider_features(loader)
    features = provider_features.get_provider_features(
        loader,
        feature=available[0],
        user_id="all",
    )

Trajectory post-processing
--------------------------

:mod:`pywearable.trajectory` operates on already calculated canonical
trajectories. Imputation records filled values separately, trends use
historical observations by default, and weekday aggregation never changes the
source coverage columns silently.
