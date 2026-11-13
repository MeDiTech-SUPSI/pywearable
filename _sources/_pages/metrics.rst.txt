Data model and missingness
==========================

pywearable separates two related table types: canonical loader frames and
feature trajectories.

Loader frames
-------------

Loader frames contain observations in provider-independent columns. Public
loaders perform this normalization before returning data.

.. list-table:: Core canonical streams
   :header-rows: 1

   * - Stream
     - Required columns
   * - Heart rate
     - ``utc_time``, ``local_time``, ``timezone``, ``heart_rate``
   * - Beat-to-beat intervals
     - ``utc_time``, ``local_time``, ``timezone``, ``bbi``
   * - Activity samples
     - ``utc_time``, ``local_time``, ``timezone``, ``duration_in_ms``,
       ``activity_steps``
   * - Daily activity summary
     - ``calendar_date``, ``activity_steps``
   * - Respiratory-rate samples
     - ``utc_time``, ``local_time``, ``timezone``, ``respiratory_rate``
   * - SpO2 samples
     - ``utc_time``, ``local_time``, ``timezone``, ``spo2``
   * - Sleep summary
     - ``sleep_summary_id``, ``calendar_date``, ``local_time``
   * - Sleep stage interval
     - ``sleep_summary_id``, ``local_time``, ``duration_in_ms``, ``type``
   * - Sleep session
     - ``id``, ``calendar_date``, ``type``, ``main_sleep``, ``local_time``,
       ``local_end_time``

``utc_time`` is timezone-naive UTC, ``local_time`` is timezone-naive
participant-local time, and ``timezone`` is the UTC offset represented by a
:class:`pandas.Timedelta`. ``calendar_date`` is a normalized participant-local
date label.

The public constants and validation functions are listed in
:mod:`pywearable.schema`. Loader authors can use them to produce the same
tables as the built-in loaders.

Feature trajectories
--------------------

Analysis functions return a :class:`pandas.DataFrame` with a two-level
``(user_id, time)`` index. The time level is named ``utc_time``, ``local_time``,
or ``calendar_date`` according to the requested clock and resolution.
Fixed-duration trajectories also retain the other clock and the applicable
``timezone`` offset as columns. For example, the default UTC-indexed
five-minute output contains ``local_time`` and ``timezone`` columns. This
allows trajectories to be joined on UTC and subsequently interpreted on each
participant's local clock, including across offset changes.

Each feature has a separate ``<feature>__coverage`` column. The value and its
support can therefore be inspected independently:

* ``0`` records a measured zero when the feature permits it;
* ``NaN`` identifies a value that is unavailable for that row;
* coverage is expressed on the scale documented by the owning feature.

The :doc:`trajectories` guide introduces the optional annotations added by
imputation and trend analysis after describing this base format.

Hierarchical feature names record their calculation path. For example,
``heart_rate_mean__weekly_mean`` is a weekly mean of daily heart-rate means.
This makes the calculation represented by a column visible in its name.

Sleep stages
------------

Canonical sleep stages are ``awake``, ``rem``, ``light``, ``deep``, and
``unmeasurable``. The general ``asleep`` value is accepted when a provider
cannot distinguish stages. Adjacent intervals with the same stage and sleep
session are coalesced during schema preparation.
