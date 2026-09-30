Canonical data model
====================

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
:mod:`pywearable.schema`. Provider field names are deliberately absent from
the canonical contract.

Feature trajectories
--------------------

Analysis functions return a :class:`pandas.DataFrame` with a two-level
``(user_id, time)`` index. The time level is named ``utc_time``, ``local_time``,
or ``calendar_date`` according to the requested clock and resolution.

Each feature has a separate ``<feature>__coverage`` column. Coverage describes
the evidence supporting the feature; it is not encoded by replacing missing
values with zero. Consequently:

* ``0`` is a valid observed value when the feature permits it;
* ``NaN`` means unavailable, unsupported, or invalid;
* coverage remains separate from the value; and
* imputed values receive ``<feature>__is_imputed`` without changing coverage.

Hierarchical feature names record their calculation path. For example, a
weekly mean calculated from valid daily values has a name such as
``heart_rate_mean__weekly_mean`` rather than silently replacing the meaning of
``heart_rate_mean``.

Sleep stages
------------

Canonical sleep stages are ``awake``, ``rem``, ``light``, ``deep``, and
``unmeasurable``. The general ``asleep`` value is accepted when a provider
cannot distinguish stages. Adjacent intervals with the same stage and sleep
session are coalesced during schema preparation.
