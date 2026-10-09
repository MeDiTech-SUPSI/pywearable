What can I extract?
===================

Feature functions return a trajectory with one row per participant and time
window. Pass one feature name, a list of names, or ``None`` where documented
to request all features in a domain.

.. important::

   The features listed here are capabilities of pywearable, not a guarantee
   that every loader can calculate all of them. A feature is available only
   when the selected loader supplies its required source stream, time
   information, and sufficient source resolution. For example, daily activity
   summaries support daily step totals, while peak-hour activity requires
   timestamped activity samples. See :doc:`data_sources` for the streams
   exposed by each built-in loader.

The examples below use the canonical functions. Named getters such as
:func:`pywearable.sleep.get_total_sleep_time` and
:func:`pywearable.heart_rate.get_daily_mean_heart_rate` are equivalent
convenience entry points.

Activity
--------

Use :func:`pywearable.activity.get_activity_statistics` to study accumulated
movement and the most active hour.

.. list-table:: Activity features
   :header-rows: 1

   * - Question
     - Feature
   * - How many steps were recorded?
     - ``activity_steps``
   * - How many steps occurred in the most active 60 minutes?
     - ``activity_peak_60min_steps``
   * - When did that 60-minute peak begin?
     - ``activity_peak_60min_start``

Peak-hour features require timestamped activity samples. Loaders backed only
by daily summaries can still provide daily step totals.

Heart-rate presence can establish wear coverage, but explicit unknown step
counts remain missing rather than becoming zero or a complete total.

Heart rate
----------

Use :func:`pywearable.heart_rate.get_heart_rate_statistics` for heart-rate
samples. Available summaries include ``mean``, ``median``, ``min``, ``max``,
``std``, ``p10``, ``p90``, and ``count``. ``heart_rate_rest`` estimates the
resting value for daily, nightly, or sleep-cycle windows.

For example, request several nightly summaries in one call::

    from pywearable import cycle, heart_rate

    nightly_hr = heart_rate.get_heart_rate_statistics(
        loader,
        statistic=[
            heart_rate.HEART_RATE_STATISTIC_MEAN,
            heart_rate.HEART_RATE_STATISTIC_MIN,
            heart_rate.HEART_RATE_STATISTIC_REST,
        ],
        user_id="participant-01",
        start_date="2026-01-01",
        end_date="2026-02-01",
        resolution=cycle.CYCLE_NIGHT,
    )

Heart-rate variability
----------------------

Use :func:`pywearable.hrv.get_hrv_statistics` for beat-to-beat interval data.

.. list-table:: HRV feature families
   :header-rows: 1

   * - Family
     - Features
   * - Time domain
     - RMSSD, SDNN, pNN20, pNN50, median, minimum, mean, and maximum NN,
       SDNN/RMSSD, CVNN, CVSD, and IQRNN
   * - Frequency domain
     - VLF, LF, and HF power, plus the LF/HF ratio

The default implementation is self-contained. Install the ``neurokit`` extra
when the NeuroKit backend is required by an analysis protocol. With
``statistic=None``, the built-in backend returns the features listed above,
whereas the NeuroKit backend returns its broader time- and frequency-domain
set. Supplying ``statistic`` always limits the output to the requested names.
Frequency-domain values from different backends are not interchangeable.

Raw BBI data use five-minute primary windows. A shorter ``window_step`` creates
overlapping estimates, which can still feed larger hierarchy levels. During
sleep-cycle extraction, ``awake`` and ``unmeasurable`` stages are excluded by
default; ``sleep_stages_to_remove`` configures or disables this filter. The
observed timestamps are retained when handling missing BBI rows.

Respiration and oxygen saturation
---------------------------------

Use :func:`pywearable.respiration.get_respiration_statistics` for respiratory
rate and SpO2.

* Respiratory rate supports mean, median, standard deviation, p10, p90, and
  sample count.
* SpO2 supports mean, median, standard deviation, p10, minimum, and sample
  count.
* ``spo2_below_threshold_fraction`` and
  ``spo2_below_threshold_duration_seconds`` quantify time below a selected
  threshold.

Night and awake-cycle helpers are available through
:func:`pywearable.respiration.get_night_respiration_statistics` and
:func:`pywearable.respiration.get_awake_respiration_statistics`.

Native nightly summaries retain missing dates in the requested interval, so
upper-level coverage includes unavailable nights rather than only returned rows.

Sleep
-----

Use :func:`pywearable.sleep.get_sleep_statistics` for one row per main sleep
or for calendar aggregates derived from those nights.

.. list-table:: Sleep feature groups
   :header-rows: 1
   :widths: 24 76

   * - Group
     - Available information
   * - Duration and continuity
     - Time in bed (TIB), total sleep time (TST), sleep period time (SPT),
       wake after sleep onset (WASO), sleep efficiency (SE), and sleep
       maintenance efficiency (SME)
   * - Stage composition
     - Deep, light, REM, NREM, awake, and unmeasurable durations; stage
       percentages; and stage counts
   * - Latencies
     - Sleep onset latency (SOL), get-up latency, and deep, light, and REM
       stage latencies
   * - Fragmentation
     - Awakening count, awakening index, median awakening duration, stage
       transition count, and stage transition index
   * - Timing
     - Bedtime, sleep onset, wake-up, get-up, and midpoint
   * - Regularity
     - Composite phase deviation (CPD) for midpoint and duration
   * - Provider summary
     - Nightly sleep score when supplied by the source

Durations and latencies are expressed in minutes. Efficiencies and stage
percentages are percentages. Timing features are participant-local clock
values and remain meaningful across midnight. CPD midpoint and duration are
expressed in hours.

The Sleep Regularity Index (SRI) compares sleep/wake state with the same local
time on the previous day. Use :func:`pywearable.sleep.get_sleep_regularity`
for SRI with comparable-epoch coverage, or
:func:`pywearable.sleep.get_sleep_regularity_index` for the SRI series alone.

Provider-native features
------------------------

Some useful outcomes are calculated by individual wearable providers rather
than by pywearable. Their availability therefore depends on both the provider
and the loader: Garmin measures are expected only from Garmin-backed sources,
for example, while Oura readiness requires Oura data.
:func:`pywearable.provider_features.get_available_provider_features` reports
which provider-native features the current loader can supply. They currently
include:

* Garmin stress and Body Battery measures;
* Fitbit stress-management scores and components; and
* Oura readiness scores and components.

Provider names remain part of these feature names because their definitions
are provider-specific::

    from pywearable import provider_features

    available = provider_features.get_available_provider_features(loader)
    readiness = provider_features.get_provider_features(
        loader,
        feature=provider_features.OURA_READINESS_SCORE,
        user_id="participant-01",
        start_date="2026-01-01",
        end_date="2026-02-01",
    )

Adherence
---------

Use :func:`pywearable.adherence.get_adherence` to quantify how much of each
requested time window is represented by valid heart-rate, beat-to-beat
interval, or sleep data.

Next steps
----------

Choose the time scale and aggregation path in :doc:`resolutions`. The domain
pages in :doc:`api` document the exact function parameters; exported feature
constants mirror the names shown here.
