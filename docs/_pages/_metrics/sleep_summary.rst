sleep-summary
=============

``load_sleep_summary`` returns one row per night, keyed by
``sleep_summary_id`` and the participant-local ``calendar_date``. The common
columns are ``local_time`` (bedtime), ``duration_in_ms`` (sleep, excluding
awake time), ``awake_duration_in_ms``, ``deep_sleep_duration_in_ms``,
``light_sleep_duration_in_ms``, ``rem_sleep_duration_in_ms``, and
``unmeasurable_sleep_duration_in_ms``. ``overall_sleep_score`` is optional;
an absent provider score is represented by a missing value, not by dropping
the night. Loader columns use ``snake_case``; existing non-camelCase feature
names such as ``TIB``, ``WASO``, and ``SME`` retain their spelling and case.

``drop_duplicates=True`` selects one night per local calendar date where a
provider has repeated summaries. Provider-specific fields can remain in the
table. In Oura, nightly features use ``long_sleep`` only; naps are available
separately through ``load_sleep_sessions``. Oura also supplies
``local_end_time`` and UTC start/end instants. Daily/nightly records are
assigned using Oura's ``day`` rather than a UTC-day conversion.

.. automethod:: pywearable.loader.BaseLoader.load_sleep_summary
