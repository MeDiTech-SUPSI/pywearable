sleep-stage
===========

``load_sleep_stage`` returns rows keyed by ``sleep_summary_id`` with
``local_time`` (stage start), ``duration_in_ms``, and ``type``. Stage labels
used by sleep feature calculations are ``deep``, ``light``, ``rem``,
``awake``, and ``unmeasurable``. Timestamps and column names use
``snake_case``. Where available, loaders also return ``utc_time`` and
``timezone_offset_in_ms``.

Oura's ``sleep_phase_5_min`` is decoded into runs of five-minute bins; it is
not a finer-grained continuous signal. If a night's recorded start and end
offsets differ, the stage UTC instants remain valid but the local stage
clock inside the night cannot be established from the export alone. Those
rows carry ``local_time_uncertain=True`` and their ``local_time`` is an
approximation based on the start offset. Night-level wake-up time uses the
recorded end time instead.

.. automethod:: pywearable.loader.BaseLoader.load_sleep_stage
