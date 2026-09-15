Oura
====

``OuraLoader`` reads copied Oura API v2 JSON exports arranged as
``participant/metric/*.json``. It never rewrites source files. Daily and
nightly tables use the provider's local ``day``; subdaily HR and HRV use
offset-aware UTC instants. Oura HR exports do not supply a reliable
per-sample local timezone, so local-day HR extraction is unavailable.

Nightly sleep features use ``long_sleep``; raw sessions include naps. The
provider's nightly and five-minute HRV values are labelled as Oura estimates
and are not interchangeable with BBI-derived HRV. Nightly respiration and
daily SpO2 are summaries rather than continuous signals. Steps, readiness,
and resilience remain vendor-labelled daily tables. Generic activity
intensity and recovery mappings are not implemented.

.. autoclass:: pywearable.loader.OuraLoader
   :members:
