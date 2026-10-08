HRV
===

Use :func:`pywearable.hrv.get_hrv_statistics` as the canonical entry point.
The convenience getters below select a common feature or resolution and refer
to it for shared parameter and coverage semantics.

Raw-BBI extraction supports overlapping primary windows through
``window_step`` and configurable sleep-stage exclusion through
``sleep_stages_to_remove``. The built-in and NeuroKit backends expose separate
default feature registries; explicit ``statistic`` selections return only the
requested features.

.. automodule:: pywearable.hrv
   :members:
   :undoc-members:
   :member-order: bysource
