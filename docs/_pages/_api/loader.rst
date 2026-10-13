Loaders
========

Portable interface
------------------

.. autoclass:: pywearable.loader.BaseLoader
   :members:
   :member-order: bysource

Supported implementations
-------------------------

.. autoclass:: pywearable.loader.LabfrontLoader
   :members: get_user_ids, load_sleep_summary, load_sleep_stage, load_sleep_sessions, load_sleep_session_stages, load_heart_rate, load_bbi, load_activity_samples, load_respiration, load_pulse_ox, load_provider_feature

.. autoclass:: pywearable.loader.LifeSnapsParquetLoader
   :members: get_user_ids, load_sleep_summary, load_sleep_stage, load_sleep_sessions, load_sleep_session_stages, load_heart_rate, load_bbi, load_activity_samples, load_respiration, load_respiration_summary, load_pulse_ox, load_spo2_summary, load_hrv_samples, load_hrv_summary, load_provider_feature

.. autoclass:: pywearable.loader.OuraLoader
   :members: get_user_ids, load_sleep_summary, load_sleep_stage, load_sleep_sessions, load_sleep_session_stages, load_heart_rate, load_activity_summary, load_respiration, load_respiration_summary, load_pulse_ox, load_spo2_summary, load_hrv_samples, load_hrv_summary, load_provider_feature

.. autoclass:: pywearable.loader.TesseraeCSVLoader
   :members: get_user_ids, load_sleep_summary, load_sleep_stage, load_sleep_sessions, load_sleep_session_stages, load_heart_rate, load_activity_samples, load_respiration, load_pulse_ox, load_provider_feature

.. autoclass:: pywearable.loader.TesseraeParquetLoader
   :members: get_user_ids, load_sleep_summary, load_sleep_stage, load_sleep_sessions, load_sleep_session_stages, load_heart_rate, load_bbi, load_activity_samples, load_respiration, load_pulse_ox, load_provider_feature
