.. include:: macros.rst

Statistics
==========
The overall ``pywearable`` package extracts features from wearable data through
a common loader interface.

Trajectory format
^^^^^^^^^^^^^^^^^
The refactored feature modules return pandas objects with a
:class:`pandas.MultiIndex`. The first index level is ``user_id`` and the
remaining level or levels identify the natural temporal resolution of the
domain.

Sleep features are returned at one row per night, indexed by ``user_id`` and
``calendar_date``. A request for one feature returns a
:class:`pandas.Series`; a request for multiple features returns a
:class:`pandas.DataFrame`::

    nights = pywearable.sleep.get_sleep_statistics(
        loader,
        user_id="all",
        start_date=None,
        end_date=None,
        statistic=["TST", "WASO"],
    )

Missing feature values do not remove the corresponding night. Sleep extraction
does not provide dictionary output or a flat-index switch.

Aggregation
^^^^^^^^^^^
Feature extraction first produces the trajectory at the domain's natural
resolution. Weekly, monthly, and whole-period summaries belong to a separate
trajectory-aggregation layer. This separation is needed for explicit
hierarchical aggregation and for suitable circular statistics for bedtime,
wake-up time, and sleep midpoint.

Legacy modules
^^^^^^^^^^^^^^
Not every feature module has completed this migration. Some legacy modules may
still expose ``return_df``, ``return_multi_index``, or ``kind`` parameters.
Their module documentation remains authoritative until they adopt the common
trajectory contract.
