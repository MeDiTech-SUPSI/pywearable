LifeSnaps
=========

``LifeSnapsParquetLoader`` reads the LifeSnaps dataset [1]_ after it has been
converted into pywearable's local Parquet layout::

    from pywearable.loader import LifeSnapsParquetLoader

    loader = LifeSnapsParquetLoader("path/to/lifesnaps-parquet")

Install the ``parquet`` optional dependency before using this loader::

    python -m pip install ".[parquet]"

The loader exposes timestamped Fitbit activity and heart-rate samples together
with provider-computed sleep, HRV, SpO2, and stress features where present in
the converted dataset. Missing source tables return empty canonical frames,
preserving the distinction between unavailable data and measured zeros.

See :class:`pywearable.loader.LifeSnapsParquetLoader` for the loader API.

References
----------

.. [1] Yfantidou, Sofia, et al. "LifeSnaps, a 4-month multi-modal dataset
   capturing unobtrusive snapshots of our lives in the wild." *Scientific
   Data* 9.1 (2022): 663.
