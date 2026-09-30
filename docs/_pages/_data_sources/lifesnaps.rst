LifeSnaps
=========

``LifeSnapsParquetLoader`` reads a LifeSnaps dataset that has already been
converted into pywearable's local Parquet layout::

    from pywearable.loader import LifeSnapsParquetLoader

    loader = LifeSnapsParquetLoader("path/to/lifesnaps-parquet")

Install the ``parquet`` optional dependency before using this loader::

    python -m pip install ".[parquet]"

The loader exposes timestamped Fitbit activity and heart-rate samples together
with provider-computed sleep, HRV, SpO2, and stress features where present in
the converted dataset. Missing source tables return empty canonical frames;
they are not interpreted as observed zeros.

See :class:`pywearable.loader.LifeSnapsParquetLoader` for the loader API.
