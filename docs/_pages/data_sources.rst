Data sources
============

Loaders translate provider-specific exports into the canonical schemas used by
the analysis modules. Code using ``activity``, ``sleep``, ``heart_rate``, and
the other domains therefore does not need to know the original field names.

The first-release public loaders are:

.. list-table:: Supported loaders
   :header-rows: 1

   * - Dataset
     - Loader
     - Input
   * - Labfront
     - :class:`pywearable.loader.LabfrontLoader`
     - Extracted Labfront study exports
   * - LifeSnaps
     - :class:`pywearable.loader.LifeSnapsParquetLoader`
     - Locally converted Parquet dataset
   * - Oura
     - :class:`pywearable.loader.OuraLoader`
     - Participant directories containing Oura API v2 JSON responses
   * - Tesserae
     - :class:`pywearable.loader.TesseraeCSVLoader` or
       :class:`pywearable.loader.TesseraeParquetLoader`
     - Published CSV layout or locally converted Parquet dataset

A loader can support samples, provider summaries, or both. Each domain
function offers the finest resolution supported by the source data.

.. toctree::
   :maxdepth: 1

   _data_sources/labfront
   _data_sources/lifesnaps
   _data_sources/oura
   _data_sources/tesserae
