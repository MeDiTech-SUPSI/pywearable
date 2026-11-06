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
     - Locally converted Parquet version of the LifeSnaps dataset
   * - Oura
     - :class:`pywearable.loader.OuraLoader`
     - Participant directories containing Oura API v2 JSON responses
   * - Tesserae
     - :class:`pywearable.loader.TesseraeCSVLoader` or
       :class:`pywearable.loader.TesseraeParquetLoader`
     - Published CSV layout or locally converted Parquet version of the
       Tesserae dataset

A loader can support samples, provider summaries, or both. Each domain
function offers the finest resolution supported by the source data.

LifeSnaps [1]_ and Tesserae [2]_ are public research datasets. Cite their
original publications when using them.

.. toctree::
   :maxdepth: 1

   _data_sources/labfront
   _data_sources/lifesnaps
   _data_sources/oura
   _data_sources/tesserae

References
----------

.. [1] Yfantidou, Sofia, et al. "LifeSnaps, a 4-month multi-modal dataset
   capturing unobtrusive snapshots of our lives in the wild." *Scientific
   Data* 9.1 (2022): 663.
.. [2] Mattingly, Stephen M., et al. "The Tesserae Project: Large-Scale,
   Longitudinal, In Situ, Multimodal Sensing of Information Workers."
   *Extended Abstracts of the 2019 CHI Conference on Human Factors in
   Computing Systems* (2019).
