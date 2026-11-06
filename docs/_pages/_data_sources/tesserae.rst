Tesserae
========

The public Tesserae dataset [1]_ can be read directly from its published CSV
layout::

    from pywearable.loader import TesseraeCSVLoader

    loader = TesseraeCSVLoader("path/to/tesserae-csv")

Detailed description of the Tesserae dataset is available on OSF: `Tesserae <https://osf.io/yvw2f/overview>`_.

The CSV version of the dataset is quite memory consuming, so pywearable offers a 
useful command that you can use to convert the full dataset in Parquet format::

    pywearable convert tesserae-parquet SOURCE_DIR OUTPUT_DIR

Afterwards, you can use the Tesserae loader for Parquet files::

    from pywearable.loader import TesseraeParquetLoader

    loader = TesseraeParquetLoader("path/to/tesserae-parquet")

See :class:`pywearable.loader.TesseraeCSVLoader` and
:class:`pywearable.loader.TesseraeParquetLoader` for their APIs.

References
----------

.. [1] Mattingly, Stephen M., et al. "The Tesserae Project: Large-Scale,
   Longitudinal, In Situ, Multimodal Sensing of Information Workers."
   *Extended Abstracts of the 2019 CHI Conference on Human Factors in
   Computing Systems* (2019).
