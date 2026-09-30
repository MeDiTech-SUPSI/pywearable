Tesserae
========

The public Tesserae dataset can be read directly from its published CSV
layout::

    from pywearable.loader import TesseraeCSVLoader

    loader = TesseraeCSVLoader("path/to/tesserae-csv")

For repeatedly analysing a local conversion, use the Parquet loader and
install the ``parquet`` optional dependency::

    from pywearable.loader import TesseraeParquetLoader

    loader = TesseraeParquetLoader("path/to/tesserae-parquet")

Both loaders expose canonical timestamped activity and heart-rate samples.
The Parquet loader also exposes BBI data when present.

See :class:`pywearable.loader.TesseraeCSVLoader` and
:class:`pywearable.loader.TesseraeParquetLoader` for their APIs.
