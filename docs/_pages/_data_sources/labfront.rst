Labfront
========

`Labfront <https://www.labfront.com/>`_ supports several wearable-device
families and questionnaire data for research studies. Pywearable's
``LabfrontLoader`` reads Garmin exports from one or more extracted Labfront
directories::

    from pywearable.loader import LabfrontLoader

    loader = LabfrontLoader("path/to/export")

Pass a list when a study is split across several directories::

    loader = LabfrontLoader(["path/to/export-1", "path/to/export-2"])

The loader stores a local file index beside the export to avoid scanning every
file on each construction. Set ``reindex_files=True`` after adding or replacing
source files::

    loader = LabfrontLoader("path/to/export", reindex_files=True)

The portable methods and constructor are documented under
:class:`pywearable.loader.LabfrontLoader`.
