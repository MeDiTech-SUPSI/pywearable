Labfront
========

`Labfront <https://www.labfront.com/>`_ collects wearable and questionnaire
data for research studies. ``LabfrontLoader`` reads one or more extracted
Labfront export directories::

    from pywearable.loader import LabfrontLoader

    loader = LabfrontLoader("path/to/export")

Pass a list when a study is split across several directories::

    loader = LabfrontLoader(["path/to/export-1", "path/to/export-2"])

The loader stores a local file index beside the export to avoid scanning every
file on each construction. Set ``reindex_files=True`` after adding or replacing
source files::

    loader = LabfrontLoader("path/to/export", reindex_files=True)

Unresolved conflicting sleep-stage snapshots are omitted by default while the
nightly summary remains available. Set ``sleep_stage_on_conflict="raise"`` to
make such a conflict fail explicitly.

The portable methods and constructor are documented under
:class:`pywearable.loader.LabfrontLoader`.
