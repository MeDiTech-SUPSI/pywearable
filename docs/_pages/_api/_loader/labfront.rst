Labfront
========

The :class:`pywearable.loader.LabfrontLoader` can be used to load
data from `Labfront <https://www.labfront.com/>`__. In order to use 
this loader, you first need to download the data from your study
hosted on `Labfront <https://app.labfront.com/>`__, unzip the files
and then pass the path to the folder in which files have been 
unzipped to the loader::

   import pywearable.loader.LabfrontLoader
   data_path = "~/Users/user.name/Downloads/labfront_data/"
   loader = LabfrontLoader(data_path)

Repeated Garmin sleep-stage export snapshots are reconciled against the
corresponding sleep summary before stages are returned. A revised export is
not concatenated with an older snapshot. If snapshots disagree and there is
no matching summary to choose between them, loading raises an error instead
of returning an inflated stage trajectory.

.. autoclass:: pywearable.loader.LabfrontLoader
   :members:
