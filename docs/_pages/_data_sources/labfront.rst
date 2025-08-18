Labfront
========
`Labfront <www.labfront.com>`_ is a platform that allows researchers
to collect data from Garmin smartwatches, Dexcom G6 and Movesense
sensors. With this platform it is also possible to collect responses
to questionnaires and assign tasks to study participants, everything with
the same mobile app.

Labfront and pywearable
-----------------------
In order to use pywearable with Labfront data, you simply need to
download Labfront data and extract the zip file into a location of 
your choice::

    import pywearable.loader
    data_folder = "..." # Path to extracted zip file with Labfront data
    labfront_loader = pywearable.loader.LabfrontLoader(data_folder)

``pywearable`` will then scan the content of the ``data_folder`` and create an index
of files in order to speed up data access. This file index will be
saved into the data directory with the name **.pywearable_file_index.csv**.
When the ``labfront_loader`` will be instantiated again for the
same directory, the contents will not be scanned and the already
available file index will be used. In case you want to force a reindex
of the data directory (e.g., data were added to the same directory) then 
you can set the parameter ``force_reindex`` to ``True``::

    import pywearable.loader
    labfront_loader = pywearable.loader.LabfrontLoader(data_folder, force_reindex=True)

Multiple Data Folder
--------------------

Labfront Data Structure
------------------------

Helper Scripts
--------------
pywearable offers two main scripts 

Available Metrics
-----------------

.. list-table:: Labfront Metrics
    :header-rows: 1
    
    * - Metric
      - Data Source
      - Description
      - Function(s)
    * - Heart rate
      - Garmin Health API
      - Heart rate ($\DeltaT$=15s)
      - ``loader.load_garmin_connect_heart_rate()``
        ``loader.load_heart_rate(source="health_api")``
    * - Heart rate
      - Garmin SDK
      - Heart rate
      - ``loader.load_garmin_device_heart_rate()``
        ``loader.load_heart_rate(source="sdk")``
