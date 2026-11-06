LifeSnaps
=========

The LifeSnaps dataset [1]_ can be loaded in two different ways.
One way is to use the ``LifeSnapsMongoDBLoader`` to load data 
directly from the LifeSnaps MongoDB database. To do so, you need to:


1. Install `MongoDB Community Edition <https://www.mongodb.com/try/download/community>`_
2. Install `MongoDB Command Line Database Tools <https://www.mongodb.com/try/download/database-tools>`_
3. *(Optional)* Install `MongoDB Compass <https://www.mongodb.com/products/tools/compass>`_
4. Download the zip file from `Zenodo <https://zenodo.org/record/7229547>`_
5. Extract the zip file in a convenient location (you will need the path of the contained bson files in the next step)
6. On a terminal window, run the following commands, by replacing ``<mongodb host>`` with the host name of your MongoDB (by default ``localhost``), ``mongodb port`` with the port of your MongoDB on your host (by default: ``27017``) and the bson files that you have previously extractd (``<path to file *.bson>``) :

    - ``mongorestore --host <mongodb host>:<mongo db port> -d rais_anonymized -c fitbit <path to file fitbit.bson>``
    - ``mongorestore --host <mongodb host>:<mongo db port> -d rais_anonymized -c sema <path to file sema.bson>``
    - ``mongorestore --host <mongodb host>:<mongo db port> -d rais_anonymized -c surveys <path to file surveys.bson>``
7. Running the previous commands load the backup ``bson`` files into your MongoDB. You can then use MongoDB Compass to query the database. This is what you should see in your MongoDB Compass window if everything was set up properly 

With the MongoDB properly setup::

    from pywearable.loader import LifeSnapsMongoDBLoader

    loader = LifeSnapsMongoDBLoader()

Once you have the LifeSnaps dataset loaded in your MongoDB, you can convert it in Parquet format using 
pywearable. To do so, run the following command on your terminal::

    pywearable convert lifesnaps-to-parquet OUTPUT_FOLDER


After converting it into pywearable's local Parquet layout, you can use ``LifeSnapsParquetLoader`` to load the data
in Parquet format::

    from pywearable.loader import LifeSnapsParquetLoader

    loader = LifeSnapsParquetLoader("path/to/lifesnaps-parquet")

Install the ``parquet`` optional dependency before using this loader::

    python -m pip install ".[parquet]"

The loader exposes timestamped Fitbit activity and heart-rate samples together
with provider-computed sleep, HRV, SpO2, and stress features where present in
the converted dataset. Missing source tables return empty canonical frames,
preserving the distinction between unavailable data and measured zeros.

See :class:`pywearable.loader.LifeSnapsParquetLoader` and :class:`pywearable.loader.LifeSnapsMongoDBLoader` for the loaders' API.

References
----------

.. [1] Yfantidou, Sofia, et al. "LifeSnaps, a 4-month multi-modal dataset
   capturing unobtrusive snapshots of our lives in the wild." *Scientific
   Data* 9.1 (2022): 663.
