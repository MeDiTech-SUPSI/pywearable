LifeSnaps Dataset
=================

The `LifeSnaps dataset <https://www.nature.com/articles/s41597-022-01764-x>`_ is a dataset containing data 
from 71 participants collected over a period of 4 months, in which more than 35 different
data types were collected, including data from Fitbit Sense and questionnaires.
Pywearable allows to easily load and process the data of this dataset.

Data
----
In order to use ``pywearable`` with the LifeSnaps dataset, it is necessary to first
have access to the dataset itself. You can access the dataset at`the following `link 
<https://www.nature.com/articles/s41597-022-01764-x>`_ .

After registering, you will be able to download the required data.
Once extracted from the zip archive, you should obtain three files named fitbit.bson, sema.bson, and surveys.bson, which will be used to create MongoDB databases containing the corresponding information.
Additionally, you will find two .csv files located in a separate folder.

MongoDB
-------
Once you have downloaded the data, you must create the database before you can use it.
To do so, follow these steps:

1. Install MongoDB Community Edition: 
    https://www.mongodb.com/try/download/community

2. Install the MongoDB Command Line Database Tools: 
    https://www.mongodb.com/try/download/database-tools

3. Open a terminal window and run the following commands, replacing:
    <mongodb host> with the hostname of your MongoDB instance (default: localhost),

    <mongodb port> with the port number of your MongoDB instance (default: 27017),

    <path to file .bson> with the path to the corresponding .bson file you previously extracted.

.. code-block:: bash
    
    mongorestore --host <mongodb host>:<mongodb port> -d rais_anonymized -c fitbit <path to file fitbit.bson>
    mongorestore --host <mongodb host>:<mongodb port> -d rais_anonymized -c sema <path to file sema.bson>
    mongorestore --host <mongodb host>:<mongodb port> -d rais_anonymized -c surveys <path to file surveys.bson>

MongoDBCompass
---------------

LifeSnaps and pywearable
------------------------
To start using the library with the LifeSnaps data and verify that everything is working correctly, simply run the following code:

.. code-block:: python

    import pywearable.loader

    loader = pywearable.loader.LifeSnapsLoader(host="localhost", port=27017)

At this point, ``pywearable`` will be connected to the database you previously created, and it will be able to extract and process the data stored in it.

Below is an example of extracting a specific metric for a given user:

.. code-block:: python

    import pywearable.loader

    loader = pywearable.loader.LifeSnapsLoader(host="localhost", port=27017)
    uuid = "..." # example of user id
    respiration = loader.load_respiration(user_id=uuid)

You can replace `load_respiration` with any of the available functions to extract different types of data.  
For a complete list of available functions, please refer to the table below.

pymongo
-------

Available Metrics
-----------------
.. list-table:: LifeSnaps Metrics
    :header-rows: 1

    * - Metric
      - Data Source
      - Description
      - Function(s)
    * - Sleep summary
      - TBD
      - TDB
      - ``loader.load_sleep_summary()``
    * - Sleep stages
      - TBD
      - TBD
      - ``loader.load_sleep_stage()``
    * - Computed temperature
      - TBD
      - TBD
      - ``loader.load_computed_temperature()``
    * - Device Temperature
      - TBD
      - TBD
      - ``loader.load_device_temperature()``
    * - Daily Heart Rate Variability Summary
      - TBD
      - TBD
      - ``loader.load_hrv_summary()``
    * - Daily SpO2
      - TBD
      - TBD
      - ``loader.load_pulse_ox()``
    * - HRV Details
      - TBD
      - TBD
      - ``loader.load_hrv_details()``
    * - Profile
      - TBD
      - TBD
      - ``loader.load_profile()``
    * - Respiratory Rate Summary
      - TBD
      - TBD
      - ``loader.load_sleep_respiration()``
    * - Stress Score
      - TBD
      - TBD
      - ``loader.load_stress_score()``
    * - Afib ECG Readings
      - TBD
      - TBD
      - ``loader.load_ecg()``
    * - Calories
      - TBD
      - TBD
      - ``loader.load_calories()``
    * - Demographic VO2 Max
      - TBD
      - TBD
      - ``loader.load_vo2_max()``
    * - Distance
      - TBD
      - TBD
      - ``loader.load_distance()``
    * - Estimated Oxygen Variation
      - TBD
      - TBD
      - ``loader.load_oxygen_variation()``
    * - Heart Rate
      - TBD
      - TBD
      - ``loader.load_heart_rate()``
    * - Resting heart rate
      - TBD
      - TBD
      - TBD
    * - Very active minutes
      - TBD
      - TBD
      - ``loader.load_very_active()``
    * - Moderately active minutes
      - TBD
      - TBD
      - ``loader.load_moderately_active()``
    * - Lightly active minutes
      - TBD
      - TBD
      - ``loader.load_lightly_active()``
    * - Sedentary minutes
      - TBD
      - TBD
      - ``loader.load_sedentary()``
    * - Steps
      - TBD
      - TBD
      - ``loader.load_steps()``
    * - Water logs
      - TBD
      - TBD
      - ``loader.load_water_logs()``
