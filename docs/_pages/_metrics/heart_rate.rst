heart-rate
================

The heart rate metric contains heart rate data
expressed in beats per minute. 
The required fields for this metric are the following ones:

.. list-table:: Heart rate fields
    :header-rows: 1
    
    * - Name
      - Description
      - Data type
    * - utc_time
      - Timezone offset
      - :class:`int` or :class:`str` or :class:`pandas.Timedelta`
    * - timezone
      - Unix timestamp in milliseconds
      - ``pywearable.constants._UNIXTIMESTAMP_IN_MS_COL``
      - :class:`int`
    * - local_time
      - Time in `ISO <https://en.wikipedia.org/wiki/ISO_8601>`_ format
      - ``pywearable.constants._ISODATE_COL``
      - :class:`pd.Timestamp`
    * - heart_rate
      - Heart rate values
      - ``pywearable.constants._BEATS_PER_MINUTE_COL``
      - :class:`float`
