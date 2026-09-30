Oura
====

``OuraLoader`` reads participant-level Oura API v2 response pages from a local
directory. The expected layout is::

    data/
        participant-01/
            sleep/
                data.json
            daily_sleep/
                data.json
            heartrate/
                data.json

Each JSON file contains the usual Oura response object with a ``data`` list.
Additional pagination files can live in the same endpoint directory::

    from pywearable.loader import OuraLoader

    loader = OuraLoader("path/to/oura-data")

Daily and nightly records use Oura's participant-local ``day`` label.
Subdaily timestamps are normalized to canonical UTC and local-time columns.
Repeated identical records are collapsed; conflicting records with the same
provider identifier raise an error.

See :class:`pywearable.loader.OuraLoader` for the loader API.
