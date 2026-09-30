Getting started
===============

``pywearable`` supports Python 3.10 and newer. The examples below use a
Labfront export, but the same domain functions work with every supported
loader.

Installation
------------

Install the project from a local checkout::

    git clone https://github.com/MeDiTech-SUPSI/pywearable.git
    cd pywearable
    python -m pip install .

Parquet-based loaders require the optional ``parquet`` dependency group::

    python -m pip install ".[parquet]"

The optional NeuroKit HRV backend is installed separately::

    python -m pip install ".[neurokit]"

Load data
---------

Choose the loader matching the source dataset and inspect its participants::

    from pywearable.loader import LabfrontLoader

    loader = LabfrontLoader("path/to/extracted/labfront-data")
    print(loader.get_user_ids())

All analysis bounds are half-open: ``start_date`` is included and
``end_date`` is excluded.

Calculate a trajectory
----------------------

Domain functions return a :class:`pandas.DataFrame` indexed by participant and
time. Feature values and their coverage are separate columns::

    from pywearable import activity

    steps = activity.get_activity_statistics(
        loader,
        statistic=activity.ACTIVITY_STATISTIC_STEPS,
        user_id="all",
        start_date="2026-01-01",
        end_date="2026-02-01",
        resolution="daily",
    )

    print(steps.columns)
    # activity_steps, activity_steps__coverage

Use a named getter when it reads more naturally. Named getters are permanent
convenience APIs and return the same canonical trajectory format::

    from pywearable import sleep

    total_sleep = sleep.get_total_sleep_time(
        loader,
        user_id="all",
        start_date="2026-01-01",
        end_date="2026-02-01",
    )

Visualize the result
--------------------

Generic plotting functions accept a trajectory directly::

    from pywearable import visualization

    visualization.plot_trajectory(
        steps,
        activity.ACTIVITY_STATISTIC_STEPS,
        user_id="participant-01",
        ylabel="Steps",
    )

See :doc:`statistics` for resolutions, hierarchy, and coverage semantics, and
:doc:`visualization` for the plotting API.
