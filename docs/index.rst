pywearable
==========

``pywearable`` loads wearable data from supported providers, normalizes it
into shared schemas, calculates comparable feature trajectories, and provides
visualizations for individual and cohort monitoring.

The library keeps source loading, feature calculation, and visualization
separate. Provider loaders translate raw exports into canonical tables;
domain modules produce trajectories with explicit coverage; plotting functions
consume those trajectories without silently recalculating them.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   _pages/getting_started
   _pages/data_sources
   _pages/metrics
   _pages/statistics
   _pages/visualization

.. toctree::
   :maxdepth: 2
   :caption: Reference

   _pages/api

Indices
========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
