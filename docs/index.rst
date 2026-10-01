pywearable
==========

``pywearable`` turns wearable exports into analysis-ready trajectories for
activity, sleep, heart rate, heart-rate variability, respiration, oxygen
saturation, provider scores, and adherence.

A typical workflow has three steps: choose a loader, extract one or more
features at the resolution needed by the analysis, and inspect or visualize
the resulting trajectory. The user guide introduces each step with runnable
examples; the API reference lists every supported parameter.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   _pages/getting_started
   _pages/data_sources
   _pages/features
   _pages/cycles
   _pages/resolutions
   _pages/trajectories
   _pages/metrics
   _pages/visualization

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   _notebooks/analysis_examples
   _notebooks/custom_loader

.. toctree::
   :maxdepth: 2
   :caption: Reference

   _pages/api

Indices
========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
