# pywearable

Do you need to deal with wearable data? Then you are in the right place!
In this repository you will find a Python package that you can use to analyse all data collected with several data sources, with a particular focus on longitudinal data. 
At the moment, we support:
- [Labfront](https://labfront.com/)
- [LifeSnaps Dataset](https://zenodo.org/records/7229547)
- Tesserae
- Oura API v2 JSON exports (in development)
  
The aim of this Python package is to offer a series of functions for the loading and analysis of the data, easing the feature extraction pipeline by offering standardized functions for the loading of the data.

 Furthermore, the aim is to build a command line interface (CLI) for data extraction and a web-based dashboard for:
- visualization of physiological data
- automatic analysis of data with feature extraction
- summary creation and data reporting

## Installation
At this stage of development, the package is still not uploaded to PyPi, and thus can be only be installed through a local installation:

1. Clone the repository with `git clone https://github.com/MeDiTech-SUPSI/pywearable.git`
2. `cd pywearable`
3. `pip install --editable .[dev]`

## Documentation

The format for documentation is [numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html).
The documentation of the package can be built locally using [sphinx](www.sphinx-doc.org).
1. `cd docs`
2. `make html`
3. Open in your browser the file index.html that you can find in docs/_build/html

## Usage
The package was designed to be used in the most straightforward way:

```
import pywearable.loader

BASE_FOLDER = "..." # Path to folder with data downloaded from Labfront

labfront_loader = pywearable.loader.LabfrontLoader(BASE_FOLDER)
sleep_summaries = labfront_loader.load_sleep_summary("user-01")
```

For a local copy of Oura API v2 JSON files in
`participant/metric/*.json` directories:

```python
from pywearable.loader import OuraLoader
from pywearable import sleep, heart_rate

loader = OuraLoader("/path/to/oura-export")
user = loader.get_user_ids()[0]
nights = loader.load_sleep_summary(user)  # long_sleep only; local calendar day
sessions = loader.load_sleep_sessions(user)  # includes naps
night_features = sleep.get_sleep_statistics(
    loader, user_id=user, start_date=None, end_date=None,
    statistic=["total_sleep_time", "wake_after_sleep_onset"],
)
hourly_hr = heart_rate.get_heart_rate_statistics(
    loader, user_id=user, resolution="1h", statistic="heart_rate_mean",
)  # UTC windows
```

Oura nightly/five-minute HRV and nightly respiration are provider estimates,
not continuous sensor signals. Local-day HR statistics require timezone
information the copied HR export does not reliably provide. Generic Oura
activity-intensity and recovery mappings remain unfinished. Sleep loader
columns and extracted feature names now use `snake_case`.
With `sleep.get_sleep_statistics(..., statistic=None)`, the default result is
the shared night-level feature set; `getup_latency`, `validation`, and
`sleep_regularity_index` require explicit selection while their cross-loader
definitions are assessed.

## Contributing

Do you want to contribute to the development of pywearable? Then drop me an email at davide.marzorati@supsi.ch!


