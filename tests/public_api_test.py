import importlib
import inspect

import pytest

CURATED_MODULES = (
    "pywearable.activity",
    "pywearable.adherence",
    "pywearable.cycle",
    "pywearable.heart_rate",
    "pywearable.hrv",
    "pywearable.loader",
    "pywearable.loader.base",
    "pywearable.loader.garmin",
    "pywearable.loader.labfront",
    "pywearable.loader.labfront.loader",
    "pywearable.loader.lifesnaps",
    "pywearable.loader.oura",
    "pywearable.loader.oura.loader",
    "pywearable.loader.tesserae",
    "pywearable.loader.tesserae.tesserae_csv",
    "pywearable.loader.tesserae.tesserae_parquet",
    "pywearable.provider_features",
    "pywearable.respiration",
    "pywearable.schema",
    "pywearable.sleep",
    "pywearable.trajectory",
    "pywearable.utils",
    "pywearable.visualization",
)

PORTABLE_LOADER_METHODS = (
    "get_user_ids",
    "resolve_user_ids",
    "load_sleep_summary",
    "load_sleep_stage",
    "load_sleep_sessions",
    "load_sleep_session_stages",
    "load_heart_rate",
    "load_bbi",
    "load_hrv_samples",
    "load_hrv_summary",
    "load_activity_samples",
    "load_activity_summary",
    "load_respiration",
    "load_respiration_summary",
    "load_pulse_ox",
    "load_spo2_summary",
    "load_provider_feature",
)


@pytest.mark.parametrize("module_name", CURATED_MODULES)
def test_curated_exports_resolve(module_name):
    module = importlib.import_module(module_name)

    assert isinstance(module.__all__, list)
    assert len(module.__all__) == len(set(module.__all__))
    assert all(not name.startswith("_") for name in module.__all__)
    assert all(hasattr(module, name) for name in module.__all__)


@pytest.mark.parametrize("module_name", CURATED_MODULES)
def test_non_private_callables_are_classified(module_name):
    module = importlib.import_module(module_name)
    defined_callables = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and getattr(value, "__module__", None) == module_name
    }
    exported_callables = {name for name in module.__all__ if name in defined_callables}

    assert defined_callables == exported_callables


@pytest.mark.parametrize("method_name", PORTABLE_LOADER_METHODS)
def test_portable_loader_methods_are_documented(method_name):
    base_loader = importlib.import_module("pywearable.loader.base").BaseLoader

    assert getattr(base_loader, method_name).__doc__
