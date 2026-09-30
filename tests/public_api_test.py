import importlib
import inspect

import pytest


CURATED_MODULES = (
    "pywearable.activity",
    "pywearable.adherence",
    "pywearable.constants",
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
    "pywearable.loader.sphyncs",
    "pywearable.loader.sphyncs.loader",
    "pywearable.loader.tesserae",
    "pywearable.loader.tesserae.tesserae_csv",
    "pywearable.loader.tesserae.tesserae_parquet",
    "pywearable.provider_features",
    "pywearable.respiration",
    "pywearable.schema",
    "pywearable.sleep",
    "pywearable.trajectory",
    "pywearable.utils",
    "pywearable.utils.legacy",
    "pywearable.visualization",
)

INTERNAL_NON_PRIVATE_CALLABLES = {
    "pywearable.loader.lifesnaps": {"LifeSnapsMongoDBLoader"},
    "pywearable.schema": {"coalesce_sleep_stage_runs"},
    "pywearable.sleep": {"compute_sleep_validation"},
}


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
    exported_callables = {
        name for name in module.__all__ if name in defined_callables
    }

    assert defined_callables == exported_callables | INTERNAL_NON_PRIVATE_CALLABLES.get(
        module_name, set()
    )


def test_loader_package_exports_only_first_release_loaders():
    loader = importlib.import_module("pywearable.loader")

    assert set(loader.__all__) == {
        "BaseLoader",
        "LabfrontLoader",
        "LifeSnapsParquetLoader",
        "OuraLoader",
        "TesseraeCSVLoader",
        "TesseraeParquetLoader",
    }
    assert "LifeSnapsMongoDBLoader" not in loader.__all__
    assert "SphyncsDataLoader" not in loader.__all__
    assert loader.LifeSnapsMongoDBLoader.__module__ == "pywearable.loader.lifesnaps"
    assert loader.SphyncsDataLoader.__module__ == "pywearable.loader.sphyncs.loader"


def test_experimental_sphyncs_loader_remains_available_from_its_subpackage():
    sphyncs = importlib.import_module("pywearable.loader.sphyncs")

    assert sphyncs.__all__ == ["SphyncsDataLoader"]
    assert sphyncs.SphyncsDataLoader.__module__ == "pywearable.loader.sphyncs.loader"
