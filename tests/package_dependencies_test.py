from importlib.metadata import requires

from packaging.requirements import Requirement


def test_duckdb_is_a_required_runtime_dependency():
    dependencies = [Requirement(value) for value in requires("pywearable")]

    assert any(
        dependency.name == "duckdb" and dependency.marker is None
        for dependency in dependencies
    )
