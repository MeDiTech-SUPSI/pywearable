# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

try:
    import pypandoc

    pandoc_path = Path(pypandoc.get_pandoc_path())
    if sys.platform == "win32" and not pandoc_path.exists():
        pandoc_path = pandoc_path.with_suffix(".exe")
    if pandoc_path.exists():
        os.environ["PATH"] = (
            str(pandoc_path.parent) + os.pathsep + os.environ.get("PATH", "")
        )
except ImportError:
    pass

sys.path.insert(0, os.path.abspath("../src/"))

project = "pywearable"
copyright = "2023-2026, Davide Marzorati"
author = "pywearable contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_ci_index.rst"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_timeout = 10

# Autodoc renders runtime annotation paths and free-form NumPy-doc type labels
# as references. Keep nitpicky mode strict for project symbols while ignoring
# third-party implementation paths and descriptive pseudo-types that have no
# stable inventory target.
nitpick_ignore_regex = [
    ("py:class", r"pandas\.(?:_libs|core)\..*"),
    ("py:class", r"matplotlib\.axes\._axes\.Axes"),
    ("py:class", r"pd\..*"),
    (
        "py:class",
        r"(?:sequence|optional|date-like|timestamp-like|timedelta|nested sequence)",
    ),
    ("py:class", r"default .*"),
    ("py:class", r"\{.*"),
    ("py:class", r'".*'),
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"
nbsphinx_execute = "never"
