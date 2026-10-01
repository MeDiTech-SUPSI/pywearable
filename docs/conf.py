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
copyright = "2023-2026, pywearable contributors"
author = "pywearable contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"
nbsphinx_execute = "never"
