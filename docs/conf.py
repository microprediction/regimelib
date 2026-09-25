import os, sys
sys.path.insert(0, os.path.abspath(".."))
project = "regimelib"
author = "Peter Cotton"
copyright = "2026, Peter Cotton"
release = "0.1.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.viewcode"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_title = "regimelib documentation"
html_static_path = ["_static"]
highlight_language = "python"
templates_path = []
exclude_patterns = ["_build"]
