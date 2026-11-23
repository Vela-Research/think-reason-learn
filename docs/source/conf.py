# ruff: noqa
"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../.."))

project = "Think Reason Learn"
copyright = f"{date.today().year}, Vela Research"
author = "Vela Research"
release = "0.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# Make Sphinx include type hints alongside docstring descriptions
autodoc_typehints = "description"
autosummary_generate = True
typehints_use_signature = False
typehints_fully_qualified = False
autoclass_content = "class"
autodoc_member_order = "groupwise"
python_use_unqualified_type_names = True
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
    "show-inheritance": True,
    "imported-members": False,
}


# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_attr_types = False
napoleon_preprocess_types = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# autodoc_pydantic settings to avoid duplication and combine fields
autodoc_pydantic_model_member_order = "groupwise"
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_field_list_style = "compact"
autodoc_pydantic_field_doc_policy = "description"
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = True

# MyST niceties
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

myst_url_schemes = ("http", "https", "mailto")

nitpicky = True

# PEP 695 type aliases, private modules, and types the inventories name differently.
nitpick_ignore_regex = [
    ("py:class", r".*TypeAliasType"),
    ("py:class", r"think_reason_learn\..*\._.*"),
    ("py:class", r"[A-Z][A-Za-z]*"),  # unqualified type alias names
    ("py:class", r"graphviz\..*"),
    ("py:class", r"(pathlib\._local\.Path|numpy\.(float|int)64|pandas\.core\..*)"),
    ("std:term", r"meta-estimator"),
    ("std:label", r"metadata_routing"),
    ("py:func", r"sklearn\.set_config"),
]

nitpick_ignore = [
    ("py:class", "T"),
    ("py:class", "S"),
    ("py:class", "TypeAliasForwardRef"),
    ("py:data", "typing.Union"),
]


templates_path = ["_templates"]
exclude_patterns = []


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["trl-fonts.css", "trl-theme.css"]

# Custom JavaScript to make external links open in new tabs
html_js_files = [
    "external_links.js",
]
html_title = "Think, Reason, Learn docs"
html_show_sourcelink = False
html_sidebars = {"**": ["docs-toc.html"]}
html_short_title = "TRL"
html_theme_options = {
    "logo": {
        "image_light": "_static/trl-mark.svg",
        "image_dark": "_static/trl-mark-dark.svg",
        "text": "Think, Reason, Learn",
        "alt_text": "Think, Reason, Learn home",
        "link": "https://thinkreasonlearn.com/",
    },
    "icon_links_label": "Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/vela-research/think-reason-learn",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "navbar_center": ["site-nav"],
    "navbar_end": ["search-button", "theme-switcher"],
    "navbar_persistent": [],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    "navigation_depth": 3,
    "footer_start": ["trl-footer"],
    "footer_center": [],
    "footer_end": [],
}
