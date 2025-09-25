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
release = "0.1.0"

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

# autodoc_pydantic settings to avoid duplication and combine fields
autodoc_pydantic_model_member_order = "groupwise"
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_field_list_style = "compact"
autodoc_pydantic_field_doc_policy = "description"
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = True

# autodoc_type_aliases = {
#     "LLMChoiceModel": "think_reason_learn.core.llms._schemas.LLMChoiceModel",
#     "LLMChoiceDict": "think_reason_learn.core.llms._schemas.LLMChoiceDict",
#     "LLMChoice": "think_reason_learn.core.llms._schemas.LLMChoice",
#     "think_reason_learn.core.llms._schemas.LLMChoice": "LLMChoice",
#     "think_reason_learn.core.llms._schemas.LLMChoiceModel": "LLMChoiceModel",
#     "think_reason_learn.core.llms._schemas.LLMChoiceDict": "LLMChoiceDict",

# }

# MyST niceties
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

myst_url_schemes = ("http", "https", "mailto")

nitpicky = True

nitpick_ignore = [
    ("py:class", "anthropic.NotGiven"),
    ("py:class", "openai.NotGiven"),
    ("py:class", "T"),
    ("py:obj", "think_reason_learn.core.llms._schemas.T"),
    ("py:class", "think_reason_learn.core._singleton.T"),
    ("py:class", "think_reason_learn.core.llms._schemas.T"),
    ("py:class", "TypeAliasForwardRef"),
    ("py:class", "LLMChoiceModel"),
    ("py:class", "asyncio.locks.Lock"),
]

nitpick_ignore_regex = [
    (r"py:class", r"DataFrame containing"),
    (r"py:class", r"done based on (prediction|semantic)"),
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
html_css_files = ["custom.css"]

# Custom JavaScript to make external links open in new tabs
html_js_files = [
    "external_links.js",
]
html_title = "Think Reason Learn"
html_short_title = "TRL"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
        "alt_text": "Think Reason Learn",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/vela-research/think-reason-learn",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_depth": 3,
}
