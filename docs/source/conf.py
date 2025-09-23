# ruff: noqa
"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Think Reason Learn"
copyright = "2025, Vela Research"
author = "Vela Research"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "myst_parser",  # Enable Markdown with MyST
    "sphinxcontrib.autodoc_pydantic",  # Better Pydantic model rendering
]

# Make Sphinx include type hints alongside docstring descriptions
autodoc_typehints = "description"  # merge types into the description
autoclass_content = "both"  # include class and __init__ docstrings
autodoc_member_order = "bysource"
add_module_names = False  # cleaner object names in docs
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    # Do not include imported members to reduce duplicates across re-exports
    "imported-members": False,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_attr_types = True  # Include type info in attributes
napoleon_preprocess_types = True  # Preprocess types for better rendering
napoleon_use_param = True  # Use :param: for parameters
napoleon_use_rtype = True  # Use :rtype: for return types

# autodoc_pydantic settings to avoid duplication and combine fields
autodoc_pydantic_model_show_json = False  # Hide JSON model if not needed
autodoc_pydantic_model_show_config_summary = False  # Hide config
autodoc_pydantic_model_member_order = "bysource"  # Order by source code
autodoc_pydantic_field_list_style = "compact"  # Compact field lists
autodoc_pydantic_field_doc_policy = "description"
autodoc_pydantic_model_show_field_summary = False  # Avoid duplicate field entries

templates_path = ["_templates"]
exclude_patterns = []


# Allow both .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Furo theme options (optional cosmetics)
html_theme = "furo"  # Modern, responsive theme (pip install furo)
html_static_path = ["_static"]

# Custom JavaScript to make external links open in new tabs
html_js_files = [
    "external_links.js",
]
html_title = "Think Reason Learn"
html_short_title = "TRL"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2e7d32",
        "color-brand-content": "#1b5e20",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/vela-research/think-reason-learn",
            "html": (
                "<svg stroke='currentColor' fill='currentColor' stroke-width='0'"
                " viewBox='0 0 16 16' height='1.2em' width='1.2em'"
                " xmlns='http://www.w3.org/2000/svg'>"
                "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z'></path>"
                "</svg>"
            ),
        }
    ],
}
