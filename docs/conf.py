# -*- coding: utf-8 -*-
import tlcpack_sphinx_addon

# -- General configuration ------------------------------------------------

# General information about the project.
project = "mlc-llm"
author = "MLC LLM Contributors"
copyright = "2023, %s" % author

# Version information.

version = "0.1.0"
release = "0.1.0"

extensions = ['sphinx_tabs.tabs', 'sphinx_toolbox.collapse']

source_suffix = [".rst"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme is set by the make target
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

templates_path = []

html_static_path = []

footer_copyright = "© 2022 MLC LLM"
footer_note = " "

html_logo = "_static/img/mlc-logo-with-text-landscape.svg"

html_theme_options = {
    "logo_only": True,
}

header_links = [
    ("Home", "https://mlc.ai/mlc-llm"),
    ("Github", "https://github.com/mlc-ai/mlc-llm"),
    ("Discord Server", "https://discord.gg/9Xpy2HGBuD"),
]

header_dropdown = {
    "name": "Other Resources",
    "items": [
        ("MLC Course", "https://mlc.ai/"),
        ("MLC Blog", "https://mlc.ai/blog"),
        ("Web LLM", "https://mlc.ai/web-llm"),
    ],
}

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "header_dropdown": header_dropdown,
    "display_github": True,
    "github_user": "mlc-ai",
    "github_repo": "mlc-llm",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
    # "header_logo": "/path/to/logo",
    # "header_logo_link": "",
    # "version_selecter": "",
}


# add additional overrides
templates_path += [tlcpack_sphinx_addon.get_templates_path()]
html_static_path += [tlcpack_sphinx_addon.get_static_path()]
