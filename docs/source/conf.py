# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: Apache-2.0

project = "RAFT"
copyright = "2026 ETH Zurich"
author = "Geothermal Energy and Geofluids Group, ETH Zurich"

extensions = [
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",  # renders $...$ and $$...$$ as math, used throughout theory.md
    "amsmath",  # renders \begin{split}/\begin{cases}/\begin{bmatrix} environments
    "colon_fence",
]

myst_heading_anchors = 4  # resolves the existing GitHub-style #section links in README.md/theory.md

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"

# Sphinx's default MathJax config already matches the \(...\)/\[...\] delimiters
# that myst-parser emits in the HTML output — no custom mathjax3_config needed.
