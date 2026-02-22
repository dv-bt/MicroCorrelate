import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "MicroCorrelate"
author = "dv-bt"
release = "0.1.1"
copyright = f"2026, {author}"

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_parser",
]

# sphinx-autoapi
autoapi_dirs = ["../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "groupwise"
autoapi_add_toctree_entry = False

# MyST
myst_enable_extensions = ["colon_fence"]

# Theme
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/dv-bt/MicroCorrelate",
    "logo": {"text": "MicroCorrelate"},
    "show_toc_level": 2,
}
html_static_path = ["_static"]

exclude_patterns = ["_build"]


def _skip_class_attributes(app, what, name, obj, skip, options):
    """Skip auto-discovered class attributes; documented via Attributes sections."""
    if what == "attribute":
        return True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", _skip_class_attributes)
