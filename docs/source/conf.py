from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import pyvista as pv
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

try:
    # First try to run sphinx_build against installed dist
    # This is primarily included for nox-based doc builds
    import pyvisual
except ImportError:
    # Fallback: add project root to sys.path
    # This is included for local dev builds without install
    sys.path.insert(0, Path(__file__).resolve().parents[2].as_posix())
    import pyvisual

try:
    from pthree import build_node_tree, node_tree_to_dict
except ImportError:
    raise ImportError(
        "The 'pthree' package is required to build the documentation. "
        "Please install it via 'pip install pthree' and try again."
    )

# ------------------------------------------------------------------------------
# Project Information
# ------------------------------------------------------------------------------
project = "pyvisual"
author = "Predictive Science Inc"
copyright = f"{datetime.now():%Y}, {author}"
version = pyvisual.__version__
release = pyvisual.__version__

# ------------------------------------------------------------------------------
# General Configuration
# ------------------------------------------------------------------------------
extensions = []

# --- HTML Theme
_logo = "https://predsci.com/doc/assets/static/psi_logo.png"
html_favicon = _logo
html_logo = _logo
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "show_prev_next": False,
    "navigation_with_keys": False,
    "show_navbar_depth": 1,
    "max_navbar_depth": 6,
    "logo": {
        "text": f"{project} v{version}",
        "image_light": _logo,
        "image_dark": _logo,
    },
    'icon_links': [
        {
            'name': 'PSI Home',
            'url': 'https://www.predsci.com/',
            'icon': 'fa fa-home fa-fw',
            "type": "fontawesome",
        },
        {
            'name': 'Repository',
            'url': 'https://github.com/predsci/pyvisual',
            "icon": "fa-brands fa-github fa-fw",
            "type": "fontawesome",
        },
        {
            'name': 'Documentation',
            'url': 'https://predsci.com/doc/pyvisual',
            "icon": "fa fa-file fa-fw",
            "type": "fontawesome",
        },
        {
            'name': 'Contact',
            'url': 'https://www.predsci.com/portal/contact.php',
            'icon': 'fa fa-envelope fa-fw',
            "type": "fontawesome",
        },
    ],
}

# --- Python Syntax
add_module_names = False
python_maximum_signature_line_length = 80

# --- Templating
templates_path = ['_templates', ]

# ------------------------------------------------------------------------------
# Viewcode Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.viewcode")

viewcode_line_numbers = True

# ------------------------------------------------------------------------------
# Autosummary Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.autosummary")

root_package = 'pyvisual'
exclude_private = False
exclude_tests = True
exclude_dunder = True
sort_members = False
exclusions = [
    '_add_stack_set', 'render_scene', '_add_grid_set',
    'TWOPI', 'XYZ_PERMUTATIONS', 'RTP_PERMUTATIONS',
    r'parsers\._', r'mesh3d\._update', r'plot3d\.Plot3d\._'
]

node_tree = build_node_tree(root_package,
                            sort_members,
                            exclude_private,
                            exclude_tests,
                            exclude_dunder,
                            exclusions)

autosummary_context = dict(pkgtree=node_tree_to_dict(node_tree))


# ------------------------------------------------------------------------------
# Inject mixin members into Plot3d's node tree entry
#
# Plot3d inherits its methods from four private mixin classes. pthree's static
# analysis only sees methods defined directly on Plot3d, so we post-process the
# node tree dict here to add the mixin members before the Jinja template runs.
# ------------------------------------------------------------------------------
import inspect as _inspect
from pyvisual.core.mixins import StackMeshMixin, GridMeshMixin, ObserverMixin, GeometryMixin

_PLOT3D_MIXINS = [StackMeshMixin, GridMeshMixin, ObserverMixin, GeometryMixin]
_PLOT3D_PATH = ['core', 'plot3d', 'Plot3d']


def _find_node(tree: dict, path: list) -> dict | None:
    node = tree
    for seg in path:
        children = node.get('children', [])
        node = next((c for c in children if c.get('name') == seg), None)
        if node is None:
            return None
    return node


def _inject_mixin_members(pkgtree: dict, class_path: list, mixins: list) -> None:
    node = _find_node(pkgtree, class_path)
    if node is None:
        return
    existing = {c['name'] for c in node.get('children', [])}
    for mixin in mixins:
        for name, member in _inspect.getmembers(mixin):
            if name in existing or name.startswith('__'):
                continue
            if isinstance(member, property):
                kind = 'property'
            elif callable(member):
                kind = 'method'
            else:
                continue
            node.setdefault('children', []).append({'name': name, 'kind': kind})
            existing.add(name)


_inject_mixin_members(autosummary_context['pkgtree'], _PLOT3D_PATH, _PLOT3D_MIXINS)

# ------------------------------------------------------------------------------
# Autodoc Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.autodoc")

autodoc_typehints = "none"
autodoc_member_order = 'bysource'
autodoc_default_options = {
    "show-inheritance": False,
}

# ------------------------------------------------------------------------------
# Numpydoc Configuration
# ------------------------------------------------------------------------------
extensions.append("numpydoc")

numpydoc_xref_param_type = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_xref_ignore = {
    "optional", "default", "of", "or",
    # pyvista internal type not in public intersphinx inventory
    "PlottableType",
}
numpydoc_xref_aliases = {
    # ------------------------------------------------------------------
    # NumPy
    # ------------------------------------------------------------------
    "np.ndarray": "numpy.ndarray",
    "ArrayLike": "numpy.typing.ArrayLike",

    # ------------------------------------------------------------------
    # Python stdlib
    # ------------------------------------------------------------------
    "Path": "pathlib.Path",
    "PathLike": "os.PathLike",
    "Callable": "collections.abc.Callable",
    "Iterable": "collections.abc.Iterable",
    "Number": "numbers.Number",
    "MappingProxyType": "types.MappingProxyType",

    # ------------------------------------------------------------------
    # PyVista — short-form (pv.*) and fully-qualified aliases
    # ------------------------------------------------------------------
    "pv.Actor": "pyvista.Actor",
    "pv.DataSet": "pyvista.DataSet",
    "pv.Grid": "pyvista.Grid",
    "pv.ImageData": "pyvista.ImageData",
    "pv.MultiBlock": "pyvista.MultiBlock",
    "pv.PartitionedDataSet": "pyvista.PartitionedDataSet",
    "pv.Plotter": "pyvista.Plotter",
    "pv.PolyData": "pyvista.PolyData",
    "pv.RectilinearGrid": "pyvista.RectilinearGrid",
    "pv.StructuredGrid": "pyvista.StructuredGrid",

    # ------------------------------------------------------------------
    # SciPy
    # ------------------------------------------------------------------
    "RegularGridInterpolator": "scipy.interpolate.RegularGridInterpolator",

    # ------------------------------------------------------------------
    # Third-party (pooch, requests)
    # ------------------------------------------------------------------
    "RequestException": "requests.exceptions.RequestException",
    "HashMismatchError": "pooch.exceptions.HashMismatchError",
    "Pooch": "pooch.Pooch",

    # ------------------------------------------------------------------
    # pyvisual — primary classes
    # ------------------------------------------------------------------
    "Plot3d": "pyvisual.core.plot3d.Plot3d",
    "SphericalMesh": "pyvisual.core.mesh3d.SphericalMesh",
    "CartesianMesh": "pyvisual.core.mesh3d.CartesianMesh",

    # ------------------------------------------------------------------
    # pyvisual — type aliases (pyvisual.core._typing)
    # ------------------------------------------------------------------
    "PathType": "pyvisual.core._typing.PathType",
    "FlColorType": "pyvisual.core._typing.FlColorType",
    "MeshFramesType": "pyvisual.core._typing.MeshFramesType",
    "PlotterFramesType": "pyvisual.core._typing.PlotterFramesType",
    "SurfaceReconstructionType": "pyvisual.core._typing.SurfaceReconstructionType",

    # ------------------------------------------------------------------
    # pyvisual — named tuples (pyvisual.core._typing)
    # ------------------------------------------------------------------
    "SolarCoordinate": "pyvisual.core._typing.SolarCoordinate",
    "SphericalCoordinate": "pyvisual.core._typing.SphericalCoordinate",
    "CartesianCoordinate": "pyvisual.core._typing.CartesianCoordinate",
    "ObserverView": "pyvisual.core._typing.ObserverView",
    "ObserverOrientation": "pyvisual.core._typing.ObserverOrientation",
}

# ------------------------------------------------------------------------------
# Intersphinx Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.intersphinx")

DOCS = Path(__file__).resolve().parents[1]
INV = DOCS / "_intersphinx"
intersphinx_cache_limit = 30
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        # (INV / "python-objects.inv").as_posix(),
        None
    ),
    "pyvista": (
        "https://docs.pyvista.org/",
        # (INV / "python-objects.inv").as_posix(),
        None
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        # (INV / "numpy-objects.inv").as_posix(),
        None
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        # (INV / "scipy-objects.inv").as_posix(),
        None
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        # (INV / "matplotlib-objects.inv").as_posix(),
        None
    ),
    "pooch": (
        "https://www.fatiando.org/pooch/latest/",
        # (INV / "pooch-objects.inv").as_posix(),
        None
    ),
    "h5py": (
        "https://docs.h5py.org/en/stable/",
        # (INV / "h5py-objects.inv").as_posix(),
        None
    ),
    "sunpy": (
        "https://docs.sunpy.org/en/stable/",
        # (INV / "sunpy-objects.inv").as_posix(),
        None
    ),
    "astropy": (
        "https://docs.astropy.org/en/stable/",
        # (INV / "astropy-objects.inv").as_posix(),
        None
    ),
    "mapflpy": (
        "https://predsci.com/doc/mapflpy/",
        None
    ),
    "psi-io": (
        "https://predsci.com/doc/psi-io/",
        None
    ),
}

# ------------------------------------------------------------------------------
# Pyvista Configuration
# ------------------------------------------------------------------------------

extensions.extend([
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
])

os.environ["PYVISTA_BUILDING_GALLERY"] = "True"
pv.global_theme.interactive = False
pv.BUILDING_GALLERY = True
pv.OFF_SCREEN = True

# ------------------------------------------------------------------------------
# Sphinx-Gallery Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx_gallery.gen_gallery")

import matplotlib
matplotlib.use("Agg")
os.environ.setdefault('SPHINX_GALLERY_BUILD', '1')

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["gallery"],
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "filename_pattern": r"\.py$",
    "plot_gallery": False,
    "run_stale_examples": False,
    "matplotlib_animations": True,
    "image_scrapers": (DynamicScraper(), 'matplotlib'),
}

# ------------------------------------------------------------------------------
# Sphinx Copy Button Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx_copybutton")

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
