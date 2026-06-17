r"""
Core sub-package for the **pyvisual** library.

This sub-package contains the primary classes and supporting infrastructure for
3D solar-physics visualization.  The key components are:

- :class:`~pyvisual.core.plot3d.Plot3d` — the main plotter, extending
  :class:`pyvista.Plotter` with spherical-coordinate awareness, observer controls,
  solar geometry helpers, and MHD dataset rendering.
- :class:`~pyvisual.core.mesh3d.SphericalMesh` — a :class:`pyvista.RectilinearGrid`
  subclass for data on spherical :math:`(r, \theta, \phi)` grids, with built-in
  arithmetic operators and NumPy ufunc support.
- :class:`~pyvisual.core.mesh3d.CartesianMesh` — a :class:`pyvista.StructuredGrid`
  subclass with an identical operator/ufunc interface for Cartesian :math:`(x, y, z)`
  grids.
- :mod:`~pyvisual.core.mixins` — the four mixin classes
  (:class:`~pyvisual.core.mixins.ObserverMixin`,
  :class:`~pyvisual.core.mixins.GeometryMixin`,
  :class:`~pyvisual.core.mixins.GridMeshMixin`,
  :class:`~pyvisual.core.mixins.StackMeshMixin`) that compose the full
  :class:`~pyvisual.core.plot3d.Plot3d` API.
- :mod:`~pyvisual.core.parsers` — input-normalisation utilities and the
  :func:`~pyvisual.core.parsers.parse_mesh_params` decorator.
- :mod:`~pyvisual.core.constants` — coordinate-frame aliases, physical constants,
  and scale-key mappings.
- :mod:`~pyvisual.core._typing` — type aliases and named tuples used throughout the
  package.
- :mod:`~pyvisual.core._styling` — immutable :class:`~types.MappingProxyType`
  dictionaries of per-render-type default kwargs.

On import this module attempts to load the PSI color theme for PyVista via
:func:`~pyvisual.utils.helpers.fetch_theme`.  If the theme file cannot be retrieved a
:class:`UserWarning` is emitted and PyVista's built-in
:class:`~pyvista.themes.DarkTheme` is used as a fallback.

Examples
--------
A minimal interactive session::

    import pyvisual as pv
    pl = pv.Plot3d()
    pl.add_sun()
    pl.show()

See Also
--------
:mod:`pyvisual.core.plot3d`
    Full API for the :class:`~pyvisual.core.plot3d.Plot3d` class.
:mod:`pyvisual.core.mesh3d`
    Mesh classes and polydata builder functions.
"""

from __future__ import annotations

import os
import warnings

import pyvista as pv

from pyvisual.utils.helpers import fetch_theme

# Load the global theme for PyVista
try:
	theme = fetch_theme()
	pv.global_theme.load_theme(str(theme))
except Exception as e:
	if not os.environ.get("SPHINX_GALLERY_BUILD"):
		warnings.warn(
			"Failed to load PyVisual theme. Using default PyVista theme. Error: " + str(e)
		)
	pv.global_theme.load_theme(pv.themes.DarkTheme())
