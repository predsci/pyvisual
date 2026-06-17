"""
**pyvisual** — 3D solar-physics visualization for MHD datasets.

**pyvisual** is developed and maintained by `Predictive Science Inc. (PSI)
<https://www.predsci.com/>`_ and provides an interactive 3D visualization layer
for solar and magnetohydrodynamic (MHD) model output.  It wraps
:class:`pyvista.Plotter` with spherical-coordinate awareness, observer controls,
and rendering utilities tuned to the PSI data ecosystem (psi-io, mapflpy).

Quick start
-----------
.. code-block:: python

    >>> from pyvisual import Plot3d
    >>>
    >>> plotter = Plot3d()
    >>> plotter.add_sun()
    >>> plotter.show()

Sub-packages
------------
:mod:`pyvisual.core`
    The :class:`~pyvisual.core.plot3d.Plot3d` plotter, mesh classes
    (:class:`~pyvisual.core.mesh3d.SphericalMesh`,
    :class:`~pyvisual.core.mesh3d.CartesianMesh`), mixin classes, parsers,
    constants, type aliases, and styling defaults.

:mod:`pyvisual.utils`
    Coordinate-transform geometry utilities
    (:mod:`~pyvisual.utils.geometry`) and miscellaneous helpers
    (:mod:`~pyvisual.utils.helpers`), including bundled-asset resolution via
    :func:`~pyvisual.utils.helpers.fetch_theme`.  Sample MHD datasets for the
    example gallery are fetched separately with the external
    :mod:`psi_data` package (:func:`psi_data.fetch_mas_data`).

See Also
--------
`PSI data conventions <https://predsci.com/doc/psi-io/guide/index.html>`_
    Documentation for the HDF file format and coordinate conventions used by
    PSI model output.
`mapflpy <https://predsci.com/doc/mapflpy/>`_
    Fieldline tracing library that integrates with **pyvisual**.
"""

from pyvisual.core.plot3d import Plot3d
from pyvisual.core.mesh3d import SphericalMesh, CartesianMesh

try:
	from importlib.metadata import version as _pkg_version
	from importlib.metadata import PackageNotFoundError
	from pathlib import Path

	__version__ = _pkg_version("psi-pyvisual")  # type: ignore[assignment]
except PackageNotFoundError as e:  # dev/editable without metadata
	try:
		import tomllib  # Python 3.11+
	except ModuleNotFoundError:  # pragma: no cover
		import tomli as tomllib  # pip install tomli

	pyproject = Path(__file__).parents[1].resolve() / "pyproject.toml"
	data = tomllib.loads(pyproject.read_text())

	project_version = data.get("project", {}).get("version", "0+unknown")
	project_version = project_version.replace('"', "").replace("'", "")
	__version__ = project_version
