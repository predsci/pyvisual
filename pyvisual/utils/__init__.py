r"""Utility sub-package for pyvisual.

This sub-package collects self-contained helper modules that support the
core rendering and data-access layers of :mod:`pyvisual`.  Nothing here
depends on :mod:`pyvisual.core`; the modules can be imported and used
independently.

Modules
-------

:mod:`~pyvisual.utils.geometry`
    Low-level mathematical building blocks for working with solar-physics
    coordinate systems.  Covers:

    - Scalar coordinate transforms between Cartesian :math:`(x, y, z)` and
      the PSI spherical convention :math:`(r, \theta, \phi)` (colatitude).
    - Vector-basis rotations for expressing field components in either frame.
    - Rigid-body rotation matrices about each Cartesian axis.
    - Solar line-of-sight geometry: Thomson sphere intersection,
      impact-parameter ↔ angle conversions, angle wrapping.
    - Ephemeris queries via the JPL Horizons service (requires an internet
      connection and :mod:`astroquery`).
    - Sphere-sampling utilities: Fibonacci lattice and local point-mesh
      generation.
    - Camera roll relative to solar north.
    - Convenience array partials: :data:`~pyvisual.utils.geometry.ij_meshgrid`
      and :data:`~pyvisual.utils.geometry.moveaxis_to_start`.

:mod:`~pyvisual.utils.helpers`
    Minimal helpers with no dependencies beyond the standard library and
    :mod:`numpy`.  Exposes :func:`~pyvisual.utils.helpers.atleast_1dnull`, a
    ``None``-preserving wrapper around :func:`numpy.atleast_1d` used throughout
    the core parsing and mesh-construction layers, and
    :func:`~pyvisual.utils.helpers.fetch_theme`, which resolves the bundled
    PyVista colour-theme file.

.. note::
    Sample MHD datasets used by the example gallery are fetched with the
    external `psi-data-utils <https://pypi.org/project/psi-data-utils/>`_
    package (import name :mod:`psi_data`); see
    :func:`psi_data.fetch_mas_data`.  Install it via the ``data`` extra.

Notes
-----
None of the modules in this sub-package are imported automatically into the
top-level :mod:`pyvisual` namespace.  Import them explicitly as needed:

.. code-block:: python

    from pyvisual.utils.geometry import spherical_to_cartesian
    from pyvisual.utils.helpers import fetch_theme

See Also
--------
:mod:`pyvisual.core` : Main rendering classes and mesh infrastructure.
"""
