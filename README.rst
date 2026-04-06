========
pyvisual
========

**3D solar-physics visualization for MHD datasets**

.. image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue
   :alt: Python versions

.. image:: https://img.shields.io/badge/license-Apache%202.0-green
   :alt: License

.. image:: https://img.shields.io/badge/version-0.9.0-orange
   :alt: Version

**pyvisual** is developed and maintained by `Predictive Science Inc. (PSI)
<https://www.predsci.com/>`_.  It wraps `PyVista's
<https://docs.pyvista.org/>`_ ``Plotter`` class with specialized capabilities
for visualizing solar and magnetohydrodynamic (MHD) model output defined on
spherical coordinate systems.  The package is tightly coupled to the PSI data
ecosystem and is tuned for use with **psi-io** and **mapflpy**.

----

Features
--------

- **Spherical-coordinate rendering** — meshes and coordinate arrays in
  :math:`(r, \theta, \phi)` are converted to Cartesian automatically before
  being passed to the VTK render pipeline.
- **Structured-grid slices** — add 1-D line slices, 2-D surface slices, and
  3-D volume slices directly from independent axis arrays.
- **Fieldline rendering** — visualize open/closed magnetic fieldline topology
  with polarity coloring or random hue assignment.
- **Observer controls** — set and query camera position, focal point,
  position angle, and line-of-sight field-of-view in spherical coordinates.
- **Solar geometry primitives** — Sun sphere, concentric shells, planar discs,
  Thomson sphere, and lon/lat grid lines.
- **Mesh arithmetic** — ``SphericalMesh`` and ``CartesianMesh`` support the
  full NumPy arithmetic operator suite and ``__array_ufunc__`` so expressions
  like ``np.log10(mesh)`` return a new mesh of the same type.
- **PSI HDF support** — load model data directly from HDF4/HDF5 files written
  by the MAS code via ``psi-io``.

----

Installation
------------

**Core dependencies** (NumPy, PyVista, psi-io, SunPy, Astropy) are installed
automatically:

.. code-block:: bash

   pip install pyvisual

**Optional extras:**

.. code-block:: bash

   pip install "pyvisual[hdf4]"     # HDF4 file support via pyhdf
   pip install "pyvisual[interp]"   # scipy interpolation utilities
   pip install "pyvisual[tracing]"  # mapflpy fieldline tracing
   pip install "pyvisual[data]"     # pooch asset fetching + matplotlib
   pip install "pyvisual[all]"      # everything

**Development install:**

.. code-block:: bash

   git clone https://bitbucket.org/predsci/pyvisual.git
   cd pyvisual
   pip install -e ".[all]"

----

Quick Start
-----------

.. code-block:: python

   import pyvisual as pv
   import numpy as np

   pl = pv.Plot3d()

   # Add the Sun (radius 1 R☉, centred at the origin)
   pl.add_sun()

   # Add a 2-D equatorial slice at r = [1, 5] R☉, θ = π/2
   r = np.linspace(1, 5, 50)
   t = np.array([np.pi / 2])   # fix colatitude at the equator
   p = np.linspace(0, 2 * np.pi, 200)
   pl.add_2d_slice(r, t, p)

   # Add lon/lat grid lines just above the Sun's surface
   pl.add_longlat_lines(lat_deg=30, lon_deg=30, radius=1.01)

   pl.show()

----

Architecture
------------

``Plot3d`` extends ``pyvista.Plotter`` through four mixin classes:

+---------------------+--------------------------------------------------+
| Mixin               | Responsibility                                   |
+=====================+==================================================+
| ``ObserverMixin``   | Camera position/orientation/FOV in spherical     |
|                     | coordinates; live camera-state readout           |
+---------------------+--------------------------------------------------+
| ``GeometryMixin``   | Sun, shells, discs, Thomson sphere,              |
|                     | longitude/latitude grid lines                    |
+---------------------+--------------------------------------------------+
| ``GridMeshMixin``   | 1-D/2-D/3-D structured-grid slices and          |
|                     | isosurface contours                              |
+---------------------+--------------------------------------------------+
| ``StackMeshMixin``  | Points, splines, surfaces, and magnetic          |
|                     | fieldlines from stacked coordinate arrays        |
+---------------------+--------------------------------------------------+

Two structured-grid mesh classes carry PSI HDF data and plug directly into
``Plot3d.add_mesh``:

- ``SphericalMesh`` (``pyvista.RectilinearGrid``) — stores data on an
  :math:`(r, \theta, \phi)` grid; axes are accessible as ``.r``, ``.t``,
  ``.p``.
- ``CartesianMesh`` (``pyvista.StructuredGrid``) — stores data on an
  :math:`(x, y, z)` grid; supports the same arithmetic interface.

----

Documentation
-------------

Full API reference, gallery examples, and guides are published at:

   https://predsci.com/doc/pyvisual/

To build the documentation locally:

.. code-block:: bash

   pip install "pyvisual[docs]"
   cd docs && make html
   # output: docs/_build/html/index.html

----

Development
-----------

.. code-block:: bash

   # Linting
   ruff check .

   # Type checking
   mypy .

   # Tests
   pytest
   pytest --cov           # with coverage report

----

Data Conventions
----------------

Solar model data is stored in HDF4 or HDF5 files written in Fortran-order
array layout.  The PSI conventions for these files are documented at:

- https://predsci.com/doc/psi-io/guide/index.html
- https://predsci.com/doc/mapflpy/

Spherical coordinates follow the PSI :math:`(r, \theta, \phi)` convention —
radius, **co**\latitude, longitude — which differs from the physics
:math:`(r, \phi, \theta)` convention used in some other codes.

----

License
-------

Apache License 2.0.  See ``LICENSE.txt`` for details.

----

Authors
-------

- Ryder Davidson — `Predictive Science Inc. <https://www.predsci.com/>`_
- Cooper Downs — `Predictive Science Inc. <https://www.predsci.com/>`_
- Andres Reyes — `Predictive Science Inc. <https://www.predsci.com/>`_
