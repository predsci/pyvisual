.. _overview:

Overview
========

**pyvisual** is a thin wrapper around `PyVista <https://docs.pyvista.org/>`_
that adds spherical-coordinate awareness, observer controls, and rendering
utilities tuned to solar and magnetohydrodynamic (MHD) datasets. It is developed
and maintained by `Predictive Science Inc. (PSI) <https://www.predsci.com/>`_ and
is tightly coupled to the PSI data ecosystem — in particular
`psi-io <https://predsci.com/doc/psi-io/guide/index.html>`_ (HDF4/HDF5 file I/O)
and `mapflpy <https://predsci.com/doc/mapflpy/>`_ (magnetic fieldline tracing).
With that said, any rectilinear grid defined in spherical coordinates is compatible
with the API.

.. attention::

   Because **pyvisual** renders through PyVista and VTK, it is *strongly*
   recommended to read the `PyVista documentation <https://docs.pyvista.org/>`_
   alongside this guide. **pyvisual** exposes only a focused subset of PyVista's
   capabilities; familiarity with the underlying library unlocks considerably more
   flexibility.

Coordinate Conventions
-----------------------

**pyvisual** uses two coordinate frames throughout.

Spherical frame (``'spherical'``)
    Coordinates are ordered :math:`(r, \theta, \phi)`, where :math:`r` is the
    radial distance in solar radii :math:`R_\odot`, :math:`\theta` is the
    colatitude measured from the north pole, and :math:`\phi` is the longitude.
    This is the native PSI convention; all model data is stored in this frame.
    Accepted aliases include ``'rtp'``, ``'polar'``, ``'psi'``, and any permutation
    of the individual axis letters (``'r'``, ``'tp'``, etc.).

Cartesian frame (``'cartesian'``)
    Coordinates are ordered :math:`(x, y, z)`, where :math:`+\hat{z}` points
    toward solar north (``SOLAR_NORTH = [0, 0, 1]``). This is the frame used
    internally by PyVista and VTK for all rendering.
    Accepted aliases include ``'xyz'``, ``'cart'``, ``'rectilinear'``, and any
    permutation of the individual axis letters.

.. note::

   **pyvisual** is principally concerned with rendering rectilinear spherical grids,
   *viz.* those produced by PSI's MAS code; more specificially, this package is
   designed to facilitate the visualization of such datasets by converting these
   structured :math:`(r, \theta, \phi)` grids into the native Cartesian coordinate
   system used by PyVista and VTK for rendering.

   The coordinate system of the :class:`~pyvista.Plotter` is always a Cartesian
   coordinate system in the Heliographic Carrington frame. Input data (however)
   is generally expected to be in the spherical frame, and the API provides utilities for
   converting between the two.

   The spherical frame is used for all observer controls,
   reference geometry, and mesh construction; the Cartesian frame is used for all
   rendering and camera manipulation. The API abstracts away the details of converting
   between the two, but it is important to understand the distinction when working with
   the package.

The Plotter: ``Plot3d``
-----------------------

:class:`~pyvisual.core.plot3d.Plot3d` is the primary entry point. It subclasses
:class:`pyvista.Plotter` and mixes in four functional areas:

.. code-block:: python

    from pyvisual import Plot3d

    plotter = Plot3d()
    plotter.add_sun()
    plotter.show()

All keyword arguments accepted by :class:`pyvista.Plotter` (lighting, window size,
off-screen rendering, etc.) are also accepted by :class:`~pyvisual.core.plot3d.Plot3d`.

Mixin Areas
-----------

The four mixin classes that compose :class:`~pyvisual.core.plot3d.Plot3d` are
defined in :mod:`pyvisual.core.mixins`.

Observer controls (:class:`~pyvisual.core.mixins.ObserverMixin`)
    Sets the camera position and orientation using spherical coordinates. The
    observer location is expressed as a :class:`~pyvisual.core._typing.SphericalCoordinate`
    :math:`(r, \theta, \phi)`, and the API also exposes the line-of-sight FOV angle
    and a position angle measured from solar north. A live camera-state readout can
    be displayed to inspect the current view parameters interactively.

Solar geometry (:class:`~pyvisual.core.mixins.GeometryMixin`)
    Adds reference geometry to a scene: the Sun sphere (radius
    :math:`1\,R_\odot`), spherical shells at arbitrary radii, planar discs
    (e.g., the ecliptic plane), longitude/latitude grid lines, and the Thomson
    sphere (the sphere of unit optical depth for Thomson scattering, centered
    halfway between the Sun and the observer).

Structured-grid slices (:class:`~pyvisual.core.mixins.GridMeshMixin`)
    Renders data from structured :math:`(r, \theta, \phi)` grids. Supports
    1-D line slices, 2-D surface slices, and 3-D volume slices, as well as
    isosurface contours. Coordinate arrays may be supplied as independent 1-D
    axis arrays or as pre-broadcast 3-D arrays.

Stacked-coordinate rendering (:class:`~pyvisual.core.mixins.StackMeshMixin`)
    Renders data where all coordinate axes share the same array shape — for
    example, in-situ spacecraft trajectories or traced fieldlines. Supports
    single points, point clouds, splines, magnetic fieldline bundles, and
    free-form surfaces.

Mesh Classes
------------

Two mesh classes are provided for loading and manipulating model data before
adding it to a plotter. These classes can be instantiated from an HDF file
path, from raw coordinate and data arrays, or from an existing PyVista dataset.

.. note::

   When instantiating from an HDF filepath, the :mod:`psi-io` library is used
   to read the file and extract the coordinate and data arrays by index.
   As such, the file must be in a format compatible with :mod:`psi-io` (e.g. HDF4 with
   Fortran-order arrays, or HDF5 with the same structure).

   *It is generally
   recommended to explicitly load the data arrays to ensure that the scales
   and data-values are properly interpreted, rather than relying on the mesh
   classes to infer them from the file.*

The motivation behind these classes is to provide a convenient container for solar
physics model data that can make full use of the powerful PyVista/VTK
`Filters <https://docs.pyvista.org/api/core/filters.html>`_ (along with a few
additional "filters" provided by **pyvisual** in the
:class:`~pyvisual.core.mesh3d.CartesianMeshFilters` and
:class:`~pyvisual.core.mesh3d.SphericalMeshFilters` mixin classes).

:class:`~pyvisual.core.mesh3d.SphericalMesh`
    Wraps :class:`pyvista.RectilinearGrid` with a spherical-frame tag. Can be
    constructed from an HDF4/HDF5 file path, raw :math:`(r, \theta, \phi)` + data
    arrays, or an existing PyVista dataset. Supports the full NumPy arithmetic
    suite (``+``, ``-``, ``*``, ``/``, ``**``, etc.) and
    :meth:`~object.__array_ufunc__` so that expressions such as
    ``np.log10(mesh)`` work element-wise on the active scalar field.

:class:`~pyvisual.core.mesh3d.CartesianMesh`
    The Cartesian counterpart, wrapping :class:`pyvista.StructuredGrid`. It shares
    the same operator API as :class:`~pyvisual.core.mesh3d.SphericalMesh`.

.. note::

   The general motivation behind the latter class (in contrast to the :class:`~pyvisual.core.mesh3d.SphericalMesh`
   class) is to facilitate the use of PyVista/VTK's
   `filters <https://docs.pyvista.org/api/core/filters>`_ on spherical grids that have
   been converted to Cartesian coordinates. This :func:`~pyvisual.utils.geometry.spherical_to_cartesian`
   transformation yeilds topological structured meshes that are, nevertheless, not composed
   of monotonically increasing coordinate arrays. Therefore, the grid's internal structure
   has to be stored explicitly *viz.* through a :class:`pyvista.StructuredGrid` rather than a
   :class:`pyvista.RectilinearGrid`.

.. warning::

   The consequence of the above note is that the point arrays of this class are
   derived from a :func:`~numpy.meshgrid`-like Cartesian product of the input scales,
   and are not stored as the three separate 1-D arrays that are typical of rectilinear grids.
   As such, these grids can be substantially more memory-intensive than their
   :class:`~pyvisual.core.mesh3d.SphericalMesh` counterparts.

Both classes are available from the top-level package::

    from pyvisual import SphericalMesh, CartesianMesh

PSI Data Ecosystem
------------------

**pyvisual** is designed for use with model output produced by PSI's MAS
(Magnetohydrodynamics Around a Sphere) code, stored in HDF4 or HDF5 files using
Fortran-order array layout. File I/O is handled by
`psi-io <https://predsci.com/doc/psi-io/guide/index.html>`_, which is a required
dependency. Fieldline tracing (for use with
:class:`~pyvisual.core.mixins.StackMeshMixin`) is provided by
`mapflpy <https://predsci.com/doc/mapflpy/>`_, an optional dependency installed
via the ``tracing`` extra.

For a more general solar-physics visualization toolkit with coordinate-frame
awareness beyond what **pyvisual** provides, consider
`sunkit-pyvista <https://docs.sunpy.org/projects/sunkit-pyvista/en/latest/>`_.

.. seealso::

   :ref:`installation`
      How to install **pyvisual** and its optional dependencies.

   :doc:`/gallery/index`
      Worked examples covering basic scenes, slicing, fieldlines, and observer setup.

   `PyVista documentation <https://docs.pyvista.org/>`_
      The upstream library that **pyvisual** extends.