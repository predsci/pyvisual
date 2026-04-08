"""
CartesianMesh Filter Methods
============================

This example demonstrates filter methods on a
:class:`~pyvisual.core.mesh3d.CartesianMesh` and contrasts their behaviour
with the equivalent operations on a
:class:`~pyvisual.core.mesh3d.SphericalMesh`.

Both mesh types expose the same pyvisual filter API
(:meth:`~pyvisual.core.mesh3d.CartesianMeshFilters.logspace`,
:meth:`~pyvisual.core.mesh3d.CartesianMeshFilters.radially_scale`,
:meth:`~pyvisual.core.mesh3d.CartesianMeshFilters.deconstruct`), but because
they wrap different PyVista grid types
(:class:`pyvista.StructuredGrid` vs :class:`pyvista.RectilinearGrid`),
unmodified PyVista filters such as :meth:`pyvista.DataSetFilters.slice_orthogonal`
and :meth:`pyvista.DataSetFilters.slice_along_axis` behave differently on each:

- On a :class:`~pyvisual.core.mesh3d.CartesianMesh` PyVista operates on the
  true spatial :math:`(x, y, z)` point coordinates, so slices are flat planes
  in 3D space.
- On a :class:`~pyvisual.core.mesh3d.SphericalMesh` PyVista treats the stored
  :math:`(r, \\theta, \\phi)` axis arrays as if they were :math:`(x, y, z)`,
  so the same filter methods instead cut surfaces of constant radius,
  colatitude, and longitude.

See also :ref:`sphx_glr_06_spherical_grid_class_p03_filters.py` for the
spherical-mesh equivalent of this example.
"""

from pyvisual import Plot3d
from pyvisual.core.mesh3d import CartesianMesh, SphericalMesh
from pyvisual.utils.data import fetch_datasets

br_file = fetch_datasets("cor", "br").cor_br

# %%
# Build Both Mesh Types From the Same File
# ----------------------------------------
#
# Both meshes are constructed from the same HDF file.  Passing
# ``iformat='rtp'`` to :class:`~pyvisual.core.mesh3d.CartesianMesh` tells the
# constructor that the file contains spherical :math:`(r, \\theta, \\phi)`
# coordinates, which are automatically converted to Cartesian
# :math:`(x, y, z)` at construction time via
# :func:`~pyvisual.utils.geometry.spherical_to_cartesian`.
# :class:`~pyvisual.core.mesh3d.SphericalMesh` reads the same file and retains
# the native spherical coordinates.

cartesian_mesh = CartesianMesh(br_file, iformat='rtp')
spherical_mesh = SphericalMesh(br_file, iformat='rtp')
cartesian_mesh
spherical_mesh

# %%
# Index-based Sub-region Extraction
# ---------------------------------
#
# Indexing a :class:`~pyvisual.core.mesh3d.CartesianMesh` with a 3-tuple of
# slices selects grid points by their index positions in each axis direction,
# exactly as for :class:`~pyvisual.core.mesh3d.SphericalMesh` (see
# :ref:`sphx_glr_06_spherical_grid_class_p03_filters.py`).
# Because both mesh classes share the same underlying grid topology — the
# Cartesian conversion is applied to the *coordinates*, not the *connectivity*
# — identical index slices select the same spatial sub-region regardless of
# which mesh type is used.

sub_mesh = cartesian_mesh[120:150, 55:85, 135:165]

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(sub_mesh,
                 cmap='seismic', clim=(-1e-1, 1e-1), opacity=0.5, show_scalar_bar=False)
plotter.show()

# %%
# Orthogonal Slices — CartesianMesh
# ---------------------------------
#
# :meth:`pyvista.DataSetFilters.slice_orthogonal` cuts three mutually
# perpendicular cross-sections through the mesh centroid.  On a
# :class:`~pyvisual.core.mesh3d.CartesianMesh` the underlying grid stores
# real :math:`(x, y, z)` point coordinates, so the three slices are flat
# planes aligned with the coordinate axes — one each of the YZ, XZ, and XY
# planes passing through the centre of the domain.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(cartesian_mesh.slice_orthogonal(),
                 cmap='seismic', clim=(-1e-1, 1e-1), opacity=0.5, show_scalar_bar=False)
plotter.show()

# %%
# Orthogonal Slices — SphericalMesh
# ---------------------------------
#
# The identical :meth:`pyvista.DataSetFilters.slice_orthogonal` call on a
# :class:`~pyvisual.core.mesh3d.SphericalMesh` produces visually different
# results.  A :class:`~pyvisual.core.mesh3d.SphericalMesh` is backed by a
# :class:`pyvista.RectilinearGrid` whose three internal axes store
# :math:`r`, :math:`\\theta`, and :math:`\\phi` values respectively.
# PyVista is unaware of this convention and treats these axes as if they were
# :math:`x`, :math:`y`, and :math:`z`.  The three orthogonal cuts therefore
# correspond to surfaces of constant radius (a spherical shell), constant
# colatitude (a cone), and constant longitude (a meridional half-plane) —
# not flat planes in 3D space.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(spherical_mesh.slice_orthogonal(),
                 cmap='seismic', clim=(-1e-1, 1e-1), opacity=0.5, show_scalar_bar=False)
plotter.show()

# %%
# Radial Scaling and Axis-aligned Slicing
# ---------------------------------------
#
# :meth:`~pyvisual.core.mesh3d.CartesianMeshFilters.radially_scale` with
# ``exp=2`` multiplies :math:`B_r` by :math:`r^2`, compensating for the
# geometric :math:`1/r^2` flux-tube expansion and making outer-corona
# structure visible at the same colour scale as the inner corona.
# :meth:`pyvista.DataSetFilters.slice_along_axis` with ``n=30`` and
# ``axis='x'`` then cuts 30 evenly-spaced planes perpendicular to the
# :math:`x`-axis through the radially-scaled Cartesian mesh.  Because the
# mesh is in true :math:`(x, y, z)` space these are flat YZ cross-sections
# distributed across the domain.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(cartesian_mesh.radially_scale(exp=2).slice_along_axis(n=30, axis='x'),
                 cmap='seismic', clim=(-1e-1, 1e-1), opacity=0.5, show_scalar_bar=False)
plotter.show()

# %%
# Log-space Radial Remapping
# --------------------------
#
# :meth:`~pyvisual.core.mesh3d.CartesianMeshFilters.logspace` computes the
# Euclidean radius :math:`r = \lVert (x, y, z) \rVert` for every grid point
# and replaces it with :math:`\ln(r) + 1`, scaling all three coordinates
# proportionally so that the point positions move radially inward or outward.
# This compresses the large dynamic range in :math:`r` — expanding the
# tightly-packed inner corona and squeezing the sparsely-sampled outer
# boundary — while preserving the direction of every point from the origin.
# The scalar data is unchanged; only the point coordinates are modified.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(cartesian_mesh.logspace(),
                 cmap='seismic', clim=(-1e-1, 1e-1), opacity=0.5, show_scalar_bar=False)
plotter.show()