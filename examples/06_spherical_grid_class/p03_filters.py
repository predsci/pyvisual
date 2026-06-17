"""
SphericalMesh Filter Methods
==============================

This example demonstrates the filter methods provided by
:class:`~pyvisual.core.mesh3d.SphericalMeshFilters`:

- :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.logspace` — remap radial
  axis coordinates to a logarithmic scale.
- :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.deconstruct` — convert
  the structured grid into a :class:`pyvista.PolyData` representation for
  fine-grained rendering control.

All filter methods return a *new* mesh, leaving the original unmodified.
"""

import numpy as np
import pyvista as pv
from pyvisual import Plot3d
from pyvisual.core.mesh3d import SphericalMesh, build_point_polydata, build_spline_polydata
from psi_data import fetch_mas_data
from pyvisual.utils.geometry import spacecraft_trajectory, spherical_to_cartesian

br_file = fetch_mas_data(domains="cor", variables="br").cor_br

# %%
# Build a Mesh
# ------------
#
# Load the coronal radial magnetic field (:math:`B_r`) HDF file and construct a
# :class:`~pyvisual.core.mesh3d.SphericalMesh` via the file-path dispatcher (see
# :ref:`sphx_glr_gallery_06_spherical_grid_class_p01_spherical_grid_init.py` for details on the
# three construction methods).  A sub-region is then sliced out in
# :math:`(r, \theta, \phi)` index space and rendered as a semi-transparent volume
# using a diverging colormap.
mesh = SphericalMesh(br_file)
mesh

# %%
#

sub_mesh = mesh[120:150, 55:85, 135:165]
sub_mesh

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(sub_mesh, cmap="seismic", clim=(-1e-1, 1e-1), opacity=0.5, show_scalar_bar=False)
plotter.show()

# %%
# Log-space Remapping and Radial Scaling
# ---------------------------------------
#
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.logspace` remaps each radial
# coordinate :math:`r \to \ln(r) + 1`, compressing the large dynamic range in
# :math:`r`.  :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.radially_scale`
# with ``exp=3`` then multiplies the scalar data by :math:`r^3`.
# Together these two operations make weak outer-corona signal visible
# alongside strong inner-corona signal.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(
	sub_mesh.logspace().radially_scale(exp=3),
	cmap="seismic",
	clim=(-1e-1, 1e-1),
	opacity=0.5,
	show_scalar_bar=False,
)
plotter.show()

# %%
# Deconstruct to PolyData (Slice Method)
# ---------------------------------------
#
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.deconstruct` converts the
# :class:`pyvista.RectilinearGrid` into a :class:`pyvista.PolyData` using one of
# three builder strategies (``'points'``, ``'splines'``, or ``'slices'``).  Here
# ``axis=1`` takes slices along the :math:`\theta` (colatitude) axis and
# ``method='slices'`` produces quad-faced surface patches.  This gives finer
# rendering control — e.g. per-face opacity or individual slice visibility — than
# the raw structured grid.
#
# .. note::
#    This method dispatches to the :func:`~pyvisual.core.mesh3d.build_point_polydata`,
#    :func:`~pyvisual.core.mesh3d.build_spline_polydata`, and
#    :func:`~pyvisual.core.mesh3d.build_slice_polydata` functions in
#    ``pyvisual.core.mesh3d``.  See their documentation for details on the
#    three builder strategies.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(
	sub_mesh.deconstruct(axis=1, method="slices"),
	cmap="seismic",
	clim=(-1e-1, 1e-1),
	opacity=0.5,
	show_scalar_bar=False,
)
plotter.show()

# %%
# Spacecraft Trajectory Interpolation
# -------------------------------------
#
# :func:`~pyvisual.utils.geometry.spacecraft_trajectory` queries JPL Horizons for
# Parker Solar Probe's ephemeris over the given date range at one-hour cadence,
# returning a ``(3, nt)`` array in :math:`(r, \theta, \phi)`.
# :func:`~pyvisual.core.mesh3d.build_spline_polydata` converts that array into a
# :class:`pyvista.PolyData` spline, and
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.interpolate_mesh` samples the
# :class:`~pyvisual.core.mesh3d.SphericalMesh` scalar field onto those trajectory
# points via :meth:`pyvista.DataObjectFilters.sample`.
#
# The second :meth:`~pyvista.Plotter.add_mesh` call projects the colored trajectory
# onto the plane :math:`r = 2 R_\odot` using
# :meth:`~pyvista.PolyDataFilters.project_points_to_plane`, producing a "shadow" projection
# for spatial context.

trajectory = spacecraft_trajectory("psp", "2024-03-28", "2024-04-01")
trajectory_polydata = build_spline_polydata(*trajectory, axis=0)
interpolated_path = mesh.interpolate_mesh(trajectory_polydata)
plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(
	interpolated_path,
	cmap="seismic",
	clim=(-5e-3, 5e-3),
	render_lines_as_tubes=True,
	line_width=5,
)
plotter.add_mesh(
	interpolated_path.project_points_to_plane(origin=(2, 0, 0), normal=(1, 0, 0)),
	cmap="seismic",
	clim=(-5e-3, 5e-3),
	render_lines_as_tubes=True,
	line_width=5,
)
plotter.show()

# %%
# Orbital-Plane Slice Through the Mesh
# ------------------------------------
#
# Two points on the trajectory (indices 0 and 50) are converted to Cartesian
# coordinates via :func:`~pyvisual.utils.geometry.spherical_to_cartesian`.  Their
# cross product defines the normal to the orbital plane, which is used to orient a
# :class:`~pyvista.PolyData` disc spanning 30 :math:`R_\odot`.
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.spherical_to_cartesian` converts
# the :class:`pyvista.RectilinearGrid` to a Cartesian
# :class:`pyvista.StructuredGrid`, and :meth:`pyvista.DataObjectFilters.sample` then
# interpolates :math:`B_r` onto the disc's points — producing a filled orbital-plane
# slice colored by the model field.

p0 = spherical_to_cartesian(*trajectory[:, 0])
p1 = spherical_to_cartesian(*trajectory[:, 50])
normal = np.cross(np.array(p0), np.array(p1))
unit_normal = normal / np.linalg.norm(normal)
trajecotry_plane = pv.Disc(
	center=(0, 0, 0), inner=0, outer=30, normal=unit_normal, r_res=255, c_res=255
)
plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(
	interpolated_path,
	cmap="seismic",
	clim=(-5e-3, 5e-3),
	render_lines_as_tubes=True,
	line_width=5,
)
plotter.add_mesh(
	trajecotry_plane.sample(mesh.spherical_to_cartesian()),
	cmap="seismic",
	clim=(-5e-3, 5e-3),
)
plotter.show()
