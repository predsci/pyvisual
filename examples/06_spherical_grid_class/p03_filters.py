"""
SphericalMesh Filter Methods
==============================

This example demonstrates the filter methods provided by
:class:`~pyvisual.core.mesh3d.SphericalMeshFilters`:

- :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.radially_scale` — multiply
  the active scalar by a power of the radial coordinate.
- :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.logspace` — remap radial
  axis coordinates to a logarithmic scale.
- :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.deconstruct` — convert
  the structured grid into a :class:`pyvista.PolyData` representation for
  fine-grained rendering control.

All filter methods return a *new* mesh, leaving the original unmodified.
"""

import numpy as np
from pyvisual import Plot3d
from pyvisual.core.mesh3d import SphericalMesh

# %%
# Setup: Dipole Field
# -------------------

r = np.linspace(1, 30, 30)
t = np.linspace(0, np.pi, 40)
p = np.linspace(0, 2 * np.pi, 80)

R, T, P = np.meshgrid(r, t, p, indexing='ij')
Br = np.cos(T) / R ** 2
mesh = SphericalMesh(r, t, p, data=Br, dataid='Br')

# %%
# Radial Scaling
# --------------
#
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.radially_scale` multiplies
# the active scalar field by :math:`r^n` (with :math:`n = e` by default).
# This converts :math:`B_r` to the radially scaled flux :math:`B_r r^e`,
# which removes the power-law falloff and highlights structural patterns
# independent of distance.  Passing ``exp=2`` scales by :math:`r^2` instead,
# recovering the standard signed flux :math:`B_r r^2`.

mesh_scaled = mesh.radially_scale(exp=2)
print(f"Br     range: [{mesh.data.min():.3f}, {mesh.data.max():.3f}]")
print(f"Br r^2 range: [{mesh_scaled.data.min():.3f}, {mesh_scaled.data.max():.3f}]")

# %%
# Logarithmic Radial Spacing
# --------------------------
#
# Solar corona data often spans several decades in radius.
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.logspace` replaces each
# radial coordinate :math:`r` with :math:`\ln(r) + 1`, compressing the outer
# corona while stretching the inner region.  This is particularly useful when
# visualizing large radial domains — here 1 to 30 :math:`R_\odot` — where
# structures near the solar surface would otherwise be invisible at true scale.

mesh_log = mesh.logspace()
print(f"original r: [{mesh.r.min():.1f}, {mesh.r.max():.1f}]")
print(f"logspace r: [{mesh_log.r.min():.3f}, {mesh_log.r.max():.3f}]")

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_mesh(mesh_log, cmap='seismic',
                 clim=(-mesh.data.max(), mesh.data.max()),
                 opacity=0.4, show_scalar_bar=False)
plotter.show()

# %%
# Deconstructing to PolyData
# --------------------------
#
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.deconstruct` converts the
# structured grid into a :class:`pyvista.PolyData` built from the coordinate
# arrays.  The ``method`` argument selects the polydata builder:
# ``'slices'`` creates quad-faced surface patches (useful for rendering
# individual :math:`\phi` shells as opaque faces), while ``'splines'`` and
# ``'points'`` produce connected lines and unconnected points respectively.
# Here ``axis=0`` declares that each radial shell is a separate patch.

polydata = mesh.deconstruct(axis=0, method='slices')
print(f"PolyData type : {type(polydata).__name__}")
print(f"Number of cells: {polydata.n_cells}")

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(polydata, frame='spherical',
                 cmap='seismic',
                 clim=(-0.5, 0.5),
                 opacity=0.3,
                 show_scalar_bar=False)
plotter.show()
