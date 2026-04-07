"""
Constructing a CartesianMesh
============================

This example demonstrates how to build a
:class:`~pyvisual.core.mesh3d.CartesianMesh` from NumPy arrays and how to
convert a :class:`~pyvisual.core.mesh3d.SphericalMesh` to Cartesian
coordinates for direct PyVista rendering.

:class:`~pyvisual.core.mesh3d.CartesianMesh` wraps
:class:`pyvista.StructuredGrid` with a Cartesian-frame tag.  Because the
underlying grid type is a structured grid (not a rectilinear grid), the three
coordinate arrays must be 3-D meshgrids of identical shape, not independent
1-D axis vectors.
"""

import numpy as np
from pyvisual import Plot3d
from pyvisual.core.mesh3d import CartesianMesh, SphericalMesh

# %%
# Building from 3-D Meshgrid Arrays
# ----------------------------------
#
# Create a :class:`~pyvisual.core.mesh3d.CartesianMesh` from a regular
# Cartesian grid.  Each coordinate array must have the same 3-D shape.
# Here we build a sphere of radial-distance data, which is everywhere equal
# to :math:`\sqrt{x^2 + y^2 + z^2}`.

x = np.linspace(-5, 5, 15)
y = np.linspace(-5, 5, 15)
z = np.linspace(-5, 5, 15)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
dist = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

mesh = CartesianMesh(X, Y, Z, data=dist, dataid='r')
print(f"dimensions : {mesh.dimensions}")
print(f"data range : [{mesh.data.min():.2f}, {mesh.data.max():.2f}]")

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(mesh, cmap='plasma', opacity=0.3, show_scalar_bar=False)
plotter.show()

# %%
# Converting a SphericalMesh to Cartesian
# ----------------------------------------
#
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.spherical_to_cartesian`
# casts a :class:`~pyvisual.core.mesh3d.SphericalMesh` to a
# :class:`pyvista.StructuredGrid` with Cartesian point coordinates.  The
# resulting dataset can be wrapped in a :class:`~pyvisual.core.mesh3d.CartesianMesh`
# for continued processing with the filter API, or added directly to the
# plotter.
#
# This conversion is useful when you want to apply Cartesian-space filters
# (such as PyVista's ``clip``, ``threshold``, or ``extract_surface``) to data
# that was originally defined on a spherical grid.

r = np.linspace(1, 5, 10)
t = np.linspace(0, np.pi, 20)
p = np.linspace(0, 2 * np.pi, 40)
R, T, P = np.meshgrid(r, t, p, indexing='ij')
Br = np.cos(T) / R ** 2

sph_mesh = SphericalMesh(r, t, p, data=Br, dataid='Br')
cart_struct = sph_mesh.spherical_to_cartesian()
cart_mesh = CartesianMesh(cart_struct)

print(f"SphericalMesh frame : {sph_mesh.user_dict['MESH_FRAME']}")
print(f"CartesianMesh frame : {cart_mesh.user_dict['MESH_FRAME']}")

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(cart_mesh, cmap='seismic', opacity=0.4, show_scalar_bar=False)
plotter.show()
