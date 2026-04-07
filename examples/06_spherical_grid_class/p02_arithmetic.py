"""
Arithmetic and NumPy Ufunc Support
====================================

:class:`~pyvisual.core.mesh3d.SphericalMesh` (and
:class:`~pyvisual.core.mesh3d.CartesianMesh`) inherit a full arithmetic suite
from :class:`~pyvisual.core.mesh3d._BaseFrameMesh`.  Standard Python
operators (``+``, ``-``, ``*``, ``/``, ``**``, etc.) and NumPy ufuncs such as
:func:`numpy.log10` and :func:`numpy.sqrt` operate element-wise on the active
scalar field and return a new mesh of the same type with the result as the
active scalar.  The coordinate arrays are never modified — only the data
changes.
"""

import numpy as np
from pyvisual import Plot3d
from pyvisual.core.mesh3d import SphericalMesh

# %%
# Build a Base Mesh
# -----------------
#
# We start with a simple dipole-like :math:`B_r` field, scaled to be
# everywhere positive so that logarithmic operations are well-defined.

r = np.linspace(1, 10, 20)
t = np.linspace(0.1, np.pi - 0.1, 30)   # avoid exact poles
p = np.linspace(0, 2 * np.pi, 60)

R, T, P = np.meshgrid(r, t, p, indexing='ij')
Br = np.abs(np.cos(T)) / R ** 2          # always positive

mesh = SphericalMesh(r, t, p, data=Br, dataid='Br')

# %%
# Radial Flux Scaling
# -------------------
#
# Multiplying by :math:`r^2` removes the geometric falloff and converts
# :math:`B_r` to the signed radial flux :math:`B_r r^2`.  The coordinate
# arrays are unchanged; only the active scalar is updated.

R_axis = np.linspace(1, 10, 20)
mesh_r2 = mesh * R_axis[:, None, None] ** 2
print(f"Br     range: [{mesh.data.min():.4f}, {mesh.data.max():.4f}]")
print(f"Br r^2 range: [{mesh_r2.data.min():.4f}, {mesh_r2.data.max():.4f}]")

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(mesh_r2, cmap='hot', opacity=0.5, show_scalar_bar=False)
plotter.show()

# %%
# NumPy Ufunc: ``np.log10``
# -------------------------
#
# The :meth:`~pyvisual.core.mesh3d._BaseFrameMesh.__array_ufunc__` hook lets
# any single-output NumPy ufunc act directly on the mesh.
# :func:`numpy.log10` applied to :math:`B_r r^2` converts the field to a
# logarithmic scale, which is useful when the data spans several decades.

mesh_log = np.log10(mesh_r2)
print(f"log10(Br r^2) range: [{mesh_log.data.min():.3f}, {mesh_log.data.max():.3f}]")

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_mesh(mesh_log, cmap='inferno', opacity=0.5, show_scalar_bar=False)
plotter.show()
