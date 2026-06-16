"""
Arithmetic and NumPy Ufunc Support
====================================

:class:`~pyvisual.core.mesh3d.SphericalMesh` (and
:class:`~pyvisual.core.mesh3d.CartesianMesh`) inherit a full arithmetic suite
from :class:`~pyvisual.core.mesh3d._BaseFrameMesh`.  Standard Python
operators (``+``, ``-``, ``*``, ``/``, ``**``, etc.) and NumPy ufuncs such as
:obj:`numpy.log10` and :obj:`numpy.sqrt` operate element-wise on the active
scalar field and return a new mesh of the same type with the result as the
active scalar.  The coordinate arrays are never modified — only the data
changes.
"""

import numpy as np
from pyvisual import Plot3d
from pyvisual.core.mesh3d import SphericalMesh
from psi_data import fetch_mas_data

br_file = fetch_mas_data(domains="cor", variables="br").cor_br

# %%
# Build a Mesh
# ------------
#
# We initialize a :class:`~pyvisual.core.mesh3d.SphericalMesh` from the HDF file path, which
# triggers the file-path dispatch path (see :ref:`sphx_glr_gallery_06_spherical_grid_class_p01_spherical_grid_init.py`
# for details on the three construction paths.
mesh = SphericalMesh(br_file)
mesh

# %%
# Radial Flux Scaling
# -------------------
#
# Multiplying by :math:`r^2` removes the geometric falloff and converts
# :math:`B_r` to the signed radial flux :math:`B_r r^2`.  The coordinate
# arrays are unchanged; only the active scalar is updated.

mesh_r2 = mesh * mesh.r[:, None, None] ** 2
print(f"Br     range: [{mesh.data.min():.4f}, {mesh.data.max():.4f}]")
print(f"Br r^2 range: [{mesh_r2.data.min():.4f}, {mesh_r2.data.max():.4f}]")

plotter = Plot3d()
plotter.show_axes()
plotter.add_mesh(mesh_r2, cmap='seismic', clim=(-1, 1), opacity=0.5, show_scalar_bar=False)
plotter.show()

# %%
# NumPy Ufunc: ``np.log10``
# -------------------------
#
# The :meth:`~pyvisual.core.mesh3d._BaseFrameMesh.__array_ufunc__` hook lets
# any single-output NumPy ufunc act directly on the mesh.
# :obj:`numpy.log10` applied to :math:`B_r r^2` converts the field to a
# logarithmic scale.
#
# To account for the sign of :math:`B_r r^2`, we take the absolute value before applying
# the logarithm using the python built-in :func:`abs` function, which also supports the
# mesh type.

mesh_log = np.log10(abs(mesh_r2))
print(f"log10(|Br r^2|) range: [{mesh_log.data.min():.3f}, {mesh_log.data.max():.3f}]")

plotter = Plot3d()
plotter.show_axes()
plotter.add_mesh(mesh_log, cmap='rainbow', clim=(-1, 1), opacity=0.5, show_scalar_bar=False)
plotter.show()
