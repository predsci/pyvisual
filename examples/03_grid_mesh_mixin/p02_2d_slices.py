"""
2-D Surface Slices
==================

This example demonstrates :meth:`~pyvisual.core.mixins.GridMeshMixin.add_2d_slice`
— the method for rendering a 2-D surface at a fixed spherical coordinate.

Exactly one of ``r``, ``t``, ``p`` must be a size-1 array, pinning that
coordinate.  The other two axes define the surface grid.  The fixed axis is
inferred automatically, and the result is rendered as a quad-faced surface
colored by the supplied ``data`` array.

The two most common 2-D slice orientations in solar physics are the *radial
shell* (fixed :math:`r`, varying :math:`\theta` and :math:`\phi`) and the
*equatorial cut* (fixed :math:`\theta = \pi/2`, varying :math:`r` and
:math:`\phi`).
"""

import numpy as np
from pyvisual import Plot3d

# %%
# Radial Shell
# ------------
#
# A full spherical surface at :math:`r = 1\,R_\odot` showing a synthetic
# low-order multipole radial field :math:`B_r \propto \sin(2\theta)\cos(3\phi)`.
# In practice this surface is produced by passing a single-index read of a PSI
# HDF file, but the API is identical when using NumPy arrays.

r = np.array([1.0])
t = np.linspace(0, np.pi, 100)
p = np.linspace(0, 2 * np.pi, 200)
T, P = np.meshgrid(t, p, indexing='ij')
data = np.sin(2 * T) * np.cos(3 * P)

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_2d_slice(r, t, p, data, cmap='seismic', clim=(-1, 1),
                     show_scalar_bar=False)
plotter.show()

# %%
# Equatorial Cut
# --------------
#
# A meridional plane at :math:`\theta = \pi/2` (the equatorial plane) showing
# the radial field :math:`B_r r^2` — scaling by :math:`r^2` removes the
# geometric falloff of a dipole and highlights the azimuthal structure at all
# distances from :math:`1` to :math:`10\,R_\odot`.

t = np.array([np.pi / 2])
r = np.linspace(1, 10, 80)
p = np.linspace(0, 2 * np.pi, 200)
R, P = np.meshgrid(r, p, indexing='ij')
Br = np.cos(2 * P) / R ** 2
data = Br * R ** 2     # radially scaled flux

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_2d_slice(r, t, p, data, cmap='seismic', clim=(-1, 1),
                     show_scalar_bar=False)
plotter.show()
