"""
1-D Line Slices
===============

This example demonstrates :meth:`~pyvisual.core.mixins.GridMeshMixin.add_1d_slice`
— the method for rendering a line slice along a single spherical coordinate
axis.

Exactly two of ``r``, ``t``, ``p`` must be size-1 arrays, fixing those
coordinates.  The one array with more than one element defines the slice
direction; **pyvisual** infers the varying axis automatically and renders the
result as a polyline colored by the supplied ``data`` array.

The two most common 1-D slice orientations in solar physics are the
*longitudinal profile* (varying :math:`\phi` at fixed :math:`r` and
:math:`\theta`) and the *meridional profile* (varying :math:`\theta` at fixed
:math:`r` and :math:`\phi`).
"""

import numpy as np
from pyvisual import Plot3d

# %%
# Longitudinal Profile
# --------------------
#
# Fix the radius at :math:`r = 1\,R_\odot` and the colatitude at the
# equatorial plane (:math:`\theta = \pi/2`), then sweep longitude
# :math:`\phi` from 0 to :math:`2\pi`.  The synthetic data mimics a
# low-order multipole radial field :math:`B_r \propto \sin(2\phi)`.

r = np.array([1.0])
t = np.array([np.pi / 2])
p = np.linspace(0, 2 * np.pi, 360)
data = np.sin(2 * p) * np.cos(p / 2)

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_1d_slice(r, t, p, data,
                     cmap='seismic', clim=(-1, 1), line_width=5)
plotter.show()

# %%
# Meridional Profile
# ------------------
#
# Fix the radius at :math:`r = 2\,R_\odot` and the longitude at
# :math:`\phi = 0`, then sweep colatitude :math:`\theta` from pole to pole.
# A dipole-like :math:`B_r \propto \cos\theta` profile changes sign at the
# equatorial plane, as expected for a pure dipole field.

r = np.array([2.0])
t = np.linspace(0, np.pi, 180)
p = np.array([0.0])
data = np.cos(t)

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_1d_slice(r, t, p, data,
                     cmap='seismic', clim=(-1, 1), line_width=5)
plotter.show()
