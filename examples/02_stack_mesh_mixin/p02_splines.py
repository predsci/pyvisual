"""
Plotting Splines
================

This example demonstrates :meth:`~pyvisual.core.mixins.StackMeshMixin.add_spline`
and :meth:`~pyvisual.core.mixins.StackMeshMixin.add_splines` — the methods for
rendering single polyline paths and bundles of splines from 1-D and N-D
spherical coordinate arrays.
"""

import numpy as np
from pyvisual import Plot3d

# %%
# Single Spline
# -------------
#
# :meth:`~pyvisual.core.mixins.StackMeshMixin.add_spline` draws a single
# line through a sequence of :math:`(r, \theta, \phi)` points.  The example
# below traces an equatorial Archimedean spiral: the longitude :math:`\phi`
# increases uniformly as the radial distance :math:`r` grows from
# :math:`1\,R_\odot` to :math:`30\,R_\odot`, approximating a Parker spiral
# in the ecliptic plane.  The spline is colored by :math:`r`.

r = np.linspace(1, 30, 100)
t = np.repeat(np.pi / 2, 100)
p = np.linspace(0, 2 * np.pi, 100)

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_spline(r, t, p, r, line_width=5)
plotter.show()

# %%
# Bundle of Splines
# -----------------
#
# :meth:`~pyvisual.core.mixins.StackMeshMixin.add_splines` renders multiple
# splines from N-D coordinate arrays.  Here, 10 meridional arcs connect the
# north and south poles at evenly spaced longitudes, tracing the surface of a
# sphere of radius :math:`5\sin\theta\,R_\odot`.  The coordinate arrays have
# shape ``(10, 100)``; ``axis=1`` declares that axis 1 (length 100) traces
# each individual spline path and axis 0 (length 10) enumerates the distinct
# meridians.  Each spline is colored by its index.

n_lines, n_pts = 10, 100
r = np.tile(5 * np.sin(np.linspace(0, np.pi, n_pts)), (n_lines, 1))
t = np.tile(np.linspace(0, np.pi, n_pts), (n_lines, 1))
p = np.tile(np.linspace(0, 2 * np.pi, n_lines)[:, None], (1, n_pts))
data = np.arange(n_lines)

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_splines(r, t, p, data, axis=1, show_scalar_bar=False)
plotter.show()
