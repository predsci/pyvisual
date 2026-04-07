"""
Plotting Points
===============

This example demonstrates :meth:`~pyvisual.core.mixins.StackMeshMixin.add_point`
and :meth:`~pyvisual.core.mixins.StackMeshMixin.add_points` — the methods for
rendering individual points and point clouds from spherical coordinate arrays.
"""

import numpy as np
from pyvisual import Plot3d

# %%
# Single Point
# ------------
#
# :meth:`~pyvisual.core.mixins.StackMeshMixin.add_point` renders a single
# location given as :math:`(r, \theta, \phi)` scalars.  Here we place a marker
# at :math:`r = 1.5\,R_\odot` on the equatorial plane
# (:math:`\theta = \pi/2`) at 90° longitude (:math:`\phi = \pi/2`).

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.add_sun()
plotter.show_axes()
plotter.add_point(1.5, np.pi / 2, np.pi / 2, color='red', point_size=15)
plotter.show()

# %%
# Point Cloud
# -----------
#
# :meth:`~pyvisual.core.mixins.StackMeshMixin.add_points` accepts arrays of
# identical shape and renders them as an unconnected point cloud.  Below, 20
# points are placed along the equatorial plane (:math:`\theta = \pi/2`),
# spiraling outward from :math:`r = 1\,R_\odot` to :math:`r = 30\,R_\odot`
# across a full longitude sweep.  The ``data`` argument (here the radial
# distance :math:`r`) controls the color mapping.

r = np.linspace(1, 30, 20)
t = np.repeat(np.pi / 2, 20)
p = np.linspace(0, 2 * np.pi, 20)

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_points(r, t, p, r, point_size=5)
plotter.show()
