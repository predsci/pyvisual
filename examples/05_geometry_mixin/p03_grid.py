"""
Structured Spline Grids
=======================

This example demonstrates :meth:`~pyvisual.core.mixins.GeometryMixin.add_grid`
— the general method for adding structured spline grids to the scene.  Each
axis is specified as a 4-tuple ``(min, max, num_splines, resolution)``, and
the method draws three sets of splines (one per axis) as a
:class:`pyvista.MultiBlock`.

The higher-level convenience methods
:meth:`~pyvisual.core.mixins.GeometryMixin.add_longitudinal_lines`,
:meth:`~pyvisual.core.mixins.GeometryMixin.add_latitudinal_lines`, and
:meth:`~pyvisual.core.mixins.GeometryMixin.add_longlat_lines` are built on top
of :meth:`~pyvisual.core.mixins.GeometryMixin.add_grid`.
"""

from math import pi
from pyvisual import Plot3d

# %%
# 3-D Structured Grid Volume
# --------------------------
#
# The grid below spans a coronal volume between
# :math:`r \in [15,\,30]\,R_\odot`, :math:`\theta \in [\pi/4,\,3\pi/4]`, and
# :math:`\phi \in [\pi/4,\,3\pi/4]`.  The 4-tuple arguments control how many
# splines run *perpendicular* to each axis (``num_splines``) and how many
# sample points trace each individual spline (``resolution``).

radial_args = 15, 30, 3, 2
theta_args = pi / 4, 3 * pi / 4, 6, 60
phi_args = pi / 4, 3 * pi / 4, 6, 60

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_grid(radial_args, theta_args, phi_args)
plotter.show()

# %%
# Meridional Cross-Section
# ------------------------
#
# Setting ``num_splines=1`` and ``resolution=1`` on the :math:`\phi` axis
# collapses the grid to a single meridional plane at :math:`\phi = 0`.
# The result shows 10 radial spokes spanning
# :math:`r \in [1,\,30]\,R_\odot` and 12 colatitudinal arcs from
# pole to pole.

radial_args = 1, 30, 10, 2
theta_args = 0, pi, 12, 60
phi_args = 0, 0, 1, 1

plotter = Plot3d(off_screen=True, window_size=(500, 500))
plotter.show_axes()
plotter.add_sun()
plotter.add_grid(radial_args, theta_args, phi_args)
plotter.show()
