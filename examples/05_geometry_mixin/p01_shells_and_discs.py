"""
Shells and Discs
================

This example demonstrates :meth:`~pyvisual.core.mixins.GeometryMixin.add_shell`
and :meth:`~pyvisual.core.mixins.GeometryMixin.add_disc` — convenience methods
for adding spherical shells and planar discs to the scene as reference
geometry.
"""

from __future__ import annotations

from math import pi

from pyvisual import Plot3d

# %%
# Source-Surface Shell
# --------------------
#
# :meth:`~pyvisual.core.mixins.GeometryMixin.add_shell` adds a spherical shell
# defined by inner and outer radii, centered at an arbitrary spherical position.
# Here, a translucent shell at :math:`r = 10\,R_\odot` represents the
# source surface — the heliospheric boundary beyond which coronal field lines
# are considered open.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_shell(outer_radius=10, opacity=0.2, color="cyan")
plotter.show()

# %%
# Positional Markers
# ------------------
#
# Shells can also be placed at off-center positions to mark specific locations
# in the corona.  The two small shells below are placed at symmetric latitudes
# in the northern and southern hemispheres at :math:`r = 2\,R_\odot`.

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_shell(2, pi / 4, 0, inner_radius=0.05, outer_radius=0.1, color="red", opacity=0.8)
plotter.add_shell(2, 3 * pi / 4, 0, inner_radius=0.05, outer_radius=0.1, color="blue", opacity=0.8)
plotter.show()

# %%
# Local Basis Discs
# -----------------
#
# :meth:`~pyvisual.core.mixins.GeometryMixin.add_disc` creates a planar disc
# whose normal is specified in the *local spherical basis*
# :math:`(\hat{r}, \hat{\theta}, \hat{\phi})` at the disc center.  The three
# discs below illustrate each basis direction at
# :math:`(r=2\,R_\odot,\,\theta=\pi/2,\,\phi=0)`:
# radial :math:`\hat{r}` (blue), colatitudinal :math:`\hat{\theta}` (white),
# and longitudinal :math:`\hat{\phi}` (red).

disc_kwargs = dict(r=2, t=pi / 2, p=0, outer_radius=0.2)

plotter = Plot3d()
plotter.show_axes()
plotter.add_sun()
plotter.add_disc(**disc_kwargs, normal=(1, 0, 0), color="blue")
plotter.add_disc(**disc_kwargs, normal=(0, 1, 0), color="white")
plotter.add_disc(**disc_kwargs, normal=(0, 0, 1), color="red")
plotter.show()
