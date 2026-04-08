"""
Line-of-Sight and Field-of-View Control
=========================================

This example demonstrates the two complementary APIs for aiming and framing the
observer's camera in helioprojective coordinates:

- :attr:`~pyvisual.core.mixins.ObserverMixin.observer_los_view` — sets the
  field of view as explicit angular extents :math:`(x_0, x_1, y_0, y_1)` in
  degrees, measured from Sun-center along the observer's line of sight.
- :attr:`~pyvisual.core.mixins.ObserverMixin.observer_fov_view` — sets the
  (square) field of view by specifying the minimum impact radius
  :math:`r_{\\min}` (in :math:`R_\\odot`) that any line of sight must clear.

Both properties require :attr:`~pyvisual.core.mixins.ObserverMixin.observer_position`
to be set first, because the focal-point calculation depends on the observer
distance.
"""

import numpy as np
from pyvisual import Plot3d

# %%
# LOS View — Asymmetric Field of View
# -------------------------------------
#
# :attr:`~pyvisual.core.mixins.ObserverMixin.observer_los_view` accepts a
# 4-tuple ``(x0, x1, y0, y1)`` of helioprojective angular extents in degrees.
# Here the observer sits at :math:`r = 50\,R_\odot` on the equatorial plane
# and looks back at the Sun with a wide horizontal sweep
# (:math:`-15°` to :math:`+15°`) and a narrower vertical window
# (:math:`-8°` to :math:`+8°`), placing the viewport aspect ratio at
# :math:`30/16 \approx 1.875`.
#
# The focal point is automatically computed from the center of the angular
# range via the Thomson sphere, so the camera always aims along the correct
# helioprojective direction.

plotter = Plot3d()
plotter.add_sun()
plotter.add_shell(inner_radius=2.0, outer_radius=2.0, opacity=0.15, color='cyan')
plotter.add_shell(inner_radius=6.0, outer_radius=6.0, opacity=0.10, color='orange')
plotter.observer_position = 50, np.pi / 2, 0
plotter.observer_los_view = -15, 15, -8, 8
plotter.show()

# %%
# LOS View — Off-Center Pointing
# --------------------------------
#
# Shifting the angular range off-center moves the focal point away from
# Sun-center.  Here the horizontal window is displaced by :math:`+5°`
# (``x0 = -5``, ``x1 = +15``) so that the camera is biased to the east
# limb of the Sun.  The vertical extent remains symmetric.

plotter = Plot3d()
plotter.add_sun()
plotter.add_shell(inner_radius=2.0, outer_radius=2.0, opacity=0.15, color='cyan')
plotter.observer_position = 50, np.pi / 2, 0
plotter.observer_los_view = -5, 15, -8, 8
plotter.show()

# %%
# FOV View — Impact-Parameter Framing
# -------------------------------------
#
# :attr:`~pyvisual.core.mixins.ObserverMixin.observer_fov_view` is a
# higher-level alternative.  Instead of raw angles, you supply a single scalar
# :math:`r_{\min}`.
#
# The result is a square viewport whose edge lines of sight just graze the
# sphere of radius :math:`r_{\min}` around Sun-center.  Setting
# :math:`r_{\min} = 1\,R_\odot` frames the entire solar disc with minimal
# margin; larger values show the extended corona.

plotter = Plot3d()
plotter.add_sun()
plotter.add_shell(inner_radius=2.5, outer_radius=2.5, opacity=0.15, color='cyan')
plotter.observer_position = 50, np.pi / 2, 0
plotter.observer_fov_view = 4
plotter.show()

# %%
# Comparing Impact Radii
# -----------------------
#
# The same observer position is used at three different impact radii to
# illustrate how a smaller :math:`r_{\min}` zooms in on the solar disc
# while a larger one opens the view out to the extended corona.
#
# .. note::
#    The observer distance is fixed at :math:`50\,R_\odot` throughout.
#    Changing :math:`r_{\min}` only rescales the angular FOV; it does not
#    move the camera.

for rmin in (1.5, 4.0, 10.0):
    plotter = Plot3d()
    plotter.add_sun()
    plotter.add_shell(inner_radius=rmin, outer_radius=rmin, opacity=0.2, color='yellow')
    plotter.observer_position = 50, np.pi / 2, 0
    plotter.observer_fov_view = rmin
    plotter.show()