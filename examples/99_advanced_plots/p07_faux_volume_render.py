# noqa: INP001
"""
Faux Volume Rendering
=====================

This example demonstrates a technique for simulating volumetric rendering of
coronal density by deconstructing a 3-D
:class:`~pyvisual.core.mesh3d.SphericalMesh` into stacked semi-transparent
radial shells — a *faux* volume that avoids the GPU memory cost of true
volumetric rendering while still conveying the large-scale 3-D structure of
the corona.

The scene is animated along a Parker Solar Probe (PSP) trajectory
obtained from the JPL Horizons ephemeris service via
:func:`~pyvisual.utils.geometry.spacecraft_trajectory`, placing the virtual
observer at each spacecraft position and framing the field of view in
helioprojective angular coordinates using
:attr:`~pyvisual.core.mixins.ObserverMixin.observer_los_view`.

.. seealso::

   :ref:`sphx_glr_gallery_04_observer_mixin_p02_orbit.py`
      Simpler orbit animation driven by a synthetic observer path.

   :ref:`sphx_glr_gallery_99_advanced_plots_p01_combining_multiple_elements.py`
      Multi-layer coronal scene that combines slices, contours, and fieldlines.
"""

# sphinx_gallery_thumbnail_path = '_static/assets/p07_faux_volume_render.png'

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from psi_data import fetch_mas_data

from pyvisual import Plot3d
from pyvisual.core.mesh3d import SphericalMesh
from pyvisual.utils.geometry import spacecraft_trajectory

# %%
# Load Data
# ---------
#
# :func:`~pyvisual.utils.geometry.spacecraft_trajectory` queries the JPL
# Horizons ephemeris for Parker Solar Probe over a four-day window and returns
# a :class:`numpy.ndarray` of shape ``(3, N)`` whose rows are
# :math:`(r,\,\theta,\,\phi)` in the Carrington frame, sampled at the default
# 1-hour cadence.
#
# :func:`psi_data.fetch_mas_data` downloads (or loads from cache)
# the CR 2309 coronal density field on the
# :math:`1\text{–}30\,R_\odot` grid and returns the path to the HDF5 file as
# the named attribute ``cor_rho``.

trajectory = spacecraft_trajectory("psp", "2024-03-27", "2024-03-31")
rho_file = fetch_mas_data(domains="cor", variables="rho").cor_rho

# %%
# Build the Faux Volume Representation
# --------------------------------------
#
# The full density grid is loaded into a
# :class:`~pyvisual.core.mesh3d.SphericalMesh`.  Indexing with ``[0, ...]``
# selects the innermost radial shell at :math:`r \approx 1\,R_\odot` as a
# 2-D reference surface for the photospheric density.
#
# The remaining shells ``[1:, ...]`` form the 3-D coronal volume.  Applying
# :obj:`numpy.log` via the mesh arithmetic suite compresses the large dynamic
# range of coronal density.
#
# :meth:`~pyvisual.core.mesh3d.SphericalMeshFilters.deconstruct` with
# ``method='slices'`` converts the 3-D :class:`pyvista.RectilinearGrid` into a
# collection of 2-D radial shell surfaces (via
# :func:`~pyvisual.core.mesh3d.build_slice_polydata`).  Rendered with PyVista's
# ``'sigmoid_7'`` opacity transfer function, the stacked shells map low
# log-density regions to near-transparent and high log-density regions to
# near-opaque, mimicking the appearance of a volumetric render without
# requiring volume rendering hardware support.

mesh = SphericalMesh(rho_file)
radial_slice_at_photosphere = mesh[0, ...]
deconstructed_mesh_volume = np.log(mesh[1:, ...]).deconstruct(method="slices")

# %%
# Animate Along the Spacecraft Trajectory
# ----------------------------------------
#
# The scene is assembled once and then animated by stepping the observer through
# each position in the PSP trajectory.  Iterating over ``trajectory.T`` yields
# one :math:`(r,\,\theta,\,\phi)` column per time step, which is assigned
# directly to :attr:`~pyvisual.core.mixins.ObserverMixin.observer_position`.
#
# :attr:`~pyvisual.core.mixins.ObserverMixin.observer_los_view` frames the
# field of view as helioprojective angular extents
# :math:`(x_0,\, x_1,\, y_0,\, y_1)` in degrees. Re-applying the FOV at
# every frame ensures consistent framing as the spacecraft distance changes
# over the trajectory.
#
# .. warning::
#
#    The interactive 3-D viewer is omitted here because the deconstructed
#    shell mesh is prohibitively large to embed in a browser. The animation
#    is also generated and cached outside the sphinx-gallery pipeline:
#    sphinx-gallery natively scrapes GIF output, but the 256-color palette
#    limit of the GIF format renders poorly at this scene's dynamic range, so
#    the movie is written as an MP4 via a separate pre-build step and embedded
#    below using a raw HTML ``<video>`` tag.  To regenerate the MP4 locally,
#    run the script directly (without the ``SPHINX_GALLERY_BUILD`` environment
#    variable set).

if not os.environ.get("SPHINX_GALLERY_BUILD"):
	# The following lines are included to standardize the output path for the sphinx-gallery
	# pre-processing pipeline mentioned above. These values can be omitted/altered if running
	# the script directly.

	output_dir = Path(os.environ.get("STATIC_ASSETS", "")).resolve()
	movie_name = "p07_faux_volume_render.mp4"
	screenshot_name = "p07_faux_volume_render.png"

	plotter = Plot3d()
	plotter.add_axes()
	plotter.add_mesh(radial_slice_at_photosphere, show_scalar_bar=False)
	plotter.add_mesh(deconstructed_mesh_volume, opacity="sigmoid_7", show_scalar_bar=False)
	plotter.open_movie(output_dir / movie_name, framerate=10)
	for position in trajectory.T:
		plotter.observer_position = position
		plotter.observer_los_view = -50, 50, -45, 45
		plotter.write_frame()
	plotter.screenshot(output_dir / screenshot_name)
	plotter.close()

# %%
# .. raw:: html
#
#    <video autoplay loop muted playsinline style="width:100%;border-radius:4px;">
#      <source src="../../_static/assets/p07_faux_volume_render.mp4" type="video/mp4">
#    </video>
