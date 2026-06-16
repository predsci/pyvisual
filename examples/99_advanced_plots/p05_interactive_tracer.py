r"""
Interactive Fieldline Tracer
=============================

This example builds a two-panel interactive scene for exploring coronal magnetic
connectivity.  The top panel displays a longitude–latitude map of :math:`B_r`
at :math:`r = 1\,R_\odot`; right-clicking any point triggers a callback that
constructs a :math:`5 \times 5` grid of launch points spanning
:math:`\pm 1^\circ` around the selection, traces fieldlines forward from
those seeds via :class:`~mapflpy.tracer.Tracer`, and renders the result in the
bottom 3-D panel.  A rolling buffer of :data:`BUFFER_SIZE` named actor groups
keeps the scene from accumulating unbounded trace bundles.

.. note::

   This example requires an interactive VTK window.  The Sphinx-Gallery
   thumbnail shows only the initial unpicked scene; run the script locally
   to use the tracer.

.. seealso::

   :ref:`sphx_glr_gallery_02_stack_mesh_mixin_p03_fieldlines.py`
      Introduction to :meth:`~pyvisual.core.mixins.StackMeshMixin.add_fieldlines`
      and fieldline coloring strategies.

   :ref:`sphx_glr_gallery_99_advanced_plots_p03_contour_based_fieldlines.py`
      Seeding fieldlines from the polarity inversion line using a
      :class:`~pyvisual.core.mesh3d.SphericalMesh` contour.

   :ref:`sphx_glr_gallery_99_advanced_plots_p01_combining_multiple_elements.py`
      Layering slices, contours, and fieldlines in a single coronal scene.
"""

import pyvista as pv
from pyvisual import Plot3d, SphericalMesh
from psi_data import fetch_mas_data
from psi_io import np_interpolate_slice_from_hdf
from mapflpy.tracer import Tracer
import numpy as np

# %%
# Load Data
# ---------
#
# :func:`psi_data.fetch_mas_data` returns paths to the CR 2309
# coronal field components :math:`(B_r, B_\theta, B_\phi)`.
# :func:`~psi_io.np_interpolate_slice_from_hdf` interpolates a 2-D
# :math:`(\theta, \phi)` shell of :math:`B_r` at radial index ``1``
# (:math:`r = 1\,R_\odot`) and returns the scalar array with its coordinate axes.

data = fetch_mas_data(domains="cor", variables=["br", "bt", "bp"])
br, *scales = np_interpolate_slice_from_hdf(data.cor_br, 1, None, None)
br_t, br_p = scales

# %%
# Initialise the Tracer and Buffer Parameters
# -------------------------------------------
#
# :class:`~mapflpy.tracer.Tracer` is constructed once so the HDF field files
# are loaded only once rather than on every pick.  :data:`DELTA` sets the
# half-width of the launch-point neighbourhood to
# :math:`1^\circ = \pi/180\,\mathrm{rad}`, and :data:`BUFFER_SIZE` controls
# how many trace bundles are kept in the scene at any time.

tracer = Tracer(*data)
NPOINTS = 5
BUFFER_SIZE = 3
DELTA = np.pi / 180

# %%
# Define the Pick Callback
# -------------------------
#
# :meth:`pyvista.Plotter.enable_point_picking` passes the picked
# :math:`(r, \theta, \phi)` to ``_plotter_callback``.  The callback stamps the
# coordinates as a text overlay in subplot ``(0, 0)``, then builds a
# :math:`N_\mathrm{pts}^2 = 25` launch-point grid via :func:`numpy.meshgrid`
# spanning :math:`[\theta \pm \Delta] \times [\phi \pm \Delta]` at fixed
# :math:`r`.  :meth:`~mapflpy.tracer.Tracer.trace_fwd` returns geometry of
# shape :math:`(M, 3, N)`; :func:`numpy.swapaxes` transposes it to
# :math:`(3, M, N)` for :meth:`~pyvisual.core.mixins.StackMeshMixin.add_fieldlines`.
# The actor name cycles modulo :data:`BUFFER_SIZE` so PyVista replaces older
# bundles rather than accumulating them.

def _plotter_callback(pt):
    plotter.subplot(0, 0)
    plotter.buffer += 1
    r, t, p = pt
    plotter.add_text(f"Selected Point: r={r:.2f}, t={t:.2f}, p={p:.2f}",
                     name="picked_point_text",
                     color='cyan',
                     font_size=12,
                     position='lower_right')

    rr, tt, pp = np.meshgrid(r,
                             np.linspace(t - DELTA, t + DELTA, NPOINTS),
                             np.linspace(p - DELTA, p + DELTA, NPOINTS),
                             indexing='ij')
    lps = np.stack((rr.ravel(), tt.ravel(), pp.ravel()), axis=0)
    fieldlines, *_ = tracer.trace_fwd(launch_points=lps)

    plotter.subplot(1, 0)
    plotter.add_points(*lps, color='white', point_size=7, name="Launch Points")
    plotter.add_fieldlines(*np.swapaxes(fieldlines, 1, 0),
                           np.arange(lps.shape[1]),
                           cmap='hsv',
                           name=f'Fieldlines{plotter.buffer % BUFFER_SIZE}',
                           show_scalar_bar=False,
                           line_width=5)

# %%
# Build and Launch the Scene
# --------------------------
#
# ``shape=(2, 1)`` stacks two panels vertically.  The bottom panel ``(1, 0)``
# holds a non-pickable 2-D :math:`B_r` shell as a fieldline backdrop.  The top
# panel ``(0, 0)`` holds a :class:`~pyvisual.core.mesh3d.SphericalMesh` of the
# same data as the interactive picking surface — ``user_dict`` is cleared to
# preserve the native :math:`(\theta, \phi)` coordinate layout that the
# callback expects.  :func:`pyvista.set_new_attribute` attaches the ``buffer``
# counter to the plotter for use inside the callback closure.

plotter = Plot3d(shape=(2, 1))
pv.set_new_attribute(plotter, "buffer", 0)

plotter.subplot(1, 0)
plotter.add_2d_slice(1, br_t, br_p, br,
                     clim=(-10, 10),
                     cmap="seismic",
                     show_scalar_bar=False,
                     pickable=False)

plotter.subplot(0, 0)
longlat_map = SphericalMesh(1, br_t, br_p, data=br)

# Must clear the user_dict's ``MESH_FRAME`` tag so the ``add_mesh`` function doesn't try to
# convert the coordinates from spherical to cartesian — the callback expects the native
# (r, t, p) layout for the picks and launch points.
longlat_map.user_dict.clear()

plotter.add_mesh(longlat_map,
                 clim=(-10, 10),
                 cmap="seismic",
                 show_scalar_bar=False,
                 pickable=True)
plotter.add_axes()

# Adjust the longitude-latitude display so that it is oriented in a way that directly maps
# to the lower panel's :math:`B_r` shell.
plotter.view_yz()
plotter.camera.up = 1, -1, 0

plotter.enable_point_picking(_plotter_callback,
                             color='cyan',
                             font_size=12,
                             point_size=8,
                             render_points_as_spheres=True)

plotter.show()