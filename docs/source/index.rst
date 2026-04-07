pyvisual Documentation
======================

**pyvisual** is developed and maintained by `Predictive Science Inc. (PSI)
<https://www.predsci.com/>`_ and provides an interactive 3D visualization layer
for solar and magnetohydrodynamic (MHD) model output defined on spherical coordinate
systems. It wraps :class:`pyvista.Plotter` with spherical-coordinate awareness,
observer controls, and rendering utilities tuned to the PSI data ecosystem
(`psi-io <https://predsci.com/doc/psi-io/guide/index.html>`_ and
`mapflpy <https://predsci.com/doc/mapflpy/>`_).

The primary entry point is :class:`~pyvisual.core.plot3d.Plot3d`, which exposes
methods for adding solar geometry primitives, rendering structured-grid slices and
isosurface contours, plotting point clouds and splines, and tracing magnetic
fieldlines — all from spherical :math:`(r, \theta, \phi)` coordinate arrays.
Because **pyvisual** renders through `PyVista <https://docs.pyvista.org/>`_ and VTK,
it is strongly recommended to familiarise yourself with the PyVista documentation to
get the most out of the package.

.. toctree::
    :hidden:

    API <api/index>
    Guide <guide/index>
    Examples <gallery/index>