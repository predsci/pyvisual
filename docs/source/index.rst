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

**pyvisual** is in active development. Although the existing API will remain unchanged,
additional quality-of-life improvements are forthcoming. Please direct any comments,
concerns, or issues to the Predictive Science Inc. research team via the distro `Issue Tracker
<https://github.com/predsci/pyvisual/issues>`_ or
`contact form <https://www.predsci.com/portal/contact.php>`_.

.. attention::

   Please be aware that **pyvisual** is available through the Python Package Index (PyPI) as
   ``psi-pyvisual``. Further installation instructions are available in the
   `User Guide <https://predsci.com/doc/pyvisual/guide/index.html>`_.

.. grid:: 3
   :gutter: 2
   :class-container: sd-mt-4

   .. grid-item-card:: User Guide
      :link: guide/index
      :link-type: doc
      :text-align: center

      Installation instructions, coordinate conventions, and an architectural
      overview of the :class:`~pyvisual.core.plot3d.Plot3d` class and its
      mixin components.

   .. grid-item-card:: Examples
      :link: gallery/index
      :link-type: doc
      :text-align: center

      Worked examples covering basic scenes, slicing, fieldlines, observer
      controls, and the mesh class API — organized by functional area.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc
      :text-align: center

      Full API documentation for all public classes, methods, functions,
      and type aliases auto-generated from source docstrings.

.. toctree::
    :hidden:
    :maxdepth: 2

    Guide <guide/index>
    Examples <gallery/index>
    API <api/index>
