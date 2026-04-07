Examples
========

The examples below are organized into galleries that mirror the functional
areas of the :class:`~pyvisual.core.plot3d.Plot3d` API.  Each gallery is
self-contained and can be run locally after a development install
(``pip install -e ".[all]"``).

.. note::

   Examples that use real coronal model data call
   :func:`~pyvisual.utils.data.fetch_datasets`, which downloads a small
   set of PSI MAS model files from the PSI asset server on first run and
   caches them under ``~/.cache/psi/`` (or ``$PYVISUAL_CACHE`` if set).
   Examples that use only NumPy arrays run without any network access.

Gallery Organization
--------------------

:ref:`sphx_glr_gallery_01_getting_started`
   Basic :class:`~pyvisual.core.plot3d.Plot3d` usage: instantiating the
   plotter, adding the Sun and reference geometry, and controlling actor
   visibility.

:ref:`sphx_glr_gallery_02_stack_mesh_mixin`
   Rendering from *stacked* coordinate arrays via
   :class:`~pyvisual.core.mixins.StackMeshMixin`: individual points, point
   clouds, single and bundled splines, and reconstructed surfaces.

:ref:`sphx_glr_gallery_03_grid_mesh_mixin`
   Rendering from *structured-grid* coordinate arrays via
   :class:`~pyvisual.core.mixins.GridMeshMixin`: 1-D, 2-D, and 3-D slices,
   isosurface contours, and magnetic fieldline tracing with
   :mod:`mapflpy`.

:ref:`sphx_glr_gallery_04_observer_mixin`
   Camera positioning and field-of-view control via
   :class:`~pyvisual.core.mixins.ObserverMixin`: spherical-coordinate
   observer placement, saving and restoring views, and orbit animations.

:ref:`sphx_glr_gallery_05_geometry_mixin`
   Solar geometry primitives via
   :class:`~pyvisual.core.mixins.GeometryMixin`: concentric shells, planar
   discs, the Thomson sphere, and structured spline grid lines.

:ref:`sphx_glr_gallery_06_spherical_grid_class`
   The :class:`~pyvisual.core.mesh3d.SphericalMesh` data container:
   construction from NumPy arrays, arithmetic and NumPy ufunc support, and
   filter methods for radial scaling and coordinate remapping.

:ref:`sphx_glr_gallery_07_cartesian_grid_class`
   The :class:`~pyvisual.core.mesh3d.CartesianMesh` data container:
   construction from meshgrid arrays and coordinate-frame conversion from
   a :class:`~pyvisual.core.mesh3d.SphericalMesh`.

:ref:`sphx_glr_gallery_08_extended_functionalities`
   Advanced observer controls: helioprojective field-of-view specification
   via angular extents (:attr:`~pyvisual.core.mixins.ObserverMixin.observer_los_view`)
   and minimum line-of-sight impact radius
   (:attr:`~pyvisual.core.mixins.ObserverMixin.observer_fov_view`).

:ref:`sphx_glr_gallery_99_advanced_plots`
   Multi-element scenes combining fieldline tracing, slices, contours,
   and interactive point-picking callbacks.

.. seealso::

   :ref:`overview`
      Conceptual introduction to the coordinate conventions and class
      hierarchy used throughout **pyvisual**.

   :doc:`/api/index`
      Full API reference for all public classes and functions.
