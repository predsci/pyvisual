Examples
========

The examples below are organized into galleries that mirror the functional
areas of the :class:`~pyvisual.core.plot3d.Plot3d` API.  Each gallery is
self-contained and can be run locally after a development install
(``pip install -e ".[all]"``), or simply by using the standard install
*i.e.* ``pip install psi-pyvisual``.

.. note::

   Examples that use real coronal model data call
   :func:`~pyvisual.utils.data.fetch_datasets`, which downloads a small
   set of PSI MAS model files from the PSI asset server on first run and
   caches them under ``~/.cache/psi/`` (or ``$PYVISUAL_CACHE`` if set).
   Examples that use only NumPy arrays run without any network access.

.. seealso::

   :ref:`overview`
      Conceptual introduction to the coordinate conventions and class
      hierarchy used throughout **pyvisual**.

   :doc:`/api/index`
      Full API reference for all public classes and functions.

.. toctree::
    :maxdepth: 2
    :hidden:

    01_getting_started/index
    02_stack_mesh_mixin/index
    03_grid_mesh_mixin/index
    04_observer_mixin/index
    05_geometry_mixin/index
    06_spherical_grid_class/index
    07_cartesian_grid_class/index
    99_advanced_plots/index
