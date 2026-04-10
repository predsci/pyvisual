.. _gallery:

Examples
========

The examples below are organized into galleries that mirror the functional
areas of the :class:`~pyvisual.core.plot3d.Plot3d` API.  Each gallery is
self-contained; to run the complete suite of examples, it is recommended to
install the optional dependencies, *viz.*:

.. code-block:: bash

   pip install "psi-pyvisual[tracing,data]"

.. seealso::

   :ref:`overview`
      Conceptual introduction to the coordinate conventions and class
      hierarchy used throughout **pyvisual**.

   :ref:`installation`
      Installation instructions for **pyvisual**, along with a list of
      its optional dependencies.

   :doc:`../api/index`
      Full API reference for all public classes and functions (including
      a variety of additional examples).

Data Access
-----------

Examples that use Predictive Science Inc's MAS model data call
:func:`~pyvisual.utils.data.fetch_datasets`, which downloads a small
set of data files from the PSI asset server on first run and
caches them under ``~/.cache/psi/`` (or ``$PYVISUAL_CACHE`` if set).
Examples that use only NumPy arrays run without any network access.

MHDweb
------

These data-fetching routines (:mod:`pyvisual.utils.data`) are
designed to provide ready access to a sample MAS run for *testing
and demonstration purposes*.

**For access to the full catalogue of Predictive Science Inc's publically
available** `MAS <https://www.predsci.com/corona/model_desc.html>`_
**(Magnetodydrodynamic Algorithm outside a Sphere) model
solutions – over 2100 runs spanning nearly 5 decades – please visit**
`MHDweb <https://predsci.com/mhdweb2/>`_.

**This web-based data-visualization suite includes a wide variety of
highly customizable data products which can be downloaded directly through
the web interface** (*e.g.*
`Spacecraft Mapping <https://predsci.com/mhdweb2/tools/spacecraft-mapping>`_,
`White Light Imagers <https://predsci.com/mhdweb2/tools/white-light>`_,
`EUV & X-Ray Emissions <https://predsci.com/mhdweb2/tools/euv>`_)
**or programmatically accessed through the** `MHDweb API <https://predsci.com/mhdweb2/api>`_.

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
