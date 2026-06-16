.. _installation:

Installation
============

.. attention::

    We highly recommend using a virtual environment to manage your
    Python packages and avoid conflicts with other projects. For
    the best results, we recommend using ``conda`` – *via* Miniforge
    (preferred), Miniconda, or Anaconda – to create and manage
    your virtual environments.

To get started with **pyvisual**, you can install it directly from PyPI:

.. code-block:: bash

    pip install psi-pyvisual

To run the :ref:`gallery` suite, include the following add-ons:

.. code-block:: bash

    pip install psi-pyvisual[tracing,data]

Requirements
------------

**pyvisual** requires Python 3.10 or later and runs on Linux and macOS.

Core Dependencies
^^^^^^^^^^^^^^^^^

The following packages are installed automatically:

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Package
     - Min. version
     - Role in **pyvisual**
   * - `NumPy <https://numpy.org/doc/stable/>`_
     - 2.1.0
     - Array operations underpinning all coordinate transforms, mesh
       construction, and data manipulation throughout the library.
   * - `PyVista <https://docs.pyvista.org/>`_
     - 0.46.0
     - 3-D rendering engine.  **pyvisual** subclasses
       :class:`pyvista.Plotter` and uses PyVista mesh types
       (:class:`~pyvista.PolyData`, :class:`~pyvista.StructuredGrid`,
       :class:`~pyvista.RectilinearGrid`) as the primary data containers
       for all visualizations.
   * - `psi-io <https://predsci.com/doc/psi-io/guide/index.html>`_
     - 2.0.6
     - PSI library for reading HDF4/HDF5 model output files. Provides *e.g.*
       :func:`~psi_io.psi_io.read_hdf_by_index` for loading subsets of
       HDF datasets by index (without having to load the entire file
       into memory), and is required for all examples that use
       PSI's MAS model data.
   * - `SunPy <https://docs.sunpy.org/>`_
     - 6.0.3
     - Solar physics toolkit used for coordinate frame transformations
       (e.g. Heliocentric Earth Ecliptic and Carrington frames) and
       spacecraft ephemeris calculations that drive the observer
       positioning utilities.
   * - `Astropy <https://docs.astropy.org/>`_
     - 6.1.3
     - Astronomical units, time handling, and coordinate infrastructure
       underlying SunPy's frame system; used indirectly via SunPy for
       JPL Horizons queries and observer geometry.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

Install these via extras (see `Standard Install`_ below):

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Package
     - Min. version
     - Role in **pyvisual**
   * - `pyhdf <https://pypi.org/project/pyhdf/>`_
     - 0.11.6
     - HDF4 file support.  Required when PSI model data is stored in the
       legacy ``.hdf`` format rather than HDF5.  Enabled by the
       ``hdf4`` extra.
   * - `SciPy <https://docs.scipy.org/doc/scipy/>`_
     - —
     - Interpolation routines for regridding or resampling data onto
       non-native coordinate grids.  Enabled by the ``interp`` extra
       (also pulled in by ``data``).
   * - `mapflpy <https://predsci.com/doc/mapflpy/>`_
     - 1.1.9
     - PSI library for tracing magnetic fieldlines on spherical grids.
       Provides :func:`~mapflpy.scripts.run_forward_tracing` and
       :func:`~mapflpy.scripts.run_fwdbwd_tracing`, which feed directly
       into :meth:`~pyvisual.core.mixins.StackMeshMixin.add_fieldlines`.
       Enabled by the ``tracing`` extra.
   * - `psi-data-utils <https://pypi.org/project/psi-data-utils/>`_
     - 1.0.0
     - PSI sample-data fetching and caching (imported as :mod:`psi_data`).
       Provides :func:`psi_data.fetch_mas_data`, which downloads sample PSI
       MAS model files (Carrington rotation 2309 HMI-driven MAS standard run)
       to ``~/.cache/psi/`` on first use.  Enabled by the ``data`` extra.
   * - `Matplotlib <https://matplotlib.org/stable/>`_
     - 3.10.7
     - Colour maps and scalar-bar rendering used internally by PyVista.
       Pulled in automatically by the ``data`` extra.

Standard Install
----------------

Install the latest release from PyPI::

    pip install psi-pyvisual

To include support for reading HDF4 files (``pyhdf``), fieldline tracing
(``mapflpy``), scipy-based interpolation, or the data-fetching utilities,
use the relevant extras:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - What it adds
   * - ``hdf4``
     - HDF4 file support via ``pyhdf``
   * - ``interp``
     - Interpolation utilities via ``scipy``
   * - ``tracing``
     - Magnetic fieldline tracing via ``mapflpy``
   * - ``data``
     - Asset fetching via ``pooch``; also installs ``scipy`` and ``matplotlib``

Install one or more extras with::

    pip install "psi-pyvisual[hdf4,tracing]"

Development Install
-------------------

Clone the repository and install in editable mode with all optional
dependencies::

    git clone https://github.com/predsci/pyvisual.git
    cd pyvisual
    pip install -e ".[all]"

Verify the installation by importing the package::

    python -c "import pyvisual; print(pyvisual.__version__)"

Development tools
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Purpose
   * - ``ruff check .``
     - Linting
   * - ``mypy .``
     - Type checking
   * - ``pytest``
     - Test suite
   * - ``pytest --cov``
     - Tests with coverage report
   * - ``cd docs && make html``
     - Build HTML documentation

.. seealso::

   :ref:`overview`
      An introduction to **pyvisual**'s capabilities and architecture.