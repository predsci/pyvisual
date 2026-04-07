.. _installation:

Installation
============

Requirements
------------

**pyvisual** requires Python 3.10 or later and runs on Linux and macOS.
The core dependencies — :mod:`numpy`, :mod:`pyvista`, :mod:`psi_io`,
:mod:`sunpy`, and :mod:`astropy` — are installed automatically.

Standard Install
----------------

Install the latest release from PyPI::

    pip install pyvisual

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

    pip install "pyvisual[hdf4,tracing]"

Development Install
-------------------

Clone the repository and install in editable mode with all optional
dependencies::

    git clone https://bitbucket.org/predsci/pyvisual.git
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