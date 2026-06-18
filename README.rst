.. |psi| image:: https://predsci.com/doc/psi_logo.png
   :target: https://predsci.com
   :alt: Predictive Science Inc.
   :width: 20px

.. |pypi| image:: https://img.shields.io/pypi/v/psi-pyvisual?logo=pypi&logoColor=white
   :target: https://pypi.org/project/psi-pyvisual
   :alt: PyPI

.. |license| image:: https://img.shields.io/pypi/l/psi-pyvisual?logo=apache&logoColor=white
   :target: https://opensource.org/license/apache-2-0/
   :alt: License

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/psi-pyvisual.svg?logo=python&label=python&logoColor=white
   :target: https://pypi.org/project/psi-pyvisual
   :alt: Python Versions

.. |deps| image:: https://img.shields.io/librariesio/github/predsci/pyvisual?logo=Libraries.io&logoColor=white
   :target: https://github.com/predsci/pyvisual/blob/main/pyproject.toml
   :alt: Libraries.io

|pypi|
|license|
|pyversions|
|deps|

|psi| **PYVISUAL** | *3D Visualizations for Spherical Coordinate Systems*
-------------------------------------------------------------------------

**pyvisual** is developed and maintained by `Predictive Science Inc. (PSI)
<https://www.predsci.com/>`_. Its principal concern is the visualization
of solar and magnetohydrodynamic (MHD) model output defined on spherical
coordinate systems. The package is tightly coupled to the PSI data ecosystem
and is tuned for use with **psi-io** and **mapflpy** (although any model
defined on a rectilinear grid in spherical coordinates is compatible with
**pyvisual**'s API).

**pyvisual** is a thin wrapper around the **PyVista** package – a powerful and
flexible (high-level) python interface for the Visualization Toolkit (VTK) library.
It is **STRONGLY** recommended to visit the exhaustive
`PyVista documentation <https://docs.pyvista.org/>`_ (along with
their `examples <https://docs.pyvista.org/examples/index.html>`_) to get a better
understanding of the underlying capabilities of the package. **pyvisual** is
intentionally limited in scope – tailored for use with PSI's data ecosystem.
For a more robust solar physics visualization package, consider using **SunPy**'s
`sunkit-pyvista <https://docs.sunpy.org/projects/sunkit-pyvista/en/latest/index.html>`_
subpackage *viz.* for coordinate-aware 3D visualizations.

To get started with **pyvisual**, visit the
`User Guide <https://predsci.com/doc/pyvisual/guide/>`_ for installation instructions,
an overview of features, and development/contribution guidelines; a gallery of
`examples <https://predsci.com/doc/pyvisual/gallery/>`_ is also available, showcasing
various use cases and functionalities of the package. Please direct any questions or
issues to the `issue tracker <https://github.com/predsci/pyvisual/issues>`_,
or `contact <https://www.predsci.com/portal/contact.php>`_ Predictive Science Inc. directly.

----

`Predictive Science Inc. <https://predsci.com>`_ |
`Repository <https://github.com/predsci/pyvisual>`_ |
`Documentation <https://predsci.com/doc/pyvisual>`_ |
`Distribution <https://pypi.org/project/psi-pyvisual>`_
