"""
:class:`Plot3d` — the primary 3D plotter for solar-physics visualization.

This module defines :class:`Plot3d`, a thin wrapper around :class:`pyvista.Plotter`
that composes four mixin classes to expose a unified API for MHD solar dataset
rendering:

- :class:`~pyvisual.core.mixins.ObserverMixin` — spherical-coordinate camera
  controls, line-of-sight FOV, and live camera-state readout.
- :class:`~pyvisual.core.mixins.GeometryMixin` — solar-geometry primitives (Sun
  sphere, shells, discs, longitude/latitude grid lines, Thompson sphere).
- :class:`~pyvisual.core.mixins.GridMeshMixin` — 1-D, 2-D, and 3-D structured-grid
  slices and isosurface contours from spherical coordinate arrays.
- :class:`~pyvisual.core.mixins.StackMeshMixin` — points, splines, surfaces, and
  magnetic fieldlines from stacked-coordinate arrays.

The class overrides :meth:`pyvista.Plotter.add_mesh` and
:meth:`pyvista.Plotter.add_composite` to accept an optional ``frame`` keyword
argument.  When ``frame`` is not ``'cartesian'``, the mesh points are converted to
Cartesian coordinates via :func:`~pyvisual.core.parsers.apply_mesh_transform` before
being forwarded to the PyVista render pipeline.

See Also
--------
:class:`pyvista.Plotter`
    The upstream plotter that :class:`Plot3d` extends.
:mod:`pyvisual.core.mixins`
    Full API for all four mixin classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyvista as pv

from pyvisual.core.mixins import (
    ObserverMixin,
    GeometryMixin,
    GridMeshMixin,
    StackMeshMixin)
from pyvisual.core.parsers import (
    validate_mesh_type,
    apply_mesh_transform)


class Plot3d(ObserverMixin, GeometryMixin, GridMeshMixin, StackMeshMixin, pv.Plotter):
    """
    PyVista Plotter extension for spherical data.

    This class is designed to be a thin wrapper around PyVista's :class:`~pyvista.Plotter` class.
    Its principal concern is to facilitate the visualization of datasets defined on a spherical
    coordinate system; more specifically, **pyvisual** is aimed at visualizing solar modeling
    data in an efficient, interactive way. This package is developed and maintained by `Predictive Science Inc.
    <https://www.predsci.com/>`_ and therefore is specifically tuned to work with the PSI data
    ecosystem. For additional information regarding the conventions and packages that comprise
    this ecosystem, visit the :ref:`overview` section.

    .. attention::
       As is noted above, **pyvisual** is a thin wrapper for the :class:`pyvista.Plotter` class.
       **PyVista** itself is a pythonic wrapper for the C++ **Visualization Toolkit** software.
       Both of these phenomenal open-source libraries possess extensive documentation and resources;
       it is highly recommended to familiarize oneself with (*viz.*) **PyVista** to get the most
       out of the **pyvisual** package in specific.


    References
    ----------
    `PyVista <https://docs.pyvista.org/>`_

    `Visualization Toolkit <https://vtk.org/>`_

    """

    PLOTTER_FRAME = 'cartesian'

    def __init__(self, *args, **kwargs):
        """
        Initialise :class:`Plot3d` and the internal camera-state observer handle.

        All positional and keyword arguments are forwarded verbatim to
        :class:`pyvista.Plotter`.

        Parameters
        ----------
        *args
            Positional arguments passed to :class:`pyvista.Plotter`.
        **kwargs
            Keyword arguments passed to :class:`pyvista.Plotter`.

        Examples
        --------
        .. pyvista-plot::

            >>> import pyvisual as pv
            >>> pl = pv.Plot3d(window_size=(800, 600))
            >>> pl.add_sun()
            >>> pl.show()
        """
        super().__init__(*args, **kwargs)
        self._camera_update_observer = None

    def _adjust_scalar_bars(self):
        """Reposition all scalar bars and center-align their annotation text.

        Iterates over the current scalar bars and places them in a vertical
        stack starting at a normalised viewport height of 0.05, incrementing
        by 0.1 per bar.  Annotation text is centered both horizontally and
        vertically.
        """
        scalar_bars = dict(self.scalar_bars)
        h = 0.05
        for v in scalar_bars.values():
            v.SetPosition(0.2, h)
            annotations = v.GetAnnotationTextProperty()
            annotations.SetJustificationToCentered()
            annotations.SetVerticalJustificationToCentered()
            h += 0.1


    def add_composite(self,
                      dataset,
                      *args,
                      frame: Optional[str] = None,
                      **kwargs):
        """Add a composite (multi-block) dataset to the scene, converting coordinates if needed.

        Extends :meth:`pyvista.Plotter.add_composite` with an optional ``frame``
        argument.  If ``dataset`` resolves to a :class:`pyvista.MultiBlock` or
        :class:`pyvista.PartitionedDataSet`, its points are converted from ``frame``
        to the plotter's Cartesian frame via
        :func:`~pyvisual.core.parsers.apply_mesh_transform` before rendering.
        Non-composite types are forwarded directly to the parent implementation.

        Parameters
        ----------
        dataset : PlottableType
            The composite dataset to add.
        *args
            Additional positional arguments forwarded to
            :meth:`pyvista.Plotter.add_composite`.
        frame : str | None, optional
            Coordinate frame of ``dataset``.  Any alias accepted by
            :func:`~pyvisual.core.parsers.fetch_canonical_frame`.  If ``None``, the
            frame stored in ``dataset.user_dict['MESH_FRAME']`` is used (if present).
            Default is ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`pyvista.Plotter.add_composite`.

        Returns
        -------
        out : pyvista.Actor
            The actor returned by the underlying PyVista plotter call.
        """
        mesh = validate_mesh_type(dataset)
        if not isinstance(mesh, (pv.MultiBlock, pv.PartitionedDataSet)):
            return super().add_composite(mesh, *args, **kwargs)
        mesh = apply_mesh_transform(mesh, frame, Plot3d.PLOTTER_FRAME)
        return super().add_composite(mesh, *args, **kwargs)


    def add_mesh(self,
                 mesh,
                 *args,
                 frame: Optional[str] = None,
                 **kwargs):
        """Add a mesh to the scene, converting coordinates from ``frame`` if needed.

        Extends :meth:`pyvista.Plotter.add_mesh` with an optional ``frame``
        argument.  File-path strings are read with :func:`pyvista.read` before
        processing.  Composite meshes are routed through
        :meth:`add_composite`; single meshes have their points converted to the
        plotter Cartesian frame via
        :func:`~pyvisual.core.parsers.apply_mesh_transform` when ``frame`` is set.

        Parameters
        ----------
        mesh : PlottableType | str | Path
            The mesh to add.  Accepts any type supported by
            :meth:`pyvista.Plotter.add_mesh`, plus file-path strings or
            :class:`pathlib.Path` objects that are read automatically.
        *args
            Additional positional arguments forwarded to
            :meth:`pyvista.Plotter.add_mesh`.
        frame : str | None, optional
            Coordinate frame of ``mesh``.  Any alias accepted by
            :func:`~pyvisual.core.parsers.fetch_canonical_frame`.  If ``None``, the
            frame stored in ``mesh.user_dict['MESH_FRAME']`` is used (if present).
            Default is ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`pyvista.Plotter.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The actor returned by the underlying PyVista plotter call.

        Examples
        --------
        Add a :class:`~pyvisual.core.mesh3d.SphericalMesh` directly — its frame is
        stored in ``user_dict`` and picked up automatically:

        .. pyvista-plot::

            >>> import pyvisual as pv
            >>> from pyvisual.core.mesh3d import SphericalMesh
            >>> import numpy as np
            >>> r = np.linspace(1, 5, 20)
            >>> t = np.linspace(0, np.pi, 30)
            >>> p = np.linspace(0, 2 * np.pi, 60)
            >>> mesh = SphericalMesh(r, t, p)
            >>> pl = pv.Plot3d()
            >>> pl.add_mesh(mesh, color='orange', opacity=0.5)
            >>> pl.show()
        """
        if isinstance(mesh, (str, Path)):
            mesh = pv.read(mesh)  # type: ignore[assignment]
        mesh = validate_mesh_type(mesh)
        if isinstance(mesh, (pv.MultiBlock, pv.PartitionedDataSet)):
            return super().add_mesh(mesh, *args, frame=frame, **kwargs)
        mesh = apply_mesh_transform(mesh, frame, Plot3d.PLOTTER_FRAME)
        return super().add_mesh(mesh, *args, **kwargs)