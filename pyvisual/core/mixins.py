"""
Mixin classes that compose the four functional areas of :class:`~pyvisual.core.plot3d.Plot3d`.

All four mixins are defined in this module and are assembled into
:class:`~pyvisual.core.plot3d.Plot3d` via multiple inheritance.

Mixin responsibilities
----------------------
:class:`ObserverMixin`
    Camera and observer controls in spherical coordinates — position, focal point,
    up vector, position angle, line-of-sight FOV, and a live camera-state readout.
:class:`GeometryMixin`
    Solar geometry primitives — Sun sphere, spherical shells, planar discs,
    longitude/latitude grid lines, and the Thomson sphere.
:class:`GridMeshMixin`
    Structured-grid rendering from independent 1-D or pre-broadcast 3-D coordinate
    arrays: 1-D line slices, 2-D surface slices, 3-D volume slices, and
    isosurface contours.
:class:`StackMeshMixin`
    Rendering from stacked N-D coordinate arrays (all axes share the same shape):
    single points, point clouds, single splines, spline bundles, magnetic
    fieldlines, and free-form surfaces.

Common patterns
---------------
Methods in :class:`GridMeshMixin` and :class:`StackMeshMixin` that accept
``r``, ``t``, ``p`` positional arguments are decorated with
:func:`~pyvisual.core.parsers.parse_mesh_params`, which promotes scalar inputs
to 1-D arrays before the method body runs.

Keyword arguments for ``add_mesh`` calls are built by merging immutable defaults
from :mod:`pyvisual.core._styling` with caller-supplied overrides using the ``|``
operator.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import wraps
from itertools import pairwise
from typing import Optional, Literal

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike

from pyvisual.core.constants import SOLAR_NORTH
from pyvisual.core.mesh3d import (
    build_spline_polydata,
    build_slice_polydata,
    build_point_polydata,
    build_surface_polydata,
    SphericalMesh)
from pyvisual.core.parsers import (
    parse_mesh_params, 
    parse_stack_mesh, 
    parse_grid_mesh, 
    parse_data)
from pyvisual.core._styling import (
    COLORMAP_KWARGS,
    SOLID_COLOR_KWARGS,
    POINTS_KWARGS,
    SPLINES_KWARGS,
    SLICES_KWARGS,
    RANDOM_COLORING_DEFAULTS,
    FL_POLARITY_COLORING_DEFAULTS,
    FIELDLINE_KWARGS,)
from pyvisual.core._typing import (
    FlColorType,
    SphericalCoordinate,
    ObserverOrientation,
    SurfaceReconstructionType, )
from pyvisual.utils.geometry import (
    ij_meshgrid,
    cartesian_to_spherical,
    spherical_to_cartesian,
    spherical_to_cartesian_vec,
    clip_angle,
    thompson_sphere,
    los_rmin2angle,
    camera_roll_wrt_solar_north,)


def render_scene(func):
    """Decorator that calls ``self.render()`` after the wrapped method returns.

    Silently ignores :class:`AttributeError` so the decorator is safe to use on
    methods called before the plotter's render window is fully initialised.

    Parameters
    ----------
    func : Callable
        A :class:`~pyvisual.core.plot3d.Plot3d` method (typically a property
        setter) that modifies camera state and should trigger an immediate
        re-render.

    Returns
    -------
    out : Callable
        The wrapped method.

    Examples
    --------
    >>> from pyvisual.core.mixins import render_scene
    >>> class MockPlotter:
    ...     @render_scene
    ...     def set_something(self):
    ...         pass
    ...     def render(self):
    ...         print("rendered")
    >>> MockPlotter().set_something()
    rendered
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        try:
            self.render()
        except AttributeError:
            pass
        return result
    return wrapper


class StackMeshMixin:
    """Mixin for rendering geometry from stacked N-D coordinate arrays.

    "Stack" arrays share the same shape across all three coordinates ``r``,
    ``t``, ``p`` — they represent individual points, spline paths, or
    surface patches where every element of the array corresponds to one
    spatial location.

    All public methods accept an optional scalar ``data`` array and
    forward remaining ``**kwargs`` to :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

    See Also
    --------
    :class:`GridMeshMixin`
        Companion mixin for independent 1-D axis arrays.
    """

    @parse_mesh_params
    def add_point(self,
                   r: ArrayLike,
                   t: ArrayLike,
                   p: ArrayLike,
                   data: Optional[ArrayLike] = None,
                   /,
                   dataid: str = 'Data',
                   **kwargs):
        """Add a single point at spherical coordinates :math:`(r, \\theta, \\phi)`.

        Parameters
        ----------
        r : ArrayLike
            Radial distance in solar radii :math:`R_\\odot`.
        t : ArrayLike
            Colatitude :math:`\\theta` in radians.
        p : ArrayLike
            Longitude :math:`\\phi` in radians.
        data : ArrayLike | None, optional
            Scalar value at this point.  Default is ``None`` (solid color).
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered point actor.

        Raises
        ------
        ValueError
            If any of ``r``, ``t``, ``p`` has size greater than 1.

        Examples
        --------
        Add a single point at :math:`r = 1.5 R_\\odot`, :math:`\\theta = \\pi/2`, :math:`\\phi = \\pi/2`

        .. pyvista-plot::

            >>> from pyvisual import Plot3d
            >>> from math import pi
            >>> plotter = Plot3d()
            >>> plotter.add_sun()
            >>> plotter.add_point(1.5, pi/2, pi/2)
            >>> plotter.show()
        """
        if not (1 == r.size == t.size == p.size):
            raise ValueError("Single point coordinate arrays should be size 1")
        return self._add_stack_set(r, t, p, data,
                                   0, dataid,
                                   slice_type='points',
                                   **kwargs)

    @parse_mesh_params
    def add_points(self,
                    r: ArrayLike,
                    t: ArrayLike,
                    p: ArrayLike,
                    data: Optional[ArrayLike] = None,
                    /,
                    axis: int = 0,
                    dataid: str = 'Data',
                    **kwargs):
        """Add a point cloud at spherical coordinates.

        Parameters
        ----------
        r, t, p : ArrayLike
            Coordinate arrays of identical shape.  Each element defines one
            point.
        data : ArrayLike | None, optional
            Scalar values at the points.  Default is ``None`` (solid color).
        axis : int, optional
            Stacking axis forwarded to the point builder.  Default is ``0``.
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered point-cloud actor.

        Examples
        --------
        Add 20 points along the equitorial plane, from :math:`r = 1 R_\\odot` to :math:`r = 30 R_\\odot`,
        colored by radial distance.

        .. pyvista-plot::

            >>> from pyvisual import Plot3d
            >>> import numpy as np
            >>> plotter = Plot3d()
            >>> plotter.add_sun()
            >>> r = np.linspace(1, 30, 20)
            >>> t = np.repeat(np.pi/2, 20)
            >>> p = np.linspace(0, 2*np.pi, 20)
            >>> data = np.arange(20)
            >>> plotter.add_points(r, t, p, data, point_size=5)
            >>> plotter.show()
        """
        return self._add_stack_set(r, t, p, data,
                                   axis, dataid,
                                   slice_type='points',
                                   **kwargs)

    @parse_mesh_params
    def add_spline(self,
                   r: ArrayLike,
                   t: ArrayLike,
                   p: ArrayLike,
                   data: Optional[ArrayLike] = None,
                   /,
                   dataid: str = 'Data',
                   **kwargs):
        """Add a single spline (polyline) through spherical coordinate points.

        Parameters
        ----------
        r, t, p : ArrayLike
            1-D coordinate arrays of identical length defining the spline path.
        data : ArrayLike | None, optional
            Per-point scalar values.  Default is ``None`` (solid color).
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered spline actor.

        Raises
        ------
        ValueError
            If ``r``, ``t``, or ``p`` is not 1-D.

        Examples
        --------
        Add 20 points along the equitorial plane, from :math:`r = 1 R_\\odot` to :math:`r = 30 R_\\odot`,
        colored by radial distance.

        .. pyvista-plot::

            >>> from pyvisual import Plot3d
            >>> import numpy as np
            >>> plotter = Plot3d()
            >>> plotter.add_sun()
            >>> r = np.linspace(1, 30, 20)
            >>> t = np.repeat(np.pi/2, 20)
            >>> p = np.linspace(0, 2*np.pi, 20)
            >>> data = np.arange(20)
            >>> plotter.add_spline(r, t, p, data, line_width=5)
            >>> plotter.show()
        """
        if not (1 == r.ndim == t.ndim == p.ndim):
            raise ValueError("Single spline coordinate arrays should be 1D")
        return self._add_stack_set(r, t, p, data,
                                   0, dataid,
                                   slice_type='splines',
                                   **kwargs)

    @parse_mesh_params
    def add_splines(self,
                    r: ArrayLike,
                    t: ArrayLike,
                    p: ArrayLike,
                    data: Optional[ArrayLike] = None,
                    /,
                    axis: int = 0,
                    dataid: str = 'Data',
                    **kwargs):
        """Add a bundle of splines through spherical coordinate arrays.

        Parameters
        ----------
        r, t, p : ArrayLike
            N-D coordinate arrays of identical shape.  The ``axis`` dimension
            enumerates individual spline paths.
        data : ArrayLike | None, optional
            Scalar values.  Default is ``None`` (solid color).
        axis : int, optional
            The axis that enumerates distinct splines.  Default is ``0``.
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered spline-bundle actor.
        """
        return self._add_stack_set(r, t, p, data,
                                   axis, dataid,
                                   slice_type='splines',
                                   **kwargs)

    @parse_mesh_params
    def add_fieldlines(self,
                    r: ArrayLike,
                    t: ArrayLike,
                    p: ArrayLike,
                    data: Optional[ArrayLike] = None,
                    /,
                    axis: int = 0,
                    dataid: str = 'Data',
                    coloring: Optional[FlColorType] = None,
                    **kwargs):
        """Add a bundle of magnetic fieldlines rendered as splines.

        Wraps :meth:`add_splines` with optional fieldline-specific coloring
        presets.  When ``coloring`` is set, the ``data`` array and default
        render kwargs are overridden automatically.

        Parameters
        ----------
        r, t, p : ArrayLike
            N-D coordinate arrays of identical shape encoding the fieldline
            paths.
        data : ArrayLike | None, optional
            Scalar data for coloring.  Required when ``coloring='polarity'``
            (must be an integer array of the five polarity sentinel values
            from :data:`~pyvisual.core._styling.FL_STATE_ANNOTATIONS`).
            Ignored when ``coloring='random'``.  Default is ``None``.
        axis : int, optional
            The axis that enumerates distinct fieldlines.  Default is ``0``.
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        coloring : {'random', 'polarity'} | None, optional
            Fieldline coloring strategy:

            ``'polarity'``
                Color each fieldline by its open/closed polarity state.
                Uses :data:`~pyvisual.core._styling.FL_POLARITY_COLORING_DEFAULTS`.
            ``'random'``
                Assign a unique random hue to each fieldline.  Uses
                :data:`~pyvisual.core._styling.RANDOM_COLORING_DEFAULTS`.
            ``None``
                Use ``data`` directly (or solid color if ``data`` is ``None``).

            Default is ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.  These are merged on
            top of the coloring preset defaults, so individual keys can be
            overridden.

        Returns
        -------
        out : pyvista.Actor
            The rendered fieldline actor.

        Raises
        ------
        ValueError
            If ``coloring='polarity'`` and ``data`` is ``None`` or does not
            contain between 1 and 5 distinct values.
        """
        match coloring:
            case 'polarity':
                if data is None or not 0 < len(set(data.ravel())) < 6:
                    raise ValueError("Polarity coloring requires")
                data = data.astype(np.int8)
                kwargs = FL_POLARITY_COLORING_DEFAULTS | FIELDLINE_KWARGS | kwargs
            case 'random':
                varray = [s for i, s in enumerate(r.shape) if i != axis]
                data = np.random.randint(0, 256, size=varray, dtype=np.uint8)
                kwargs = RANDOM_COLORING_DEFAULTS | kwargs
        return self._add_stack_set(r, t, p, data,
                                   axis, dataid,
                                   slice_type='splines',
                                   **kwargs)

    def _add_stack_set(self,
                       r: ArrayLike,
                       t: ArrayLike,
                       p: ArrayLike,
                       data: Optional[ArrayLike],
                       /,
                       axis: int,
                       dataid: str,
                       slice_type: Literal['points', 'splines'],
                       **kwargs):
        """Internal dispatcher for stack-mesh rendering.

        Validates that ``r``, ``t``, ``p`` have identical shapes via
        :func:`~pyvisual.core.parsers.parse_stack_mesh`, selects the appropriate
        builder function, assigns scalar data, merges default kwargs, and
        calls :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Parameters
        ----------
        r, t, p : ArrayLike
            Spherical coordinate arrays of identical shape.
        data : ArrayLike | None
            Scalar data array, or ``None`` for solid-color rendering.
        axis : int
            Stacking axis forwarded to the polydata builder.
        dataid : str
            Name of the scalar array.
        slice_type : {'points', 'splines'}
            Which polydata builder to use.
        **kwargs
            Additional keyword arguments merged on top of the styling defaults
            and forwarded to :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered actor.
        """
        r, t, p = parse_stack_mesh(r, t, p)
        match slice_type:
            case 'points':
                builder, opt_kwargs = build_point_polydata, POINTS_KWARGS
            case 'splines':
                builder, opt_kwargs = build_spline_polydata, SPLINES_KWARGS
            case _:
                raise ValueError(f"Invalid slice_type: {slice_type}")
        grid = builder(r, t, p, axis, frame='spherical')
        if data is not None:
            grid[dataid] = parse_data(data, r.shape, axis)
            kwargs = COLORMAP_KWARGS | opt_kwargs | kwargs
        else:
            kwargs = SOLID_COLOR_KWARGS | opt_kwargs | kwargs
        return self.add_mesh(grid, **kwargs)

    def add_surface(self,
                    r: ArrayLike,
                    t: ArrayLike,
                    p: ArrayLike,
                    data: Optional[ArrayLike] = None,
                    /,
                    axis: int = 0,
                    dataid: str = 'Data',
                    method: SurfaceReconstructionType = 'delaunay_2d',
                    surface_kwargs: Optional[dict] = None,
                    **kwargs):
        """Add a reconstructed surface through stacked spherical coordinate arrays.

        Builds a :class:`pyvista.PolyData` surface via
        :func:`~pyvisual.core.mesh3d.build_surface_polydata` and adds it to the
        scene.

        Parameters
        ----------
        r, t, p : ArrayLike
            Spherical coordinate arrays of identical shape encoding the point
            positions on the surface.
        data : ArrayLike | None, optional
            Scalar values at each point.  Default is ``None`` (solid color).
        axis : int, optional
            Stacking axis forwarded to the surface builder.  Default is ``0``.
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        method : SurfaceReconstructionType, optional
            Surface reconstruction method.  Default is ``'delaunay_2d'``.
        surface_kwargs : dict | None, optional
            Extra keyword arguments forwarded to the chosen reconstruction
            method.  Default is ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered surface actor.
        """
        r, t, p = parse_stack_mesh(r, t, p)
        skwargs = surface_kwargs or {}
        grid = build_surface_polydata(r, t, p, axis, method, frame='spherical', **skwargs)
        if data is not None:
            grid[dataid] = parse_data(data, r.shape, axis)
            kwargs = COLORMAP_KWARGS | SLICES_KWARGS | kwargs
        else:
            kwargs = SOLID_COLOR_KWARGS | SLICES_KWARGS | kwargs
        return self.add_mesh(grid, **kwargs)


class GridMeshMixin:
    """Mixin for rendering geometry from independent structured-grid coordinate arrays.

    "Grid" arrays are either:

    - Three 1-D axis arrays of arbitrary (independent) lengths, broadcast to a
      full 3-D meshgrid by :func:`~pyvisual.core.parsers.parse_grid_mesh`.
    - Three pre-broadcast 3-D arrays of identical shape.

    Methods infer the fixed/varying axes from the size of each input array and
    dispatch to the appropriate polydata builder.

    See Also
    --------
    :class:`StackMeshMixin`
        Companion mixin for stacked N-D coordinate arrays.
    """

    @parse_mesh_params
    def add_1d_slice(self,
                     r: ArrayLike,
                     t: ArrayLike,
                     p: ArrayLike,
                     data: Optional[ArrayLike] = None,
                     /,
                     dataid: str = 'Data',
                     **kwargs):
        """Add a 1-D line slice along the single varying spherical coordinate axis.

        Exactly two of ``r``, ``t``, ``p`` must be scalar (size 1), defining a
        fixed coordinate value.  The one array with more than one element defines
        the slice direction; the varying axis is inferred automatically.

        Parameters
        ----------
        r, t, p : ArrayLike
            Spherical coordinate arrays.  Exactly two must be size 1.
        data : ArrayLike | None, optional
            Scalar values along the slice.  Default is ``None`` (solid color).
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered line actor.

        Raises
        ------
        ValueError
            If the number of size-1 arrays is not exactly 2.
        """
        mesh_shape = (r.size, t.size, p.size)
        if sum(scale == 1 for scale in mesh_shape) != 2:
            raise ValueError("1D slices requires exactly two fixed scales.")
        try:
            axis = next(i for i, s in enumerate(mesh_shape) if s > 1)
        except StopIteration:
            msg = f"Could not infer slice axis from input shapes: {mesh_shape}"
            raise ValueError(msg)
        return self._add_grid_set(r, t, p, data,
                                  axis, dataid,
                                  slice_type='splines',
                                  **kwargs)

    @parse_mesh_params
    def add_2d_slice(self,
                     r: ArrayLike,
                     t: ArrayLike,
                     p: ArrayLike,
                     data: Optional[ArrayLike] = None,
                     /,
                     dataid: str = 'Data',
                     **kwargs):
        """Add a 2-D surface slice at a fixed spherical coordinate.

        Exactly one of ``r``, ``t``, ``p`` must be scalar (size 1), fixing that
        coordinate.  The other two axes define the surface grid; the fixed axis is
        inferred automatically.

        Parameters
        ----------
        r, t, p : ArrayLike
            Spherical coordinate arrays.  Exactly one must be size 1.
        data : ArrayLike | None, optional
            Scalar field on the 2-D slice.  Default is ``None`` (solid color).
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered surface actor.

        Raises
        ------
        ValueError
            If the number of size-1 arrays is not exactly 1.
        """
        mesh_shape = (r.size, t.size, p.size)
        if sum(scale == 1 for scale in mesh_shape) != 1:
            raise ValueError("2D slices requires exactly one fixed scale.")
        try:
            axis = next(i for i, s in enumerate(mesh_shape) if s == 1)
        except StopIteration:
            msg = f"Could not infer slice axis from input shapes: {mesh_shape}"
            raise ValueError(msg)
        return self._add_grid_set(r, t, p, data,
                                  axis, dataid,
                                  slice_type='slices',
                                  **kwargs)

    @parse_mesh_params
    def add_3d_slice(self,
                     r: ArrayLike,
                     t: ArrayLike,
                     p: ArrayLike,
                     data: Optional[ArrayLike] = None,
                     /,
                     axis: int = 0,
                     dataid: str = 'Data',
                     slice_type: Literal['points', 'splines', 'slices'] = 'points',
                     **kwargs):
        """Add a 3-D volume slice rendered as points, splines, or surface quads.

        All three coordinate arrays must have more than one element (no fixed
        axis).  The rendering geometry type is controlled by ``slice_type``.

        Parameters
        ----------
        r, t, p : ArrayLike
            Spherical coordinate arrays.  Each must have size > 1.
        data : ArrayLike | None, optional
            Scalar field.  Default is ``None`` (solid color).
        axis : int, optional
            Stacking axis forwarded to the builder function.  Default is ``0``.
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        slice_type : {'points', 'splines', 'slices'}, optional
            Polydata builder to use.  Default is ``'points'``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered 3-D slice actor.
        """
        return self._add_grid_set(r, t, p, data,
                                  axis, dataid, slice_type,
                                  **kwargs)

    def _add_grid_set(self,
                      r: np.ndarray,
                      t: np.ndarray,
                      p: np.ndarray,
                      data: Optional[np.ndarray],
                      /,
                      axis: int,
                      dataid: str,
                      slice_type: Literal['points', 'splines', 'slices'],
                      **kwargs):
        """Internal dispatcher for grid-mesh rendering.

        Validates and broadcasts ``r``, ``t``, ``p`` via
        :func:`~pyvisual.core.parsers.parse_grid_mesh`, selects the builder
        function, assigns scalar data, merges styling defaults, and calls
        :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Parameters
        ----------
        r, t, p : np.ndarray
            Spherical coordinate arrays — 1-D or pre-broadcast 3-D.
        data : np.ndarray | None
            Scalar data, or ``None`` for solid-color rendering.
        axis : int
            Stacking axis forwarded to the builder.
        dataid : str
            Name of the scalar array.
        slice_type : {'points', 'splines', 'slices'}
            Polydata builder to use.
        **kwargs
            Additional keyword arguments merged with styling defaults and
            forwarded to :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered actor.
        """
        r, t, p = parse_grid_mesh(r, t, p)
        match slice_type:
            case 'points':
                builder, opt_kwargs = build_point_polydata, POINTS_KWARGS
            case 'splines':
                builder, opt_kwargs = build_spline_polydata, SPLINES_KWARGS
            case 'slices':
                builder, opt_kwargs = build_slice_polydata, SLICES_KWARGS
            case _:
                raise ValueError(f"Invalid slice_type: {slice_type}")
        grid = builder(r, t, p, axis, frame='spherical')
        if data is not None:
            grid[dataid] = parse_data(data, r.shape, axis)
            kwargs = COLORMAP_KWARGS | opt_kwargs | kwargs
        else:
            kwargs = SOLID_COLOR_KWARGS | opt_kwargs | kwargs
        return self.add_mesh(grid, **kwargs)

    @parse_mesh_params
    def add_contour(self,
                    r: ArrayLike,
                    t: ArrayLike,
                    p: ArrayLike,
                    data: ArrayLike,
                    /,
                    dataid: str = 'Data',
                    isovalue: Optional[ArrayLike] = None,
                    **kwargs):
        """Add an isosurface contour from a 3-D spherical scalar field.

        Builds a :class:`~pyvisual.core.mesh3d.SphericalMesh`, extracts the
        isosurface at ``isovalue`` via :meth:`pyvista.DataSet.contour`, and adds
        the resulting surface mesh to the scene.

        Parameters
        ----------
        r, t, p : ArrayLike
            Spherical coordinate arrays.  May be 1-D axis vectors or 3-D
            meshgrids.
        data : ArrayLike
            Scalar field from which the isosurface is extracted.
        dataid : str, optional
            Name for the scalar array.  Default is ``'Data'``.
        isovalue : ArrayLike | None, optional
            Contour value(s) at which to extract the isosurface.  Defaults to
            ``0`` when ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            The rendered isosurface actor.
        """
        sgrid = SphericalMesh(r, t, p, data=data, dataid=dataid, iformat='rtp')
        mesh = sgrid.contour(np.atleast_1d(isovalue if isovalue is not None else 0))
        return self.add_mesh(mesh, **kwargs | SLICES_KWARGS)



class ObserverMixin:
    """Mixin for camera and observer controls expressed in spherical coordinates.

    Provides properties and methods for:

    - Setting the observer position, focal point, and up vector in
      :math:`(r, \\theta, \\phi)` spherical coordinates.
    - Controlling the observer's position angle (roll about the line of sight).
    - Defining the line-of-sight field of view by angular extents or by minimum
      impact radius.
    - Displaying live camera-state text in the viewport.
    - Fixing the camera up to solar north for interactive exploration.

    All position/orientation properties return named tuples from
    :mod:`pyvisual.core._typing` and accept the same types as setters.
    Setters are decorated with :func:`render_scene` so the viewport refreshes
    immediately after each change.
    """

    def add_fixed_solar_north_observer(self,
                                       p_angle: float = 0) -> None:
        """
        Fix the camera's up vector to solar north (0, 0, 1) and apply a fixed roll (p angle).

        .. note::
           This method is intended to set a fixed camera up vector (with a specified roll)
           for **interactive scene exploration only**. Programmatically setting the observer orientation
           using ``observer_orientation`` or ``observer_los_view`` will override this method's
           behavior and result in undefined behavior.

        See Also
        --------
        :meth:`pyvista.Plotter.enable_terrain_style`
            Interactivity style used by this method to fix the camera up vector.

        """
        self.camera.up = SOLAR_NORTH
        self.camera.roll += p_angle
        self.enable_terrain_style()

    def remove_fixed_solar_north_observer(self):
        """Disable the fixed solar north observer style and return to default camera behavior."""
        self.disable_terrain_style()

    def add_camera_update(self, *, include: Optional[set[str] | str] = None, **kwargs):
        """Add a live camera-state text overlay to the upper left corner of the plot.

        The text is updated on ``'ModifiedEvent'`` triggers from the active camera
        and is primarily intended for debugging and development.  It can display the
        observer position/focus/orientation in spherical coordinates and/or the
        raw PyVista camera position/focal point/roll.

        Parameters
        ----------
        include : set[str] | str | None, optional
            Controls which camera fields are displayed:

            ``None``
                Show all available fields (default).
            ``'spherical'``
                Show only the spherical observer fields (``observer_position``,
                ``observer_focus``, ``observer_orientation``, ``observer_viewup``).
            ``'cartesian'``
                Show only the Cartesian camera fields (``camera_position``,
                ``camera_focal_point``, ``camera_roll``, ``camera_up``).
            A string naming a single field
                Show only that one field.
            A set of field name strings
                Show the intersection with all available fields.

            Default is ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`pyvista.Plotter.add_text` (e.g. ``font_size``, ``color``,
            ``name``).

        Returns
        -------
        out : None
            The text overlay is managed as a VTK observer and updated in-place.

        See Also
        --------
        :meth:`remove_camera_update`
            Remove the observer and its text overlay.
        """
        cupdate_name = kwargs.setdefault('name', "camera_update")
        if cupdate_name is None:
            kwargs['name'] = "camera_update"

        spherical_display = {"observer_position", "observer_focus", "observer_orientation", "observer_viewup"}
        cartesian_display = {"camera_position", "camera_focal_point", "camera_roll", "camera_up"}

        match include:
            case None:
                options = spherical_display | cartesian_display
            case str():
                match include:
                    case "spherical":
                        options = spherical_display
                    case "cartesian":
                        options = cartesian_display
                    case _:
                        options = {include,}
            case Iterable():
                options = (spherical_display | cartesian_display) & set(include)
            case _:
                options = set()

        def _format_tuple(args):
            if hasattr(args, '_fields'):
                return tuple(f"{field}: {arg:.2f}" for arg, field in zip(args, args._fields, strict=True))
            elif isinstance(args, Iterable):
                return tuple(f"{arg:.2f}" for arg in args)
            else:
                return f"{args:.2f}"

        def _update_text(caller, event):
            values = [
                f"{opt}: {_format_tuple(getattr(self, opt))}" if opt.startswith('observer') else
                f"{opt}: {_format_tuple(getattr(self.camera, opt.removeprefix('camera_')))}" for opt in options
            ]
            msg = '\n'.join(values)
            self.add_text(msg, **kwargs)

        self._camera_update_observer = self.camera.AddObserver("ModifiedEvent", _update_text)
        _update_text(self.camera, "ModifiedEvent")

    def remove_camera_update(self):
        """Remove the camera state text observer."""
        self.iren.remove_observer(self._camera_update_observer)

    @property
    def observer_viewup(self):
        """Get or set the camera up vector in spherical coordinates.

        Returns
        -------
        out : SphericalCoordinate
            The camera up vector expressed as :math:`(r, \\theta, \\phi)`.
        """
        r, t, p = cartesian_to_spherical(*self.camera.up)
        return SphericalCoordinate(r, t, p)

    @observer_viewup.setter
    @render_scene
    def observer_viewup(self, args):
        r, t, p = args
        self.camera.up = spherical_to_cartesian(r, t, p)

    @property
    def observer_position(self):
        """
        Get or set the observer's position in spherical coordinates.

        Returns
        -------
        position : SphericalCoordinate
            The observer's position as a named tuple of r, t, p data.
        """
        r, t, p = cartesian_to_spherical(*self.camera.position)
        return SphericalCoordinate(r, t, p)

    @observer_position.setter
    @render_scene
    def observer_position(self, args):
        r, t, p = args
        self.camera.position = spherical_to_cartesian(r, t, p)

    @property
    def observer_focus(self):
        """
        Get or set the observer's focal point in spherical coordinates.

        Returns
        -------
        focus : SphericalCoordinate
            The observer's focal point as a named tuple of r, t, p data.
        """
        r, t, p = cartesian_to_spherical(*self.camera.focal_point)
        return SphericalCoordinate(r, t, p)

    @observer_focus.setter
    @render_scene
    def observer_focus(self, args):
        r, t, p = args
        self.camera.focal_point = spherical_to_cartesian(r, t, p)

    @property
    def observer_orientation(self):
        """
        Get or set the observer's orientation (p angle).

        Returns
        -------
        orientation : ObserverOrientation
            The observer's orientation as a named tuple of r, t, p data.
        """
        p_angle = camera_roll_wrt_solar_north(*self.camera_position)
        return ObserverOrientation(p_angle=p_angle)

    @observer_orientation.setter
    @render_scene
    def observer_orientation(self, arg):
        p_angle = arg
        current_p_angle = camera_roll_wrt_solar_north(*self.camera_position)
        self.camera.roll += (p_angle - current_p_angle)

    @property
    def observer_los_view(self):
        """
        Get or set the observer's line-of-sight field-of-view extents.

        .. warning::
            This property getter is not yet implemented.

        The setter accepts a 4-tuple ``(x0, x1, y0, y1)`` of angular extents
        in degrees defining the horizontal and vertical FOV, and adjusts the
        camera focal point, view angle, and window size accordingly.
        """
        warnings.warn("NOT IMPLEMENTED")
        return

    @observer_los_view.setter
    @render_scene
    def observer_los_view(self, args):
        x0, x1, y0, y1 = args
        elongation = (x0 + x1) / 2
        altitude = (y0 + y1) / 2
        radius, t, p = cartesian_to_spherical(*self.camera.position)
        latitude = 90 - np.rad2deg(t)
        longitude = clip_angle(np.rad2deg(p), max_value=180)
        aspect_ratio = abs(x1 - x0) / abs(y1 - y0)
        _, window_vertical_size = self.window_size
        self.window_size = (int(window_vertical_size * aspect_ratio), int(window_vertical_size))
        self.camera.view_angle = abs(y1 - y0)
        fp = thompson_sphere(
            elongation, altitude, longitude, latitude, radius, self.observer_orientation.p_angle,
        )
        self.camera.focal_point = fp

    @property
    def observer_fov_view(self):
        """
        Get or set the observer's field-of-view by minimum LOS impact radius.

        .. warning::
            This property getter is not yet implemented.

        The setter accepts a single value ``rmin`` (in solar radii). The symmetric
        angular half-extent is computed from the observer distance and ``rmin`` via
        :func:`los_rmin2angle`, then forwarded to :attr:`observer_los_view`.

        Raises
        ------
        ValueError
            If the observer distance is less than ``rmin``.
        """
        warnings.warn("NOT IMPLEMENTED")
        return

    @observer_fov_view.setter
    @render_scene
    def observer_fov_view(self, args):
        rmin = abs(args)
        dobs = self.observer_position.r
        if rmin < dobs:
            angle = abs(los_rmin2angle(rmin, dobs))
            self.observer_los_view = -angle, angle, -angle, angle
        else:
            msg = f"Observer distance {dobs:.2f} is less than FOV {rmin:.2f}."
            raise ValueError(msg)

    def set_observer_pov(self,
                         r,
                         t,
                         p,
                         p_angle,
                         x0, x1,
                         y0, y1):
        """
        Set the observer point-of-view from a spherical position and LOS field-of-view extents.

        Sets the camera position, adjusts the up vector to avoid gimbal lock near
        the poles, resizes the window to match the FOV aspect ratio, and computes
        the focal point via :func:`thompson_sphere`.

        Parameters
        ----------
        r : float
            Observer radial distance in solar radii.
        t : float
            Observer colatitude in radians.
        p : float
            Observer longitude in radians.
        p_angle : float
            Observer position angle (roll about the LOS) in degrees.
        x0, x1 : float
            Horizontal extent of the field of view in degrees (e.g. elongation range).
        y0, y1 : float
            Vertical extent of the field of view in degrees (e.g. altitude range).
        """
        self.camera.position = spherical_to_cartesian(r, t, p)
        # self.camera.up = SOLAR_NORTH

        # set the camera up angle based on latitude
        # - Avoid VTK errors when latitude is too close to the Z-axis for camera up to be along Z
        #   by using the longitude to set x and y components. You still get the same effect
        b_angle = 90 - np.rad2deg(t)
        longitude = np.rad2deg(p)
        if np.abs(b_angle) > 80.:
            nx = -np.sign(b_angle)*np.cos(p)
            ny = -np.sign(b_angle)*np.sin(p)
            self.camera.up = (nx, ny, 0)
        else:
            self.camera.up = SOLAR_NORTH

        # Set the camera up roll (not working?!?).
        # self.camera.roll += p_angle

        elongation = (x0 + x1) / 2
        altitude = (y0 + y1) / 2
        latitude = 90 - np.rad2deg(t)
        longitude = clip_angle(np.rad2deg(p), max_value=180)
        aspect_ratio = abs(x1 - x0) / abs(y1 - y0)
        _, window_vertical_size = self.window_size
        self.window_size = (int(window_vertical_size * aspect_ratio), int(window_vertical_size))
        self.camera.view_angle = abs(y1 - y0)
        self.camera.focal_point = thompson_sphere(
            elongation, altitude, longitude, latitude, r, p_angle
        )
        # self.render()

    @render_scene
    def set_solar_north_up(self):
        """Set the camera up vector to solar north ``(0, 0, 1)`` and re-render the scene."""
        self.camera.up = SOLAR_NORTH


class GeometryMixin:
    """Mixin for adding solar geometry primitives to the 3-D scene.

    Provides convenience methods for:

    - The Sun sphere centred at the origin with radius :math:`1\\,R_\\odot`.
    - Concentric spherical shells defined by inner/outer radii.
    - Planar discs specified in a local spherical basis.
    - The Thomson sphere for a given observer position.
    - Longitude, latitude, and combined lon/lat grid lines on a sphere.
    - A general structured grid of splines via :meth:`add_grid`.
    """

    def add_sun(self, **kwargs):
        """Add a sphere representing the Sun at the origin.

        Constructs a :class:`pyvista.Sphere` with radius :math:`1\\,R_\\odot`
        centred at the origin with a resolution of 180 × 360.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.
            Defaults are ``color='orange'``, ``opacity=1``, ``name='SUN'``; these
            can be overridden by passing the same keys in ``**kwargs``.

        Returns
        -------
        out : pyvista.Actor
            The rendered Sun actor.

        Examples
        --------
        .. pyvista-plot::

            >>> import pyvisual as pv
            >>> pl = pv.Plot3d()
            >>> pl.add_sun()
            >>> pl.show()
        """
        sun = pv.Sphere(radius=1.0,
                        center=(0, 0, 0),
                        theta_resolution=180,
                        phi_resolution=360)
        kw_overrides = dict(color='orange', opacity=1, name='SUN') | kwargs
        return self.add_mesh(sun, **kw_overrides)

    def add_shell(self,
                  r: float = 0.,
                  t: float = 0.,
                  p: float = 0.,
                  inner_radius: float = 0.,
                  outer_radius: float = 1.,
                  **kwargs):
        """
        Add a spherical shell to the plot, defined by inner and outer radii and centered at a given position.

        Parameters
        ----------
        r : float
            Radius of the center of the shell in solar radii.
        t : float
            Co-latitude of the center of the shell in radians.
        p : float
            Longitude of the center of the shell in radians.
        inner_radius : float
            Inner radius of the shell in solar radii.
        outer_radius : float
            Outer radius of the shell in solar radii.
        kwargs : dict
            Additional keyword arguments for styling the shell mesh.

        Returns
        -------
        out : pyvista.Actor
            The mesh actor representing the shell added to the plot.
        """
        center = spherical_to_cartesian(r, t, p)
        shell_inner = pv.Sphere(radius=inner_radius,
                                center=center,
                                theta_resolution=180,
                                phi_resolution=360)
        shell_outer = pv.Sphere(radius=outer_radius,
                        center=center,
                        theta_resolution=180,
                        phi_resolution=360)
        return self.add_mesh(pv.MultiBlock([shell_inner, shell_outer]), **kwargs)

    def add_disc(self,
                 r: float = 0.,
                 t: float = 0.,
                 p: float = 0.,
                 inner_radius: float = 0.,
                 outer_radius: float = 1.,
                 normal: tuple[float, float, float] = (1,0,0),
                 **kwargs):
        """
        Create a planar disc in 3D space using spherical coordinates and add it to the scene.

        The disc center is defined by the spherical position ``(r, t, p)`` and converted to
        Cartesian coordinates via :func:`spherical_to_cartesian`. The disc's surface normal
        is defined in a *local spherical basis* as ``normal=(nr, nt, np)`` and converted to a
        Cartesian vector at the same angular location ``(t, p)`` via
        :func:`spherical_to_cartesian_vec`.

        Internally, this constructs a :class:`pyvista.Disc` with a fixed circumferential
        resolution (``c_res=720``) and forwards all remaining keyword arguments to
        :meth:`add_mesh`.

        Parameters
        ----------
        r, t, p : float, optional
            Spherical coordinates of the disc center, where:
            - ``r`` is the radial distance in solar radii
            - ``t`` is the colatitude in radians
            - ``p`` is the longitude in radians
        inner_radius : float, optional
            Inner radius of the disc (hole radius). Use ``0`` for a solid disc.
        outer_radius : float, optional
            Outer radius of the disc.
        normal : tuple[float, float, float], optional
            Disc normal specified as components in the local spherical basis at ``(t, p)``,
            typically ``(n_r, n_t, n_p)``. This is converted into a 3D Cartesian normal
            vector before building the disc.
        **kwargs
            Additional keyword arguments passed through to :meth:`add_mesh` (e.g. ``color``,
            ``opacity``, ``name``, ``style``, ``line_width``, etc.).

        Returns
        -------
        out : pyvista.Actor
            Returned actor from :meth:`add_mesh`.

        Notes
        -----
        - This method always uses ``c_res=720`` for the disc circumferential resolution.
        """
        center = spherical_to_cartesian(r, t, p)
        normal = spherical_to_cartesian_vec(*normal, t, p)
        disc = pv.Disc(center=center,
                       inner=inner_radius,
                       outer=outer_radius,
                       normal=normal,
                       c_res=360)
        return self.add_mesh(disc, **kwargs)

    def add_thompson_sphere(self,
                            *pos,
                            theta_resolution: int = 180,
                            phi_resolution: int = 360,
                            **kwargs):
        """
        Add a Thompson sphere centered halfway between the origin and the observer.

        This helper constructs a :class:`pyvista.Sphere` intended to represent the
        "Thomson sphere" for a given observer location. The observer location is
        taken from the current camera position unless an explicit spherical position
        is provided.

        The constructed sphere has:

        - **radius** = ``||observer_position|| / 2``
        - **center** = ``observer_position / 2``

        Parameters
        ----------
        *pos : float
            Optional observer position specified in spherical coordinates ``(r, t, p)``.
            If omitted, ``self.camera.position`` is used as the observer position in
            Cartesian coordinates.
        **kwargs
            Additional keyword arguments forwarded to :meth:`add_mesh` (e.g. ``color``,
            ``opacity``, ``style``, ``line_width``, ``name``, etc.).

        Returns
        -------
        out : pyvista.Actor
            Returned actor from :meth:`add_mesh`.

        Notes
        -----
        - The sphere mesh is built with a high angular resolution:
          ``theta_resolution=720`` and ``phi_resolution=1440``.
        - If you pass ``*pos``, it is interpreted as spherical coordinates and converted
          to Cartesian via :func:`spherical_to_cartesian`. If you want to pass a Cartesian
          position explicitly, call :meth:`add_mesh` with a custom :class:`pyvista.Sphere`
          instead.

        Raises
        ------
        ValueError
            If ``*pos`` is provided but does not match the expected signature of
            :func:`spherical_to_cartesian`.

        Examples
        --------
        Use current camera position as the observer:

        >>> actor = plotter.add_thompson_sphere(opacity=0.2)  # doctest: +SKIP

        Provide an observer position in spherical coordinates:

        >>> actor = plotter.add_thompson_sphere(1.0, np.pi/2, 0.0, opacity=0.2)  # doctest: +SKIP
        """
        observer_position = self.camera.position if not pos else spherical_to_cartesian(*pos)
        tsphere = pv.Sphere(radius=np.linalg.norm(observer_position) / 2,
                            center=tuple(pos / 2 for pos in observer_position),
                            theta_resolution=theta_resolution,
                            phi_resolution=phi_resolution)
        return self.add_mesh(tsphere, **kwargs)

    def add_longitudinal_lines(self,
                               lon_deg: int = 30,
                               radius: float = 1.01,
                               **kwargs):
        """
        Add longitudinal (meridian) grid lines to the plot.

        This builds a collection of polylines representing meridians (constant
        longitude / varying colatitude) on a sphere of the given ``radius``. The
        resulting polyline set is added as a single :class:`pyvista.PolyData`.

        Parameters
        ----------
        lon_deg : int, optional
            Spacing between adjacent meridians in **degrees**. For example, ``30``
            produces 12 meridians around the sphere.
        radius : float, optional
            Sphere radius on which the lines are drawn. A value slightly larger than
            a rendered surface (e.g. ``1.01``) can help prevent z-fighting.
        **kwargs
            Additional keyword arguments forwarded to :meth:`add_mesh` to control
            styling (e.g. ``color``, ``opacity``, ``line_width``, etc.). This method
            also merges in default styling via ``SOLID_COLOR_KWARGS | SPLINE_KWARGS``.

        Returns
        -------
        out : pyvista.Actor
            Returned actor from :meth:`add_mesh`.

        Notes
        -----
        - Meridians are sampled with 180 points in colatitude (theta).
        - Longitudes span ``[0, 2π)`` with ``int(360/lon_deg)`` meridians.

        Examples
        --------
        Add longitudinal lines every 15° at a radius of 1.02 R☉:

        .. pyvista-plot::

            >>> from pyvisual import Plot3d
            >>> plotter = Plot3d()
            >>> plotter.add_sun()
            >>> actor = plotter.add_longitudinal_lines(lon_deg=15, radius=1.02, line_width=2)  # doctest: +SKIP
            >>> plotter.show()
        """
        rargs = (radius, radius, 1, 1)
        targs = (0, np.pi, 0, 360)
        pargs = (0, 2 * np.pi, int(360 / lon_deg), 360)
        return self.add_grid(rargs, targs, pargs, **kwargs)

    def add_latitudinal_lines(self,
                              lat_deg: int = 15,
                              radius: float = 1.01,
                              **kwargs):
        """Add latitudinal (parallel) grid lines to the plot.

        Builds a collection of polylines representing parallels (constant
        colatitude / varying longitude) on a sphere of the given ``radius``.

        Parameters
        ----------
        lat_deg : int, optional
            Spacing between adjacent parallels in **degrees**.  For example,
            ``15`` produces 12 parallels from pole to pole.
        radius : float, optional
            Sphere radius on which the lines are drawn.  Default is ``1.01``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            Returned actor from :meth:`add_grid`.
        """
        rargs = (radius, radius, 1, 1)
        targs = (0, np.pi, int(180 / lat_deg), 360)
        pargs = (0, 2 * np.pi, 0, 180)
        return self.add_grid(rargs, targs, pargs, **kwargs)

    def add_longlat_lines(self,
                          lat_deg: int = 15,
                          lon_deg: int = 30,
                          radius: float = 1.01,
                          **kwargs):
        """Add combined longitude and latitude grid lines to the plot.

        Convenience method that draws both meridians (constant longitude) and
        parallels (constant colatitude) simultaneously on a sphere of the given
        ``radius``.

        Parameters
        ----------
        lat_deg : int, optional
            Spacing between adjacent parallels in degrees.  Default is ``15``.
        lon_deg : int, optional
            Spacing between adjacent meridians in degrees.  Default is ``30``.
        radius : float, optional
            Sphere radius on which the lines are drawn.  Default is ``1.01``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            Returned actor from :meth:`add_grid`.

        See Also
        --------
        :meth:`add_longitudinal_lines`
            Meridians only.
        :meth:`add_latitudinal_lines`
            Parallels only.
        """
        rargs = (radius, radius, 1, 1)
        targs = (0, np.pi, int(180 / lat_deg), 180)
        pargs = (0, 2 * np.pi, int(360 / lon_deg), 360)
        return self.add_grid(rargs, targs, pargs, **kwargs)

    def add_grid(self,
                 rargs: tuple[float, float, int, int],
                 targs: tuple[float, float, int, int],
                 pargs: tuple[float, float, int, int],
                 **kwargs):
        """Add a general structured spline grid defined by spherical axis parameters.

        Each axis is specified as a 4-tuple ``(min, max, num_splines, resolution)``
        where:

        - ``min``, ``max`` define the axis range.
        - ``num_splines`` is the number of grid lines **perpendicular** to this
          axis (i.e. along the other two axes).
        - ``resolution`` is the number of sample points along this axis for the
          splines that run **along** it.

        The method constructs three sets of splines (one per axis) and adds them
        as a :class:`pyvista.MultiBlock`.

        Parameters
        ----------
        rargs : tuple[float, float, int, int]
            ``(r_min, r_max, r_num, r_res)`` — radial axis parameters.
        targs : tuple[float, float, int, int]
            ``(t_min, t_max, t_num, t_res)`` — colatitude axis parameters.
        pargs : tuple[float, float, int, int]
            ``(p_min, p_max, p_num, p_res)`` — longitude axis parameters.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.

        Returns
        -------
        out : pyvista.Actor
            Returned actor from :meth:`~pyvisual.core.plot3d.Plot3d.add_mesh`.
        """
        rmin, rmax, rnum, rres = rargs
        tmin, tmax, tnum, tres = targs
        pmin, pmax, pnum, pres = pargs
        # Generate spherical coordinates for the latitudinal lines

        scales = ij_meshgrid(np.linspace(rmin, rmax, rnum),
                              np.linspace(tmin, tmax, tnum),
                              np.linspace(pmin, pmax, pres))
        pgrid = build_spline_polydata(*scales, axis=2, frame='spherical')

        scales = ij_meshgrid(np.linspace(rmin, rmax, rnum),
                              np.linspace(tmin, tmax, tres),
                              np.linspace(pmin, pmax, pnum))
        tgrid = build_spline_polydata(*scales, axis=1, frame='spherical')

        scales = ij_meshgrid(np.linspace(rmin, rmax, rres),
                              np.linspace(tmin, tmax, tnum),
                              np.linspace(pmin, pmax, pnum))
        rgrid = build_spline_polydata(*scales, axis=0, frame='spherical')
        kwargs = SOLID_COLOR_KWARGS | SPLINES_KWARGS | kwargs

        return self.add_mesh(pv.MultiBlock([rgrid, tgrid, pgrid]), **kwargs)