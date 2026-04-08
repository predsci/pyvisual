"""
Mesh classes and polydata builder functions for structured solar-physics grids.

This module provides the two primary mesh classes used throughout **pyvisual** —
:class:`SphericalMesh` and :class:`CartesianMesh` — together with the abstract
base classes and standalone polydata builder functions they depend on.

Class hierarchy
---------------
.. code-block:: text

    _BaseFrameMesh (ABC)
    ├── CartesianMesh(pv.StructuredGrid, CartesianMeshFilters)
    └── SphericalMesh(pv.RectilinearGrid, SphericalMeshFilters)

    _BaseFrameFilters (ABC)
    ├── CartesianMeshFilters
    └── SphericalMeshFilters

Polydata builders
-----------------
:func:`build_point_polydata`
    Unconnected point cloud from coordinate arrays.
:func:`build_spline_polydata`
    Line-connected splines from stacked coordinate arrays.
:func:`build_slice_polydata`
    Quad-faced surface patch from 2-D coordinate arrays.
:func:`build_surface_polydata`
    Reconstructed surface from scattered points.
:func:`build_thompson_sphere`
    Sphere centered halfway between the origin and an observer position.

Operator support
----------------
Both mesh classes inherit the full arithmetic suite from :class:`_BaseFrameMesh`
(``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``, unary ``neg``/``pos``/``abs``)
and :meth:`~_BaseFrameMesh.__array_ufunc__`, so that NumPy ufuncs (e.g.
``np.sqrt(mesh)``) operate element-wise on the active scalar field and return a
new mesh instance of the same type.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import singledispatchmethod, cached_property, reduce, wraps
from itertools import pairwise
from numbers import Real, Number
from pathlib import Path
from typing import Optional, ClassVar, Callable

import numpy as np
import pyvista as pv
from numpy._typing import ArrayLike
from psi_io import read_hdf_by_index
from scipy.interpolate import RegularGridInterpolator

from pyvisual.core.constants import FRAME_SCALES, FRAME_ALIASES
from pyvisual.core.parsers import parse_grid_mesh, parse_data, generate_transforms, fetch_canonical_frame, _normalize_frame, apply_mesh_transform
from pyvisual.core._typing import SurfaceReconstructionType, MeshFramesType
from pyvisual.utils.geometry import moveaxis_to_start, ij_meshgrid, spherical_to_cartesian, cartesian_to_spherical
from pyvisual.utils.helpers import atleast_1dnull


def _update_mesh_frame(func: Callable):
    """Decorator that stamps the canonical ``frame`` name into the result's ``user_dict``.

    After the wrapped function returns a :class:`pyvista.DataSet`, writes
    ``result.user_dict['MESH_FRAME'] = fetch_canonical_frame(frame)`` when
    ``frame`` is not ``None``.  Applied to the polydata builder functions so that
    :func:`~pyvisual.core.parsers.apply_mesh_transform` can later resolve the
    correct coordinate conversion.

    Parameters
    ----------
    func : Callable
        Builder function whose return value is a :class:`pyvista.DataSet`.  Must
        accept a ``frame`` keyword argument.

    Returns
    -------
    out : Callable
        The wrapped function with the same signature as ``func``.
    """
    @wraps(func)
    def wrapper(self, *args, frame: Optional[MeshFramesType] = None, **kwargs):
        result = func(self, *args, frame=frame, **kwargs)
        if frame is not None:
            result.user_dict.update(MESH_FRAME=fetch_canonical_frame(frame))
        return result
    return wrapper


def _update_user_dict(func: Callable):
    """Decorator that propagates the mesh's ``user_dict`` to the result of a filter method.

    After the wrapped method returns a new mesh object, copies all entries from
    ``self.user_dict`` into ``result.user_dict`` so that metadata (e.g.
    ``'MESH_FRAME'``) is preserved across filter operations.

    Parameters
    ----------
    func : Callable
        A filter method of :class:`CartesianMeshFilters` or
        :class:`SphericalMeshFilters` that returns a new mesh object.

    Returns
    -------
    out : Callable
        The wrapped method with the same signature as ``func``.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        result.user_dict.update(self.user_dict)
        return result
    return wrapper


@_update_mesh_frame
def build_point_polydata(d1: np.ndarray,
                         d2: np.ndarray,
                         d3: np.ndarray,
                         axis: int,
                         frame: Optional[MeshFramesType] = None):
    """Build an unconnected :class:`pyvista.PolyData` point cloud from coordinate arrays.

    Each element of ``d1``, ``d2``, ``d3`` along ``axis`` defines one point.
    The arrays are moved so that ``axis`` is the leading dimension, then all
    remaining dimensions are flattened to produce a ``(N, 3)`` points matrix.

    Parameters
    ----------
    d1, d2, d3 : np.ndarray
        Coordinate arrays for the three spatial dimensions (e.g. ``r``, ``t``,
        ``p`` or ``x``, ``y``, ``z``).  All arrays must have the same shape.
    axis : int
        The axis along which distinct "splines" or "sets" are stacked.  This axis
        is moved to position 0 before flattening.
    frame : MeshFramesType | None, optional
        Coordinate frame label written to ``result.user_dict['MESH_FRAME']`` via
        the :func:`_update_mesh_frame` decorator.  Default is ``None``.

    Returns
    -------
    out : pyvista.PolyData
        A point-cloud polydata with no connectivity (no lines or faces).
    """
    d1, d2, d3 = (moveaxis_to_start(d, axis) for d in (d1, d2, d3))
    N = d1.shape[0]
    d1, d2, d3 = (d.reshape(N, -1) for d in (d1, d2, d3))
    points = np.column_stack([d1.ravel(), d2.ravel(), d3.ravel()])
    return pv.PolyData(points)


@_update_mesh_frame
def build_spline_polydata(d1: np.ndarray,
                          d2: np.ndarray,
                          d3: np.ndarray,
                          axis: int,
                          frame: Optional[MeshFramesType] = None):
    """Build a line-connected :class:`pyvista.PolyData` of splines from coordinate arrays.

    After moving ``axis`` to the leading dimension, the arrays are reshaped to
    ``(N, S)`` where ``N`` is the number of points per spline and ``S`` is the
    number of splines.  Each column of the reshaped arrays becomes one polyline
    with ``N`` vertices.

    Parameters
    ----------
    d1, d2, d3 : np.ndarray
        Coordinate arrays for the three spatial dimensions.  All arrays must have
        the same shape.
    axis : int
        The axis that enumerates distinct splines (moved to the leading dimension).
    frame : MeshFramesType | None, optional
        Coordinate frame label written to ``result.user_dict['MESH_FRAME']``.
        Default is ``None``.

    Returns
    -------
    out : pyvista.PolyData
        A polydata with ``S`` polylines, each containing ``N`` points.
    """
    d1, d2, d3 = (moveaxis_to_start(d, axis) for d in (d1, d2, d3))
    N = d1.shape[0]
    d1, d2, d3 = (d.reshape(N, -1) for d in (d1, d2, d3))
    N, S = d1.shape

    points = np.column_stack([d1.ravel(), d2.ravel(), d3.ravel()])
    idx = np.arange(S * N).reshape(N, S).T
    lines = np.hstack([np.full((S, 1), N), idx]).ravel()

    return pv.PolyData(points, lines=lines)


@_update_mesh_frame
def build_slice_polydata(d1: np.ndarray,
                         d2: np.ndarray,
                         d3: np.ndarray,
                         axis: int,
                         frame: Optional[MeshFramesType] = None):
    """Build a quad-faced :class:`pyvista.PolyData` surface patch from coordinate arrays.

    The coordinate arrays are moved so that ``axis`` is the leading dimension,
    then reshaped to ``(S, M, N)`` where ``S`` is the number of independent surface
    patches, and ``M × N`` is the grid resolution of each patch.  Quads are
    constructed for all ``(M-1) × (N-1)`` interior cells of each patch.

    Parameters
    ----------
    d1, d2, d3 : np.ndarray
        Coordinate arrays for the three spatial dimensions.  All arrays must have
        the same shape.
    axis : int
        The "batch" axis that enumerates independent surface patches (moved to
        the leading dimension).
    frame : MeshFramesType | None, optional
        Coordinate frame label written to ``result.user_dict['MESH_FRAME']``.
        Default is ``None``.

    Returns
    -------
    out : pyvista.PolyData
        A quad-faced polydata surface.
    """
    d1, d2, d3 = (moveaxis_to_start(d, axis) for d in (d1, d2, d3))
    *batch, M, N = d1.shape
    S = int(np.prod(batch)) if batch else 1
    d1, d2, d3 = (d.reshape(S, M, N) for d in (d1, d2, d3))

    points = np.column_stack([d1.ravel(), d2.ravel(), d3.ravel()])

    nquads = (M - 1) * (N - 1)
    fi, fj = np.mgrid[0:M-1, 0:N-1]
    fi, fj = fi.ravel(), fj.ravel()
    q0 = fi * N + fj
    q1 = fi * N + fj + 1
    q2 = (fi + 1) * N + fj + 1
    q3 = (fi + 1) * N + fj

    offsets = (np.arange(S) * M * N)[:, None]  # (S, 1)
    fours = np.full((S, nquads), 4)
    faces = np.column_stack([
        fours.ravel(),
        (q0 + offsets).ravel(),
        (q1 + offsets).ravel(),
        (q2 + offsets).ravel(),
        (q3 + offsets).ravel(),
    ]).ravel()

    return pv.PolyData(points, faces=faces)


def build_surface_polydata(d1: np.ndarray,
                           d2: np.ndarray,
                           d3: np.ndarray,
                           axis: int,
                           method: SurfaceReconstructionType = 'reconstruct_surface',
                           frame: Optional[MeshFramesType] = None,
                           **kwargs):
    """Build a surface mesh from scattered coordinate arrays using a reconstruction method.

    First assembles a point cloud via :func:`build_point_polydata`, then applies
    one of three surface-reconstruction strategies controlled by ``method``.

    Parameters
    ----------
    d1, d2, d3 : np.ndarray
        Coordinate arrays for the three spatial dimensions.  All arrays must have
        the same shape.
    axis : int
        Axis along which distinct point sets are stacked.
    method : SurfaceReconstructionType, optional
        Surface reconstruction strategy:

        ``'delaunay_2d'``
            Project onto a plane and triangulate via
            :meth:`pyvista.PolyData.delaunay_2d`.
        ``'delaunay_3d'``
            Full volumetric tetrahedralisation, then extract the outer surface.
        ``'reconstruct_surface'``
            Implicit surface reconstruction via
            :meth:`pyvista.PolyData.reconstruct_surface`.

        Default is ``'reconstruct_surface'``.
    frame : MeshFramesType | None, optional
        Coordinate frame label written to the point-cloud ``user_dict['MESH_FRAME']``
        via :func:`build_point_polydata`.  Default is ``None``.
    **kwargs
        Additional keyword arguments forwarded to the chosen reconstruction method.

    Returns
    -------
    out : pyvista.PolyData
        The reconstructed surface polydata.
    """
    pdata = build_point_polydata(d1, d2, d3, axis, frame=frame)
    match method:
        case 'delaunay_2d':
            return pdata.delaunay_2d(**kwargs)
        case 'delaunay_3d':
            return pdata.delaunay_3d(**kwargs).extract_surface(algorithm='geometry')
        case 'reconstruct_surface':
            return pdata.reconstruct_surface(**kwargs)
        case _:
            return pdata


# @_update_mesh_frame
def build_thompson_sphere(d1: float,
                          d2: float,
                          d3: float,
                          theta_resolution: int = 180,
                          phi_resolution: int = 360,
                          frame: Optional[MeshFramesType] = None):
    """Build a Thomson sphere centered halfway between the origin and an observer position.

    The Thomson sphere for a given observer has:

    .. math::

       \\text{radius} = \\frac{\\|\\mathbf{r}_{\\text{obs}}\\|}{2}, \\quad
       \\text{center} = \\frac{\\mathbf{r}_{\\text{obs}}}{2}

    where :math:`\\mathbf{r}_{\\text{obs}}` is the Cartesian observer position.
    If ``frame`` is not ``'cartesian'``, the input coordinates are converted
    before computing the sphere parameters.

    Parameters
    ----------
    d1, d2, d3 : float
        Observer position coordinates in the frame given by ``frame``.
    theta_resolution : int, optional
        Number of theta-angle divisions on the sphere.  Default is ``180``.
    phi_resolution : int, optional
        Number of phi-angle divisions on the sphere.  Default is ``360``.
    frame : MeshFramesType | None, optional
        Coordinate frame of ``(d1, d2, d3)``.  If not ``'cartesian'``, the
        appropriate conversion function is applied.  Default is ``None``.

    Returns
    -------
    out : pyvista.PolyData
        A spherical polydata object with ``MESH_FRAME='cartesian'`` stamped
        in its ``user_dict``.
    """
    observer_position = np.array((d1, d2, d3))
    if transforms := generate_transforms(frame, 'cartesian'):
        observer_position = reduce(lambda v, f: f(*v), transforms, observer_position)
    tsphere = pv.Sphere(radius=np.linalg.norm(observer_position) / 2,
                        center=tuple(pos / 2 for pos in observer_position),
                        theta_resolution=theta_resolution,
                        phi_resolution=phi_resolution)
    tsphere.user_dict.update(MESH_FRAME='cartesian')
    return tsphere


class _BaseFrameFilters(ABC):
    """Abstract base class defining the filter interface for frame-aware mesh classes.

    Both :class:`CartesianMeshFilters` and :class:`SphericalMeshFilters` implement
    this interface.  Methods that operate on mesh coordinates are frame-specific (e.g.
    ``logspace`` modifies the radial axis differently for spherical vs Cartesian meshes),
    while the public API is identical across frames.
    """

    @abstractmethod
    def interpolate_mesh(self, mesh):
        """Interpolate ``mesh`` data onto the points of this mesh.

        Parameters
        ----------
        mesh : pyvista.DataSet
            Source mesh whose active scalars are sampled.

        Returns
        -------
        out : pyvista.DataSet
            A copy of this mesh with the interpolated scalar values assigned.
        """
        ...

    @abstractmethod
    def radially_scale(self, *args, exp: Optional[Number] = None):
        """Multiply the active scalars by a power of the radial coordinate.

        When ``args`` are provided they are interpreted as ``(xp, fp)`` lookup
        tables passed to :func:`numpy.interp` to define a spatially-varying scale
        factor :math:`s(r)`.  The data are then multiplied by
        :math:`(s(r) \\cdot r)^{\\exp}` where :math:`\\exp = e` when omitted.

        Parameters
        ----------
        *args : ArrayLike
            Optional ``(xp, fp)`` arguments for :func:`numpy.interp` to define
            a spatially-varying radial scale factor.  If omitted, ``s(r) = 1``.
        exp : Number | None, optional
            Exponent applied to the scaled radius.  Default is Euler's number
            :math:`e`.

        Returns
        -------
        out : CartesianMesh | SphericalMesh
            A copy of the mesh with scaled scalar data.
        """
        ...

    @abstractmethod
    def radially_unscale(self, *args, exp: Optional[Number] = None):
        """Divide the active scalars by a power of the radial coordinate.

        Inverse operation of :meth:`radially_scale`.

        Parameters
        ----------
        *args : ArrayLike
            Optional ``(xp, fp)`` arguments for :func:`numpy.interp`.
        exp : Number | None, optional
            Exponent applied to the scaled radius.  Default is :math:`e`.

        Returns
        -------
        out : CartesianMesh | SphericalMesh
            A copy of the mesh with de-scaled scalar data.
        """
        ...

    @abstractmethod
    def logspace(self, base: Optional[Number] = None, offset: float = 1):
        """Remap mesh point positions from linear to logarithmic radial spacing.

        Replaces each radial coordinate :math:`r` with
        :math:`\\log_b(r) + \\text{offset}` (natural log when ``base`` is ``None``).
        Useful for displaying data that spans several decades in radius.

        Parameters
        ----------
        base : Number | None, optional
            Logarithm base.  If ``None``, the natural logarithm is used.
            Default is ``None``.
        offset : float, optional
            Additive offset applied after the logarithm.  Default is ``1``.

        Returns
        -------
        out : CartesianMesh | SphericalMesh
            A copy of the mesh with log-spaced radial coordinates.
        """
        ...

    @abstractmethod
    def expspace(self, base: Optional[Number] = None, offset: float = 1):
        """Remap mesh point positions from linear to exponential radial spacing.

        Inverse of :meth:`logspace`.  Replaces each radial coordinate :math:`r`
        with :math:`b^{r - \\text{offset}}` (natural exponent when ``base`` is
        ``None``).

        Parameters
        ----------
        base : Number | None, optional
            Exponentiation base.  If ``None``, :math:`e^{r - \\text{offset}}` is
            used.  Default is ``None``.
        offset : float, optional
            Subtractive offset applied before exponentiation.  Default is ``1``.

        Returns
        -------
        out : CartesianMesh | SphericalMesh
            A copy of the mesh with exponentially-spaced radial coordinates.
        """
        ...

    @abstractmethod
    def deconstruct(self, axis: int = 0, method: str = 'splines'):
        """Convert the structured mesh into a polydata representation.

        Extracts the mesh coordinate arrays and active scalar data and rebuilds
        them as a :class:`pyvista.PolyData` using one of the builder functions
        (points, splines, or slice quads) selected by ``method``.

        Parameters
        ----------
        axis : int, optional
            The stacking axis forwarded to the builder function.  Default is ``0``.
        method : {'points', 'splines', 'slices'}, optional
            Builder strategy.  Default is ``'splines'``.

        Returns
        -------
        out : pyvista.PolyData
            The deconstructed polydata with the active scalar data assigned.

        Raises
        ------
        ValueError
            If ``method`` is not one of ``'points'``, ``'splines'``, or
            ``'slices'``.
        """
        ...


class CartesianMeshFilters(_BaseFrameFilters):
    """Filter operations for :class:`CartesianMesh`.

    Implements the :class:`_BaseFrameFilters` interface for meshes whose points
    are expressed in Cartesian :math:`(x, y, z)` coordinates.  Radial operations
    compute the radius as the Euclidean norm of the point coordinates.
    """

    @_update_user_dict
    def interpolate_mesh(self, mesh):
        return mesh.sample(self)

    @_update_user_dict
    def radially_scale(self, *args, exp: Optional[Number] = None):
        mesh = self.copy()
        radius = np.linalg.norm(mesh.points, axis=1).reshape(mesh.dimensions, order='F')
        r = np.interp(radius, *args) * radius if args else radius
        mesh.data *= r ** np.exp(1) if exp is None else r ** exp
        return mesh

    @_update_user_dict
    def radially_unscale(self, *args, exp: Optional[Number] = None):
        mesh = self.copy()
        radius = np.linalg.norm(mesh.points, axis=1).reshape(mesh.dimensions, order='F')
        r = np.interp(radius, *args) * radius if args else radius
        mesh.data /= r ** np.exp(1) if exp is None else r ** exp
        return mesh

    @_update_user_dict
    def logspace(self, base: Optional[Number] = None, offset: float = 1):
        mesh = self.copy()
        r = np.linalg.norm(mesh.points, axis=1, keepdims=True)
        r_new = np.log(r) + offset if base is None else np.log(r) / np.log(base) + offset
        mesh.points = mesh.points * (r_new / r)
        return mesh

    @_update_user_dict
    def expspace(self, base: Optional[Number] = None, offset: float = 1):
        mesh = self.copy()
        r = np.linalg.norm(mesh.points, axis=1, keepdims=True)
        r_new = np.exp(r - offset) if base is None else base ** (r - offset)
        mesh.points = mesh.points * (r_new / r)
        return mesh

    @_update_user_dict
    def deconstruct(self, axis: int = 0, method: str = 'splines'):
        x, y, z = self.scales
        match method:
            case 'points':
                builder = build_point_polydata
            case 'splines':
                builder = build_spline_polydata
            case 'slices':
                builder = build_slice_polydata
            case _:
                msg = f"Unsupported deconstruction method: {method}"
                raise ValueError(msg)
        dec_mesh = builder(x, y, z, axis, frame='cartesian')
        dec_mesh[self.active_scalars_name] = parse_data(self.data, x.shape, axis)
        return dec_mesh

    def cartesian_to_spherical(self):
        """Convert this mesh's point coordinates from Cartesian to spherical.

        Returns
        -------
        out : pyvista.StructuredGrid
            A new :class:`pyvista.StructuredGrid` with points expressed in
            :math:`(r, \\theta, \\phi)` coordinates and
            ``user_dict['MESH_FRAME'] = 'spherical'``.
        """
        mesh = pv.StructuredGrid(self)
        mesh.points = np.column_stack(cartesian_to_spherical(*mesh.points.T))
        mesh.user_dict.update(MESH_FRAME='spherical')
        return mesh


class SphericalMeshFilters(_BaseFrameFilters):
    """Filter operations for :class:`SphericalMesh`.

    Implements the :class:`_BaseFrameFilters` interface for meshes stored as
    :class:`pyvista.RectilinearGrid` with axes ``(r, θ, φ)`` in the PSI spherical
    convention.  Radial operations access the radial axis directly via
    :attr:`SphericalMesh.r` rather than computing it from Cartesian norms.
    """

    @_update_user_dict
    def interpolate_mesh(self, mesh):
        return mesh.sample(self)

    @_update_user_dict
    def radially_scale(self, *args, exp: Optional[Number] = None):
        mesh = self.copy()
        r = np.interp(mesh.r, *args) * mesh.r if args else mesh.r
        r = np.expand_dims(r, (1, 2))
        mesh.data *= r ** np.exp(1) if exp is None else r ** exp
        return mesh

    @_update_user_dict
    def radially_unscale(self, *args, exp: Optional[Number] = None):
        mesh = self.copy()
        r = np.interp(mesh.r, *args) * mesh.r if args else mesh.r
        r = np.expand_dims(r, (1, 2))
        mesh.data /= r ** np.exp(1) if exp is None else r ** exp
        return mesh

    @_update_user_dict
    def logspace(self, base: Optional[Number] = None, offset: float = 1):
        mesh = self.copy()
        if base is None:
            mesh.r = np.log(mesh.r) + offset
        else:
            mesh.r = np.log(mesh.r) / np.log(base) + offset
        return mesh

    @_update_user_dict
    def expspace(self, base: Optional[Number] = None, offset: float = 1):
        mesh = self.copy()
        if base is None:
            mesh.r = np.exp(mesh.r - offset)
        else:
            mesh.r = base ** (mesh.r - offset)
        return mesh

    @_update_user_dict
    def deconstruct(self, axis: int = 0, method: str = 'splines'):
        r, t, p = parse_grid_mesh(*self.scales)
        match method:
            case 'points':
                builder = build_point_polydata
            case 'splines':
                builder = build_spline_polydata
            case 'slices':
                builder = build_slice_polydata
            case _:
                msg = f"Unsupported deconstruction method: {method}"
                raise ValueError(msg)
        dec_mesh = builder(r, t, p, axis, frame='spherical')
        dec_mesh[self.active_scalars_name] = parse_data(self.data, r.shape, axis)
        return dec_mesh

    def spherical_to_cartesian(self):
        """Convert this mesh's point coordinates from spherical to Cartesian.

        Casts the :class:`pyvista.RectilinearGrid` to a
        :class:`pyvista.StructuredGrid` and replaces the point coordinates with
        their Cartesian equivalents via
        :func:`~pyvisual.utils.geometry.spherical_to_cartesian`.

        Returns
        -------
        out : pyvista.StructuredGrid
            A new structured grid with points in :math:`(x, y, z)` coordinates
            and ``user_dict['MESH_FRAME'] = 'cartesian'``.
        """
        mesh = self.cast_to_structured_grid()
        mesh.points = np.column_stack(spherical_to_cartesian(*mesh.points.T))
        mesh.user_dict.update(MESH_FRAME='cartesian')
        return mesh


class _BaseFrameMesh(ABC):
    """Abstract base class shared by :class:`SphericalMesh` and :class:`CartesianMesh`.

    Provides:

    - A ``@singledispatchmethod`` dispatch mechanism (``_dispatch_input``) that
      accepts :class:`str` / :class:`pathlib.Path` (HDF file), raw arrays or
      scalars, or an existing :class:`pyvista.DataSet`.
    - The full NumPy arithmetic operator suite (``+``, ``-``, ``*``, ``/``,
      ``//``, ``%``, ``**``, unary ``neg``/``pos``/``abs``) operating element-wise
      on the active scalar field.
    - :meth:`__array_ufunc__` so that NumPy ufuncs (e.g. ``np.sqrt(mesh)``) work
      transparently.
    - Convenient :attr:`data` and :attr:`scales` properties.

    Subclasses must define :attr:`MESH_FRAME`, :meth:`_parse_iformat`,
    :meth:`_set_arrays`, and :meth:`_dispatch_pyvista`.

    Attributes
    ----------
    MESH_FRAME : str
        Class-level canonical frame name (``'spherical'`` or ``'cartesian'``).
    """

    MESH_FRAME: ClassVar[str]

    def __init__(self,
                 *args,
                 data: Optional[ArrayLike] = None,
                 dataid: Optional[str] = None,
                 iformat: Optional[str] = None,
                 **kwargs):
        """Initialise the mesh from a file path, raw coordinate arrays, or an existing dataset.

        Dispatches the first positional argument to :meth:`_dispatch_input` and
        then stamps :attr:`MESH_FRAME` into :attr:`pyvista.DataSet.user_dict`.

        Parameters
        ----------
        *args
            First positional argument is the primary input; it may be:

            - A :class:`str` or :class:`pathlib.Path` — an HDF4/HDF5 file read
              via :func:`psi_io.read_hdf_by_index`.  Subsequent positional
              arguments are forwarded as index arguments to the file reader.
            - An :class:`~collections.abc.Iterable` or :class:`~numbers.Number` —
              interpreted as the first coordinate axis (``r`` or ``x``);
              additional axes follow as positional args.
            - A :class:`pyvista.DataSet` — shallow- or deep-copied into this
              mesh.

        data : ArrayLike | None, optional
            Scalar data array.  When reading from a file, overrides the data
            loaded from the file.  Default is ``None``.
        dataid : str | None, optional
            Name used when assigning ``data`` as a PyVista point array.
            Default is ``None``.
        iformat : str | None, optional
            Format string describing the order of coordinate axes in the input
            (e.g. ``'rtp'``, ``'ptr'``, ``'xyz'``).  Defaults to the canonical
            frame order when ``None``.
        **kwargs
            Additional keyword arguments forwarded to the internal parse/copy
            methods.
        """
        super().__init__()
        if args:
            self._dispatch_input(*args,
                                 data=data,
                                 dataid=dataid,
                                 iformat=iformat,
                                 **kwargs)
        self.user_dict.update(MESH_FRAME=self.MESH_FRAME)

    def __add__(self, other):
        """Return a copy of this mesh with ``self.data + other`` as the active scalar."""
        return self._binary_op(other, np.add)

    def __radd__(self, other):
        """Return a copy of this mesh with ``other + self.data`` as the active scalar."""
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        """Return a copy of this mesh with ``self.data - other`` as the active scalar."""
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other):
        """Return a copy of this mesh with ``other - self.data`` as the active scalar."""
        return self._binary_op(other, np.subtract, swap=True)

    def __mul__(self, other):
        """Return a copy of this mesh with ``self.data * other`` as the active scalar."""
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other):
        """Return a copy of this mesh with ``other * self.data`` as the active scalar."""
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        """Return a copy of this mesh with ``self.data / other`` as the active scalar."""
        return self._binary_op(other, np.true_divide)

    def __rtruediv__(self, other):
        """Return a copy of this mesh with ``other / self.data`` as the active scalar."""
        return self._binary_op(other, np.true_divide, swap=True)

    def __floordiv__(self, other):
        """Return a copy of this mesh with ``self.data // other`` as the active scalar."""
        return self._binary_op(other, np.floor_divide)

    def __rfloordiv__(self, other):
        """Return a copy of this mesh with ``other // self.data`` as the active scalar."""
        return self._binary_op(other, np.floor_divide, swap=True)

    def __mod__(self, other):
        """Return a copy of this mesh with ``self.data % other`` as the active scalar."""
        return self._binary_op(other, np.mod)

    def __rmod__(self, other):
        """Return a copy of this mesh with ``other % self.data`` as the active scalar."""
        return self._binary_op(other, np.mod, swap=True)

    def __pow__(self, other):
        """Return a copy of this mesh with ``self.data ** other`` as the active scalar."""
        return self._binary_op(other, np.power)

    def __rpow__(self, other):
        """Return a copy of this mesh with ``other ** self.data`` as the active scalar."""
        return self._binary_op(other, np.power, swap=True)

    def __neg__(self):
        """Return a copy of this mesh with negated active scalar data."""
        result = self.copy()
        result.data = np.negative(self.data)
        return result

    def __pos__(self):
        """Return a shallow copy of this mesh (unary ``+`` is a no-op on scalars)."""
        return self.copy()

    def __abs__(self):
        """Return a copy of this mesh with absolute values of the active scalar data."""
        result = self.copy()
        result.data = np.abs(self.data)
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Apply a NumPy ufunc element-wise to the active scalar data.

        Allows expressions like ``np.sqrt(mesh)`` or ``np.log10(mesh1 / mesh2)``
        to return a new mesh of the same type rather than a plain array.

        Only single-output ufuncs called via ``__call__`` are supported;
        all other cases return :obj:`NotImplemented`.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc being applied.
        method : str
            The ufunc dispatch method (only ``'__call__'`` is supported).
        *inputs
            The ufunc inputs.  Instances of this mesh class are replaced by their
            :attr:`data` arrays before the ufunc is called.
        **kwargs
            Additional keyword arguments forwarded to the ufunc.

        Returns
        -------
        out : CartesianMesh | SphericalMesh
            A copy of this mesh with the ufunc result as the active scalar.
        """
        if method != '__call__' or ufunc.nout != 1:
            return NotImplemented
        new_inputs = [x.data if isinstance(x, type(self)) else x for x in inputs]
        result = self.copy()
        result.data = getattr(ufunc, method)(*new_inputs, **kwargs)
        return result

    def _binary_op(self, other, ufunc, swap: bool = False):
        """Apply a binary ufunc to ``self.data`` and ``other``.

        Parameters
        ----------
        other : CartesianMesh | SphericalMesh | ArrayLike
            Right-hand operand.  If it is the same type as ``self``, its
            :attr:`data` is used; otherwise it is used directly (scalar or array
            broadcast).
        ufunc : numpy.ufunc
            Binary ufunc to apply (e.g. :func:`numpy.add`).
        swap : bool, optional
            When ``True``, swap the operand order to support reflected operations
            (e.g. ``other - self``).  Default is ``False``.

        Returns
        -------
        out : CartesianMesh | SphericalMesh
            A copy of ``self`` with the ufunc result as the active scalar.
        """
        result = self.copy()
        a = self.data
        b = other.data if isinstance(other, type(self)) else other
        result.data = ufunc(b, a) if swap else ufunc(a, b)
        return result

    @singledispatchmethod
    def _dispatch_input(self,
                        uinput,
                        *args,
                        **kwargs):
        """Dispatch constructor input to the appropriate parsing handler.

        The base implementation raises :class:`NotImplementedError`.  Three
        specialisations registered below handle the accepted input types:
        file paths, raw coordinate arrays, and existing PyVista datasets.

        Parameters
        ----------
        uinput
            Primary input.  Must be one of :class:`str` / :class:`pathlib.Path`,
            :class:`~collections.abc.Iterable` / :class:`~numbers.Number`, or
            :class:`pyvista.DataSet`.
        *args
            Additional positional arguments whose meaning depends on ``uinput``.
        **kwargs
            Keyword arguments forwarded to ``_parse_input`` or ``_dispatch_pyvista``.

        Raises
        ------
        NotImplementedError
            If ``uinput`` is not one of the three accepted types.
        """
        msg = f"Unsupported input type: {type(uinput)}"
        raise NotImplementedError(msg)

    @_dispatch_input.register
    def _(self,
          uinput: str | Path,
          *args,
          data,
          dataid,
          iformat,
          **kwargs):
        """Handle file-path input by reading HDF data via :func:`psi_io.read_hdf_by_index`."""
        ifile = Path(uinput)
        file_data, *scales = read_hdf_by_index(ifile, *args, dataset_id=dataid, return_scales=True)
        return self._parse_input(iformat, file_data if data is None else data, *scales, **kwargs)

    @_dispatch_input.register
    def _(self,
          uinput: Iterable | Number,
          *args,
          data,
          iformat,
          **kwargs):
        """Handle raw coordinate array input by forwarding to :meth:`_parse_input`."""
        return self._parse_input(iformat, data, uinput, *args, **kwargs)

    @_dispatch_input.register
    def _(self,
          uinput: pv.DataSet,
          *args,
          iformat,
          **kwargs):
        """Handle PyVista dataset input by forwarding to :meth:`_dispatch_pyvista`."""
        deep = kwargs.pop('deep', False)
        if args or kwargs:
            warnings.warn("Additional arguments are ignored when input is a PyVista DataSet.")
        return self._dispatch_pyvista(uinput, iformat, deep)

    def _parse_input(self,
                     iformat,
                     data,
                     *scales,
                     **kwargs):
        """Normalise the format string, reorder scales, and call :meth:`_set_arrays`.

        Parses the ``iformat`` string via :meth:`_parse_iformat` to obtain the
        per-axis order, maps the supplied ``scales`` to the correct axis slots,
        and then calls :meth:`_set_arrays` with consistently ordered arrays.

        Parameters
        ----------
        iformat : str | None
            Axis-order format string (e.g. ``'rtp'``, ``'ptr'``).  ``None``
            falls back to the canonical frame order.
        data : ArrayLike | None
            Scalar data to assign as the active scalar field.
        *scales
            Coordinate arrays in the order implied by ``iformat``.
        **kwargs
            Additional scale arrays that may be specified by axis letter
            (e.g. ``r=...``, ``t=...``).

        Raises
        ------
        ValueError
            If the number of scales does not match the format string, or if a
            required axis is missing.
        """
        sformat, sorder = self._parse_iformat(iformat)
        scales = atleast_1dnull(*scales, astuple=True)

        try:
            smap = dict(zip(sformat, scales, strict=True))
            smap |= {k: np.atleast_1d(kwargs.pop(k)) for k in sorder if k in kwargs}
        except ValueError as e:
            msg = f"Format string '{sformat}' does not match the number of scales read from the file: {len(scales)}."
            raise ValueError(msg) from e

        try:
            *scales, data = *(smap[k] for k in sorder), atleast_1dnull(data)
        except KeyError as e:
            missing = set(sorder) - smap.keys()
            msg = f"Missing scales for dimensions: {missing}. Provided scales: {set(smap.keys())}."
            raise ValueError(msg) from e

        self._set_arrays(sorder, data, *scales)

    @abstractmethod
    def _parse_iformat(self, iformat: Optional[str]) -> tuple[str, str]:
        """Parse ``iformat`` into a detected format string and the canonical axis order.

        Parameters
        ----------
        iformat : str | None
            Raw format string supplied by the user.

        Returns
        -------
        out : tuple[str, str]
            A 2-tuple ``(sformat, sorder)`` where ``sformat`` is the normalised
            detected axis string and ``sorder`` is the canonical axis-order key
            (e.g. ``'rtp'`` or ``'xyz'``).
        """
        ...

    @abstractmethod
    def _set_arrays(self, sorder, data, *scales):
        """Assign coordinate arrays and scalar data to the underlying PyVista dataset.

        Parameters
        ----------
        sorder : str
            Canonical axis-order key.
        data : np.ndarray | None
            Scalar data to assign as the active scalar field.
        *scales : np.ndarray
            Coordinate arrays in the order given by ``sorder``.
        """
        ...

    @abstractmethod
    def _dispatch_pyvista(self, uinput, iformat, deep):
        """Populate this mesh from an existing :class:`pyvista.DataSet`.

        Parameters
        ----------
        uinput : pyvista.DataSet
            Source dataset.
        iformat : str | None
            Optional frame format string; used to convert coordinates if needed.
        deep : bool
            If ``True``, perform a deep copy; otherwise shallow copy.

        Raises
        ------
        TypeError
            If ``uinput`` is not of a supported PyVista subtype.
        """
        ...

    @property
    def scales(self):
        """The three coordinate axis arrays as a tuple ``(x, y, z)``.

        For :class:`SphericalMesh` this corresponds to ``(r, t, p)``; for
        :class:`CartesianMesh` it corresponds to the three Cartesian axes.

        Returns
        -------
        out : tuple[np.ndarray, np.ndarray, np.ndarray]
            A 3-tuple of 1-D or 3-D coordinate arrays depending on the mesh type.
        """
        return self.x, self.y, self.z

    @property
    def data(self):
        """The active scalar field reshaped to the mesh dimensions in Fortran order.

        Returns ``None`` when no scalar data has been assigned.

        Returns
        -------
        out : np.ndarray | None
            Array of shape ``self.dimensions`` with Fortran memory order, or
            ``None`` if no active scalar is set.
        """
        if self.active_scalars is None:
            return None
        return self.active_scalars.reshape(self.dimensions, order='F')

    @data.setter
    def data(self, value):
        if value is None:
            self.clear_data()
        else:
            self[self.active_scalars_name or 'Data'] = parse_data(value, self.dimensions, order='F')


class CartesianMesh(_BaseFrameMesh, pv.StructuredGrid, CartesianMeshFilters):
    """A :class:`pyvista.StructuredGrid` for data on a Cartesian :math:`(x, y, z)` grid.

    Inherits the full :class:`_BaseFrameMesh` operator suite, the
    :class:`CartesianMeshFilters` methods, and PyVista's structured-grid API.
    The class-level :attr:`MESH_FRAME` is ``'cartesian'`` and is stamped into
    :attr:`~pyvista.DataSet.user_dict` on construction.

    Coordinate format strings like ``'xyz'``, ``'yxz'``, or any single-axis
    permutation are accepted via the ``iformat`` constructor argument; spherical
    inputs (``'rtp'``) are automatically converted to Cartesian before being
    stored.

    .. note::
       The general motivation behind this class (in contrast to the :class:`SphericalMesh`
       class) is to facilitate the use of PyVista/VTK's
       `filters <https://docs.pyvista.org/api/core/filters>`_ on spherical grids that have
       been converted to Cartesian coordinates. This :func:`~pyvisual.utils.geometry.spherical_to_cartesian`
       transformation yeilds topological structured meshes that are, nevertheless, not composed
       of monotonically increasing coordinate arrays. Therefore, the grid's internal structure
       has to be stored explicitly *viz.* through a :class:`pyvista.StructuredGrid` rather than a
       :class:`pyvista.RectilinearGrid`.

    .. warning::
       The consequence of the above note is that the point arrays of this class are
       derived from a :func:`~numpy.meshgrid`-like Cartesian product of the input scales,
       and are not stored as the three separate 1-D arrays that are typical of rectilinear grids.
       As such, these grids can be substantially more memory-intensive than their
       :class:`SphericalMesh` counterparts.

    Parameters
    ----------
    *args
        See :meth:`_BaseFrameMesh.__init__`.
    data : ArrayLike | None, optional
        Scalar data array.  Default is ``None``.
    dataid : str | None, optional
        Array name used when assigning ``data``.  Default is ``None``.
    iformat : str | None, optional
        Axis-order format string (e.g. ``'xyz'``, ``'yxz'``).  Default is
        ``None`` (assumes ``'xyz'`` order).
    **kwargs
        Forwarded to :meth:`_BaseFrameMesh.__init__`.

    Examples
    --------
    .. pyvista-plot::

        >>> import numpy as np
        >>> from pyvisual.core.mesh3d import CartesianMesh
        >>> x = np.linspace(-5, 5, 20)
        >>> y = np.linspace(-5, 5, 20)
        >>> z = np.linspace(-5, 5, 20)
        >>> mesh = CartesianMesh(x, y, z)

    See Also
    --------
    :class:`SphericalMesh`
        Companion class for spherical :math:`(r, \\theta, \\phi)` grids.
    """

    MESH_FRAME = 'cartesian'

    def _parse_iformat(self, iformat: Optional[str]) -> tuple[str, str]:
        msg = ""
        if not iformat:
            msg = "No format string provided. Assuming 'xyz' order for scales. "
            sorder = FRAME_SCALES[self.MESH_FRAME]
            sformat = _normalize_frame(iformat or sorder)
        else:
            try:
                sformat = iformat.lower()
                sorder = FRAME_SCALES[fetch_canonical_frame(sformat)]
            except KeyError as e:
                msg = (f"Unrecognized format string. "
                       f"Expected one of: {set(FRAME_ALIASES.keys())}. ")
                raise ValueError(msg) from e
        if set(sformat) - set(sorder):
            msg = f"{msg}Format string must only contain '{sorder}', got '{sformat}'"
            raise ValueError(msg)
        return sformat, sorder

    def _set_arrays(self, sorder, data, *scales):
        if any(d0.shape != d1.shape for (d0, d1) in pairwise(scales)):
            scales = ij_meshgrid(*scales)
        if transforms := generate_transforms(sorder, self.MESH_FRAME):
            scales = reduce(lambda v, f: f(*v), transforms, scales)
        x, y, z = scales
        self._from_arrays(x=x, y=y, z=z)
        self.data = data

    def _dispatch_pyvista(self, uinput, iformat, deep):
        if isinstance(uinput, (pv.StructuredGrid, pv.RectilinearGrid, pv.ImageData)):
            mesh = apply_mesh_transform(uinput, iformat, self.MESH_FRAME)
            self.deep_copy(mesh) if deep else self.shallow_copy(mesh)
        else:
            msg = (f"Cannot construct CartesianMesh from {type(uinput).__name__}; "
                   f"expected StructuredGrid, RectilinearGrid, or ImageData")
            raise TypeError(msg)

    def __getitem__(self, key):
        """Slice the mesh grid or retrieve a named point/cell array.

        When ``key`` is a plain string or integer it falls through to the
        standard PyVista ``__getitem__`` (array lookup).  Tuple and ellipsis
        keys are expanded and forwarded to PyVista's structured-grid slicing,
        then wrapped in a new :class:`CartesianMesh`.

        Parameters
        ----------
        key : str | int | tuple | Ellipsis
            Index key.  Strings and bare integers retrieve named data arrays;
            tuples (with optional :obj:`Ellipsis`) perform spatial slicing.

        Returns
        -------
        out : CartesianMesh | np.ndarray
            A sliced :class:`CartesianMesh` for spatial keys, or the requested
            array for string/integer keys.
        """
        if not isinstance(key, tuple) and key is not Ellipsis:
            return super(_BaseFrameMesh, self).__getitem__(key)
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            n_missing = len(self.dimensions) - (len(key) - 1)
            idx = key.index(Ellipsis)
            key = key[:idx] + (slice(None),) * n_missing + key[idx + 1:]

        return CartesianMesh(super(_BaseFrameMesh, self).__getitem__(key))


class SphericalMesh(_BaseFrameMesh, pv.RectilinearGrid, SphericalMeshFilters):
    """A :class:`pyvista.RectilinearGrid` for data on a spherical :math:`(r, \\theta, \\phi)` grid.

    The three PSI spherical axes — radius :math:`r`, colatitude :math:`\\theta`,
    and longitude :math:`\\phi` — are stored in the underlying
    :class:`pyvista.RectilinearGrid` as the ``x``, ``y``, and ``z`` axes
    respectively.  Convenience properties :attr:`r`, :attr:`t`, and :attr:`p`
    alias these axes for clarity.

    Inherits the full :class:`_BaseFrameMesh` operator suite, the
    :class:`SphericalMeshFilters` methods, and PyVista's rectilinear-grid API.
    The class-level :attr:`MESH_FRAME` is ``'spherical'``.

    Parameters
    ----------
    *args
        See :meth:`_BaseFrameMesh.__init__`.
    data : ArrayLike | None, optional
        Scalar data array.  Default is ``None``.
    dataid : str | None, optional
        Array name.  Default is ``None``.
    iformat : str | None, optional
        Axis-order format string (e.g. ``'rtp'``, ``'ptr'``).  Default is
        ``None`` (assumes ``'rtp'`` order).
    **kwargs
        Forwarded to :meth:`_BaseFrameMesh.__init__`.

    Examples
    --------
    Construct from 1-D axis arrays::

        >>> import numpy as np
        >>> from pyvisual.core.mesh3d import SphericalMesh
        >>> r = np.linspace(1, 5, 20)
        >>> t = np.linspace(0, np.pi, 30)
        >>> p = np.linspace(0, 2 * np.pi, 60)
        >>> mesh = SphericalMesh(r, t, p)
        >>> mesh.dimensions
        (20, 30, 60)

    Slice along the radial axis::

        >>> sub = mesh[5:10, ...]
        >>> sub.dimensions
        (5, 30, 60)

    See Also
    --------
    :class:`CartesianMesh`
        Companion class for Cartesian :math:`(x, y, z)` grids.
    :meth:`SphericalMeshFilters.spherical_to_cartesian`
        Convert this mesh's coordinates to Cartesian for direct PyVista rendering.
    """

    MESH_FRAME = 'spherical'

    def _parse_iformat(self, iformat: Optional[str]) -> tuple[str, str]:
        sorder = FRAME_SCALES[self.MESH_FRAME]
        sformat = _normalize_frame(iformat or sorder)
        if set(sformat) - set(sorder):
            msg = f"Format string must only contain '{sorder}', got '{sformat}'"
            raise ValueError(msg)
        return sformat, sorder

    def _set_arrays(self, iformat, data, *scales):
        r, t, p = scales
        self._from_arrays(x=r, y=t, z=p)
        self.data = data

    def _dispatch_pyvista(self, uinput, iformat, deep):
        if isinstance(uinput, (pv.RectilinearGrid, pv.ImageData)):
            self.deep_copy(uinput) if deep else self.shallow_copy(uinput)
        else:
            msg = (f"Cannot construct SphericalMesh from {type(uinput).__name__}; "
                   f"expected RectilinearGrid, or ImageData")
            raise TypeError(msg)

    def __getitem__(self, key):
        """Slice the mesh grid or retrieve a named point/cell array.

        When ``key`` is a plain string or integer it falls through to the
        standard PyVista ``__getitem__`` (array lookup).  Tuple and ellipsis
        keys perform spatial slicing: coordinate axes and the scalar data are
        sliced independently and a new :class:`SphericalMesh` is constructed
        from the results.

        Parameters
        ----------
        key : str | int | tuple | Ellipsis
            Index key.  Strings retrieve named data arrays; tuples (with
            optional :obj:`Ellipsis`) index the ``(r, t, p)`` axes.

        Returns
        -------
        out : SphericalMesh | np.ndarray
            A sliced :class:`SphericalMesh` for spatial keys, or the requested
            array for string/integer keys.

        Examples
        --------
        Slice the radial axis::

            >>> sub = mesh[5:10, ...]

        Fix colatitude to a single value::

            >>> equatorial = mesh[:, 45, :]
        """
        # legacy behavior which looks for a point or cell array
        if not isinstance(key, tuple) and key is not Ellipsis:
            return super(_BaseFrameMesh, self).__getitem__(key)
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            n_missing = len(self.dimensions) - (len(key) - 1)
            idx = key.index(Ellipsis)
            key = key[:idx] + (slice(None),) * n_missing + key[idx + 1:]

        scales = np.atleast_1d(*(scale[index].copy() for scale, index in zip(self.scales, key, strict=True)))
        data = np.atleast_1d(self.data[key].copy()) if self.data is not None else None
        return SphericalMesh(*scales, data=data)

    @property
    def r(self):
        """Radial axis :math:`r` (alias for :attr:`~pyvista.RectilinearGrid.x`).

        Returns
        -------
        out : np.ndarray
            1-D array of radial coordinate values in solar radii
            :math:`R_\\odot`.
        """
        return self.x

    @r.setter
    def r(self, value):
        self.x = value

    @property
    def t(self):
        """Colatitude axis :math:`\\theta` (alias for :attr:`~pyvista.RectilinearGrid.y`).

        Returns
        -------
        out : np.ndarray
            1-D array of colatitude values in radians, measured from the north
            pole.
        """
        return self.y

    @t.setter
    def t(self, value):
        self.y = value

    @property
    def p(self):
        """Longitude axis :math:`\\phi` (alias for :attr:`~pyvista.RectilinearGrid.z`).

        Returns
        -------
        out : np.ndarray
            1-D array of longitude values in radians.
        """
        return self.z

    @p.setter
    def p(self, value):
        self.z = value

    @property
    def interpolator(self):
        """A :class:`~scipy.interpolate.RegularGridInterpolator` over the active scalar data.

        Constructed with ``bounds_error=False`` so that out-of-bounds queries
        return ``NaN`` rather than raising an exception.

        Returns
        -------
        out : scipy.interpolate.RegularGridInterpolator
            Interpolator for the current :attr:`data` array on the
            :math:`(r, \\theta, \\phi)` grid.
        """
        return RegularGridInterpolator(self.scales, self.data, bounds_error=False)

    def cast_to_cartesian_mesh(self):
        """Convert this mesh to a :class:`CartesianMesh`.

        Passes ``self`` to :class:`CartesianMesh`, which calls
        :meth:`_dispatch_pyvista` to apply the spherical-to-Cartesian coordinate
        transform via :func:`~pyvisual.core.parsers.apply_mesh_transform`.

        Returns
        -------
        out : CartesianMesh
            A new :class:`CartesianMesh` with points in Cartesian coordinates
            and the same active scalar data.
        """
        return CartesianMesh(self)

