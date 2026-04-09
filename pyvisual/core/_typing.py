"""
Type aliases and named tuples for the **pyvisual** core.

All public type aliases and named tuples used for structured coordinate and observer
data are defined here.  Importing from this module keeps type annotations consistent
across :mod:`~pyvisual.core.mixins`, :mod:`~pyvisual.core.mesh3d`, and
:mod:`~pyvisual.core.parsers`.

Named tuples
------------
:class:`SolarCoordinate`
    Heliographic Carrington coordinates ``(radius, b_angle, longitude)``.
:class:`SphericalCoordinate`
    PSI spherical coordinates ``(r, t, p)`` — radius, colatitude, longitude.
:class:`CartesianCoordinate`
    Cartesian coordinates ``(x, y, z)``.
:class:`ObserverView`
    Line-of-sight field-of-view extents ``(x0, x1, y0, y1)`` in degrees.
:class:`ObserverOrientation`
    Observer position angle ``(p_angle,)`` in degrees.
"""

from collections import namedtuple
from os import PathLike
from pathlib import Path

from typing import Literal, Union, TypeAlias

from pyvista import VectorLike

PlottableType: TypeAlias = Union[
    VectorLike[float], 'DataSet', 'MultiBlock', 'PartitionedDataSet', str, Path
]

PathType = str | Path | PathLike[str]
"""Type alias for filesystem path arguments.

Accepted wherever a file path can be passed — covers plain :class:`str`,
:class:`pathlib.Path`, and any :class:`os.PathLike` object that yields a string.
"""

FlColorType = Literal['random', 'polarity']
"""Literal type for fieldline coloring strategies.

``'random'``
    Each fieldline is assigned a random hue from the ``'hsv'`` colormap.
``'polarity'``
    Fieldlines are colored by their open/closed magnetic polarity state, using the
    five-category scheme defined in :data:`~pyvisual.core._styling.FL_STATE_ANNOTATIONS`.
"""

FL_COLOR_TYPE = {'random', 'polarity'}
"""Runtime set of valid fieldline coloring strategy strings.

Mirrors :data:`FlColorType` as a plain :class:`set` for membership testing in
validation logic.
"""

SurfaceReconstructionType = Literal['delaunay_2d', 'delaunay_3d', 'reconstruct_surface']
"""Literal type for surface-reconstruction methods used in
:func:`~pyvisual.core.mesh3d.build_surface_polydata`.

``'delaunay_2d'``
    Project points onto a plane and triangulate via :meth:`pyvista.PolyData.delaunay_2d`.
``'delaunay_3d'``
    Full 3-D Delaunay tetrahedralisation followed by surface extraction.
``'reconstruct_surface'``
    Implicit surface reconstruction via :meth:`pyvista.PolyData.reconstruct_surface`.
"""

SURFACE_RECONSTRUCTION_TYPE = {'delaunay_2d', 'delaunay_3d', 'reconstruct_surface'}
"""Runtime set of valid surface-reconstruction method strings.

Mirrors :data:`SurfaceReconstructionType` as a plain :class:`set`.
"""

MeshFramesType = Literal["cartesian", "spherical"]
"""Literal type for the two supported mesh coordinate frames."""

PlotterFramesType = Literal["cartesian"]
"""Literal type for the plotter's internal coordinate frame.

:class:`~pyvisual.core.plot3d.Plot3d` always renders in the Cartesian frame;
meshes in other frames are converted before being passed to PyVista.
"""

SolarCoordinate = namedtuple('SolarCoordinate', ['radius', 'b_angle', 'longitude'])
"""Named tuple representing a heliographic Carrington coordinate.

Fields
------
radius : float
    Heliocentric distance in solar radii :math:`R_\\odot`.
b_angle : float
    Heliographic latitude (B\\ :sub:`0` angle) in degrees.
longitude : float
    Carrington longitude in degrees.
"""

SphericalCoordinate = namedtuple('SphericalCoordinate', ['r', 't', 'p'])
"""Named tuple representing a PSI spherical coordinate :math:`(r, \\theta, \\phi)`.

Fields
------
r : float
    Radial distance in solar radii :math:`R_\\odot`.
t : float
    Colatitude :math:`\\theta` in radians, measured from the north pole.
p : float
    Longitude :math:`\\phi` in radians.
"""

CartesianCoordinate = namedtuple('CartesianCoordinates', ['x', 'y', 'z'])
"""Named tuple representing a Cartesian coordinate :math:`(x, y, z)`.

Fields
------
x : float
    Cartesian x-coordinate in solar radii :math:`R_\\odot`.
y : float
    Cartesian y-coordinate in solar radii :math:`R_\\odot`.
z : float
    Cartesian z-coordinate in solar radii :math:`R_\\odot`.
"""

ObserverView = namedtuple('ObserverView', ['x0', 'x1', 'y0', 'y1'])
"""Named tuple representing line-of-sight field-of-view extents in degrees.

Fields
------
x0 : float
    Left edge of the horizontal (elongation) FOV extent in degrees.
x1 : float
    Right edge of the horizontal (elongation) FOV extent in degrees.
y0 : float
    Bottom edge of the vertical (altitude) FOV extent in degrees.
y1 : float
    Top edge of the vertical (altitude) FOV extent in degrees.
"""

ObserverOrientation = namedtuple('ObserverOrientation', ['p_angle'])
"""Named tuple representing the observer's position angle.

Fields
------
p_angle : float
    Roll of the camera about the line-of-sight axis, measured from solar north,
    in degrees.  A value of ``0`` means solar north points straight up in the image
    plane.
"""
