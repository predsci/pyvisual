r"""Solar-physics geometry utilities for coordinate transforms, rotations, and ephemeris queries.

This module provides the low-level mathematical building blocks used throughout
:mod:`pyvisual` to work with data on spherical grids in the PSI
:math:`(r, \theta, \phi)` coordinate system and to interface with
:mod:`sunpy` and :mod:`astropy` for solar-frame transformations.

Coordinate Convention
---------------------
All functions in this module follow the **physics/colatitude** convention:

- :math:`r` — radial distance (in :math:`R_\odot` unless otherwise stated).
- :math:`\theta` (``t``) — *colatitude* from the :math:`+z` axis (solar north),
  :math:`\theta \in [0, \pi]`.
- :math:`\phi` (``p``) — azimuth from :math:`+x` toward :math:`+y`,
  :math:`\phi \in [0, 2\pi)`.

This matches the internal PSI convention used in MAS model output files.  For
details of the PSI file format see the
`psi-io documentation <https://predsci.com/doc/psi-io/guide/overview.html>`_.

Contents
--------
- Scalar coordinate transforms: :func:`cartesian_to_spherical`,
  :func:`spherical_to_cartesian`.
- Vector-basis rotations: :func:`cartesian_to_spherical_vec`,
  :func:`spherical_to_cartesian_vec`.
- Rigid-body rotations: :func:`rotate_position_about_x`,
  :func:`rotate_position_about_y`, :func:`rotate_position_about_z`.
- Solar-imagery geometry: :func:`thompson_sphere`, :func:`los_rmin2angle`,
  :func:`los_angle2rmin`, :func:`clip_angle`.
- Ephemeris and trajectory utilities: :func:`query_horizons_ephemeris`,
  :func:`spacecraft_trajectory`.
- Sphere sampling: :func:`fibonacci_lattice`, :func:`cartesian_pointmesh`.
- Camera utilities: :func:`camera_roll_wrt_solar_north`.
- Array partials: :data:`ij_meshgrid`, :data:`moveaxis_to_start`.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import astropy.time as astro_time
import astropy.coordinates as astro_coord
import sunpy.coordinates as sun_coord
import astropy.units as u
from astropy.table import QTable
from numpy.typing import ArrayLike

from pyvisual.core.constants import SOLAR_NORTH, TWOPI, timestamp_format_ms

ij_meshgrid = partial(np.meshgrid, indexing='ij')
r""":func:`numpy.meshgrid` with ``indexing='ij'`` (matrix indexing) pre-applied.

:func:`numpy.meshgrid` defaults to ``indexing='xy'``, which transposes the
first two output arrays relative to the input order.  PSI spherical grids use
:math:`(r, \theta, \phi)` ordering that must be preserved after broadcasting,
so ``indexing='ij'`` is always required.  This partial fixes that argument so
call sites cannot accidentally omit it.

The resulting broadcasted arrays follow the same axis ordering as the inputs:
the first axis corresponds to :math:`r`, the second to :math:`\theta`, and the
third to :math:`\phi`.

Parameters
----------
*xi : ArrayLike
    1-D coordinate arrays to broadcast, typically ``(r, t, p)``.
**kwargs
    Any additional keyword arguments accepted by :func:`numpy.meshgrid`
    (e.g. ``copy``, ``sparse``).  ``indexing`` is already fixed to ``'ij'``
    and cannot be overridden.

Returns
-------
out : list[np.ndarray]
    List of broadcasted arrays, one per input, each with shape
    ``(len(r), len(t), len(p))`` for three 1-D inputs.

See Also
--------
:func:`numpy.meshgrid` : The underlying NumPy function.
:data:`moveaxis_to_start` : Companion partial used to reorder the stack axis
    after broadcasting.

Examples
--------
Broadcast three 1-D spherical coordinate arrays into 3-D grids:

>>> import numpy as np
>>> r = np.linspace(1, 5, 4)
>>> t = np.linspace(0, np.pi, 6)
>>> p = np.linspace(0, 2 * np.pi, 8)
>>> R, T, P = ij_meshgrid(r, t, p)
>>> R.shape
(4, 6, 8)
"""

moveaxis_to_start = partial(np.moveaxis, destination=0)
""":func:`numpy.moveaxis` with ``destination=0`` pre-applied.

Moves a nominated source axis to position 0 (the leading axis) while keeping
all other axes in their original relative order.  This is used throughout the
polydata builder functions in :mod:`pyvisual.core.mesh3d` to bring the
"stack" axis to the front before reshaping, enabling uniform downstream
iteration over batches of curves or surface slices.

For example, if a fieldline array has shape ``(N_r, N_lines)`` with the line
index on axis 1, ``moveaxis_to_start(arr, 1)`` returns a view of shape
``(N_lines, N_r)`` without copying data.

Parameters
----------
a : np.ndarray
    Input array.
source : int | Sequence[int]
    Original position(s) of the axis (or axes) to move.
**kwargs
    Any additional keyword arguments accepted by :func:`numpy.moveaxis`.
    ``destination`` is already fixed to ``0`` and cannot be overridden.

Returns
-------
out : np.ndarray
    View of ``a`` with the specified axis moved to position 0.

See Also
--------
:func:`numpy.moveaxis` : The underlying NumPy function.
:data:`ij_meshgrid` : Companion partial used to broadcast 1-D coordinate
    arrays before axis reordering.

Examples
--------
Move the line-index axis of a fieldline coordinate array to the front:

>>> import numpy as np
>>> arr = np.zeros((100, 5, 3))  # shape: (N_r, N_lines, xyz)
>>> out = moveaxis_to_start(arr, 1)
>>> out.shape
(5, 100, 3)
"""


def cartesian_to_spherical(x: ArrayLike,
                           y: ArrayLike,
                           z: ArrayLike
                           ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Convert Cartesian :math:`(x, y, z)` coordinates to spherical :math:`(r, \theta, \phi)`.

    Uses the physics/colatitude convention throughout pyvisual:

    - :math:`r` is the radial distance from the origin.
    - :math:`\theta` (``t``) is the *colatitude* measured from the
      :math:`+z` axis, :math:`\theta \in [0, \pi]`.
    - :math:`\phi` (``p``) is the azimuth measured from :math:`+x` toward
      :math:`+y` in the :math:`xy`-plane, wrapped to :math:`\phi \in [0, 2\pi)`.

    Parameters
    ----------
    x : ArrayLike
        :math:`x` component. Broadcast together with ``y`` and ``z`` following
        NumPy broadcasting rules.
    y : ArrayLike
        :math:`y` component.
    z : ArrayLike
        :math:`z` component.

    Returns
    -------
    r : np.ndarray
        Radial distance, :math:`r = \sqrt{x^2 + y^2 + z^2}`.
    t : np.ndarray
        Colatitude :math:`\theta = \arctan2\!\left(\sqrt{x^2+y^2},\, z\right)`,
        in :math:`[0, \pi]`.
    p : np.ndarray
        Longitude :math:`\phi = \operatorname{fmod}(\arctan2(y,\,x) + 2\pi,\, 2\pi)`,
        guaranteed to be in :math:`[0, 2\pi)`.

    Notes
    -----
    :math:`\theta` is computed via :math:`\arctan2\!\left(\sqrt{x^2+y^2}, z\right)`
    rather than :math:`\arccos(z/r)` for numerical stability near the poles.

    At the origin (:math:`x = y = z = 0`), :math:`r = 0` and both :math:`\theta`
    and :math:`\phi` evaluate to ``0`` (the behavior of ``arctan2(0, 0)``).

    See Also
    --------
    :func:`spherical_to_cartesian` : Inverse transform.
    :func:`cartesian_to_spherical_vec` : Equivalent rotation for vector components.

    Examples
    --------
    A point on the :math:`+x` axis lies at colatitude :math:`\theta = \pi/2`
    and longitude :math:`\phi = 0`:

    >>> r, t, p = cartesian_to_spherical(1.0, 0.0, 0.0)
    >>> float(r), float(t), float(p)
    (1.0, 1.5707963267948966, 0.0)

    A point on the :math:`+z` axis (solar north) has :math:`\theta = 0`:

    >>> r, t, p = cartesian_to_spherical(0.0, 0.0, 5.0)
    >>> float(r), float(t), float(p)
    (5.0, 0.0, 0.0)
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    r = np.sqrt(x2 + y2 + z2)
    t = np.arctan2(np.sqrt(x2 + y2), z)
    p = np.arctan2(y, x)

    # arctan2 returns values from -pi to pi but I want 0-2pi --> use fmod
    p = np.fmod(p + TWOPI, TWOPI)

    return r, t, p


def spherical_to_cartesian(r: ArrayLike,
                           t: ArrayLike,
                           p: ArrayLike
                           ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Convert spherical :math:`(r, \theta, \phi)` coordinates to Cartesian :math:`(x, y, z)`.

    Inverse of :func:`cartesian_to_spherical`, using the same physics/colatitude
    convention: :math:`\theta` is colatitude from :math:`+z` and :math:`\phi`
    is azimuth from :math:`+x` toward :math:`+y`.

    Parameters
    ----------
    r : ArrayLike
        Radial distance :math:`r \geq 0`. Broadcast together with ``t`` and
        ``p`` following NumPy broadcasting rules.
    t : ArrayLike
        Colatitude :math:`\theta \in [0, \pi]`.
    p : ArrayLike
        Longitude :math:`\phi \in [0, 2\pi)`.

    Returns
    -------
    x : np.ndarray
        :math:`x = r \sin\theta \cos\phi`.
    y : np.ndarray
        :math:`y = r \sin\theta \sin\phi`.
    z : np.ndarray
        :math:`z = r \cos\theta`.

    Notes
    -----
    The full transformation is:

    .. math::

       \begin{pmatrix} x \\ y \\ z \end{pmatrix}
       =
       r \begin{pmatrix}
           \sin\theta\cos\phi \\
           \sin\theta\sin\phi \\
           \cos\theta
       \end{pmatrix}

    See Also
    --------
    :func:`cartesian_to_spherical` : Inverse transform.
    :func:`spherical_to_cartesian_vec` : Equivalent rotation for vector components.

    Examples
    --------
    A point at :math:`(r, \theta, \phi) = (1, \pi/2, \pi/2)` lies on the
    :math:`+y` axis:

    >>> import numpy as np
    >>> x, y, z = spherical_to_cartesian(1.0, np.pi / 2, np.pi / 2)
    >>> float(x), float(y), float(z)
    (0.0, 1.0, 0.0)

    Solar north (:math:`\theta = 0`) always maps to the :math:`+z` axis:

    >>> x, y, z = spherical_to_cartesian(3.0, 0.0, 0.0)
    >>> float(x), float(y), float(z)
    (0.0, 0.0, 3.0)
    """
    ct = np.cos(t)
    st = np.sin(t)
    cp = np.cos(p)
    sp = np.sin(p)
    x = r*cp*st
    y = r*sp*st
    z = r*ct
    return x, y, z


def cartesian_to_spherical_vec(vr: ArrayLike,
                               vt: ArrayLike,
                               vp: ArrayLike,
                               t: ArrayLike,
                               p: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Rotate vector components from the local spherical basis to the Cartesian basis.

    Given a vector expressed in the local orthonormal spherical basis
    :math:`(\hat{e}_r, \hat{e}_\theta, \hat{e}_\phi)` at angular position
    :math:`(\theta, \phi)`, returns the equivalent components in the global
    Cartesian basis :math:`(\hat{x}, \hat{y}, \hat{z})`.

    The spherical basis vectors follow the PSI/physics colatitude convention:

    - :math:`\hat{e}_r` — radial outward.
    - :math:`\hat{e}_\theta` — in the direction of increasing :math:`\theta`
      (pointing toward :math:`-z` near the north pole).
    - :math:`\hat{e}_\phi` — in the direction of increasing :math:`\phi`
      (prograde around :math:`+z`).

    Parameters
    ----------
    vr : ArrayLike
        Radial component :math:`v_r` in the spherical basis. Broadcast together
        with all other inputs following NumPy broadcasting rules.
    vt : ArrayLike
        Colatitudinal component :math:`v_\theta` in the spherical basis.
    vp : ArrayLike
        Azimuthal component :math:`v_\phi` in the spherical basis.
    t : ArrayLike
        Colatitude :math:`\theta` of the evaluation point(s), in :math:`[0, \pi]`.
    p : ArrayLike
        Longitude :math:`\phi` of the evaluation point(s), in :math:`[0, 2\pi)`.

    Returns
    -------
    vx : np.ndarray
        :math:`x`-component in the Cartesian basis.
    vy : np.ndarray
        :math:`y`-component in the Cartesian basis.
    vz : np.ndarray
        :math:`z`-component in the Cartesian basis.

    Notes
    -----
    The rotation is the transpose (inverse) of the Jacobian of
    :func:`spherical_to_cartesian` evaluated at :math:`(\theta, \phi)`:

    .. math::

       \begin{pmatrix} v_x \\ v_y \\ v_z \end{pmatrix}
       =
       \begin{pmatrix}
           \sin\theta\cos\phi & \cos\theta\cos\phi & -\sin\phi \\
           \sin\theta\sin\phi & \cos\theta\sin\phi &  \cos\phi \\
           \cos\theta          & -\sin\theta          &  0
       \end{pmatrix}
       \begin{pmatrix} v_r \\ v_\theta \\ v_\phi \end{pmatrix}

    Inputs are assumed to be orthonormal spherical-basis components, not
    covariant or contravariant coordinate-basis components.

    See Also
    --------
    :func:`spherical_to_cartesian_vec` : Inverse rotation (Cartesian → spherical basis).
    :func:`cartesian_to_spherical` : Scalar coordinate transform.
    :func:`spherical_to_cartesian` : Inverse scalar coordinate transform.

    Examples
    --------
    A purely azimuthal unit vector :math:`\hat{e}_\phi` at
    :math:`\theta = \pi/2,\, \phi = 0` points along :math:`+y`:

    >>> import numpy as np
    >>> vx, vy, vz = cartesian_to_spherical_vec(0.0, 0.0, 1.0, t=np.pi / 2, p=0.0)
    >>> float(vx), float(vy), float(vz)
    (0.0, 1.0, 0.0)

    A purely radial unit vector :math:`\hat{e}_r` at the same location points
    along :math:`+x`:

    >>> vx, vy, vz = cartesian_to_spherical_vec(1.0, 0.0, 0.0, t=np.pi / 2, p=0.0)
    >>> float(vx), float(vy), float(vz)
    (1.0, 0.0, 0.0)
    """
    st = np.sin(t)
    ct = np.cos(t)
    sp = np.sin(p)
    cp = np.cos(p)

    # rotate the vector field components
    vx = vr*st*cp + vt*ct*cp - vp*sp
    vy = vr*st*sp + vt*ct*sp + vp*cp
    vz = vr*ct - vt*st

    return vx, vy, vz


def spherical_to_cartesian_vec(vx: ArrayLike,
                               vy: ArrayLike,
                               vz: ArrayLike,
                               t: ArrayLike,
                               p: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Rotate vector components from the Cartesian basis to the local spherical basis.

    Given a vector expressed in the global Cartesian basis
    :math:`(\hat{x}, \hat{y}, \hat{z})` at angular position
    :math:`(\theta, \phi)`, returns the equivalent components in the local
    orthonormal spherical basis
    :math:`(\hat{e}_r, \hat{e}_\theta, \hat{e}_\phi)`.

    The spherical basis vectors follow the PSI/physics colatitude convention:

    - :math:`\hat{e}_r` — radial outward.
    - :math:`\hat{e}_\theta` — in the direction of increasing :math:`\theta`.
    - :math:`\hat{e}_\phi` — in the direction of increasing :math:`\phi`
      (prograde around :math:`+z`).

    Parameters
    ----------
    vx : ArrayLike
        :math:`x`-component in the Cartesian basis. Broadcast together with all
        other inputs following NumPy broadcasting rules.
    vy : ArrayLike
        :math:`y`-component in the Cartesian basis.
    vz : ArrayLike
        :math:`z`-component in the Cartesian basis.
    t : ArrayLike
        Colatitude :math:`\theta` of the evaluation point(s), in :math:`[0, \pi]`.
    p : ArrayLike
        Longitude :math:`\phi` of the evaluation point(s), in :math:`[0, 2\pi)`.

    Returns
    -------
    vr : np.ndarray
        Radial component :math:`v_r` in the spherical basis.
    vt : np.ndarray
        Colatitudinal component :math:`v_\theta` in the spherical basis.
    vp : np.ndarray
        Azimuthal component :math:`v_\phi` in the spherical basis.

    Notes
    -----
    The rotation is the inverse (transpose) of the matrix in
    :func:`cartesian_to_spherical_vec`:

    .. math::

       \begin{pmatrix} v_r \\ v_\theta \\ v_\phi \end{pmatrix}
       =
       \begin{pmatrix}
           \sin\theta\cos\phi & \sin\theta\sin\phi &  \cos\theta \\
           \cos\theta\cos\phi & \cos\theta\sin\phi & -\sin\theta \\
           -\sin\phi           & \cos\phi             &  0
       \end{pmatrix}
       \begin{pmatrix} v_x \\ v_y \\ v_z \end{pmatrix}

    Inputs are assumed to be orthonormal spherical-basis components, not
    covariant or contravariant coordinate-basis components.

    See Also
    --------
    :func:`cartesian_to_spherical_vec` : Inverse rotation (spherical → Cartesian basis).
    :func:`spherical_to_cartesian` : Inverse scalar coordinate transform.
    :func:`cartesian_to_spherical` : Scalar coordinate transform.

    Examples
    --------
    A :math:`+z` unit vector at the north pole (:math:`\theta = 0`) is purely radial:

    >>> vr, vt, vp = spherical_to_cartesian_vec(0.0, 0.0, 1.0, t=0.0, p=0.0)
    >>> float(vr), float(vt), float(vp)
    (1.0, 0.0, 0.0)

    A :math:`+y` unit vector at :math:`\theta = \pi/2,\, \phi = 0` is purely
    azimuthal :math:`(\hat{e}_\phi)`:

    >>> import numpy as np
    >>> vr, vt, vp = spherical_to_cartesian_vec(0.0, 1.0, 0.0, t=np.pi / 2, p=0.0)
    >>> float(vr), float(vt), float(vp)
    (0.0, 0.0, 1.0)
    """
    st = np.sin(t)
    ct = np.cos(t)
    sp = np.sin(p)
    cp = np.cos(p)

    # rotate the vector field components
    vr = vx*st*cp + vy*st*sp + vz*ct
    vt = vx*ct*cp + vy*ct*sp - vz*st
    vp = -vx*sp + vy*cp

    return vr, vt, vp


def rotate_position_about_x(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    angle: float,) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Rotate Cartesian position vectors about the :math:`+x` axis.

    Applies a right-handed rotation by ``angle`` degrees about :math:`+x`
    to the input coordinates.

    Parameters
    ----------
    x : ArrayLike
        :math:`x`-coordinate. Unchanged by this rotation. Broadcast together
        with ``y`` and ``z`` following NumPy broadcasting rules.
    y : ArrayLike
        :math:`y`-coordinate.
    z : ArrayLike
        :math:`z`-coordinate.
    angle : float
        Rotation angle in **degrees**. Positive values follow the right-hand
        rule about :math:`+x`.

    Returns
    -------
    xout : np.ndarray
        :math:`x`-coordinate after rotation (copy of ``x``).
    yout : np.ndarray
        :math:`y`-coordinate after rotation.
    zout : np.ndarray
        :math:`z`-coordinate after rotation.

    Notes
    -----
    The rotation matrix about :math:`+x` by angle :math:`\alpha` is:

    .. math::

       \begin{pmatrix} x' \\ y' \\ z' \end{pmatrix}
       =
       \begin{pmatrix}
           1 & 0            &  0           \\
           0 & \cos\alpha & -\sin\alpha \\
           0 & \sin\alpha &  \cos\alpha
       \end{pmatrix}
       \begin{pmatrix} x \\ y \\ z \end{pmatrix}

    See Also
    --------
    :func:`rotate_position_about_y` : Rotation about the :math:`+y` axis.
    :func:`rotate_position_about_z` : Rotation about the :math:`+z` axis.

    Examples
    --------
    Rotate :math:`(0, 1, 0)` by :math:`+90°` about :math:`+x` to point along
    :math:`+z`:

    >>> x2, y2, z2 = rotate_position_about_x(0.0, 1.0, 0.0, 90.0)
    >>> float(x2), float(y2), float(z2)
    (0.0, 0.0, 1.0)
    """
    rad = np.deg2rad(angle)

    s = np.sin(rad)
    c = np.cos(rad)

    xout = x*0 + x  # enforce a copy that inherits the proper type
    yout = c*y - s*z
    zout = s*y + c*z

    return xout, yout, zout


def rotate_position_about_y(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    angle: float,) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Rotate Cartesian position vectors about the :math:`+y` axis.

    Applies a right-handed rotation by ``angle`` degrees about :math:`+y`
    to the input coordinates.

    Parameters
    ----------
    x : ArrayLike
        :math:`x`-coordinate. Broadcast together with ``y`` and ``z`` following
        NumPy broadcasting rules.
    y : ArrayLike
        :math:`y`-coordinate. Unchanged by this rotation.
    z : ArrayLike
        :math:`z`-coordinate.
    angle : float
        Rotation angle in **degrees**. Positive values follow the right-hand
        rule about :math:`+y`.

    Returns
    -------
    xout : np.ndarray
        :math:`x`-coordinate after rotation.
    yout : np.ndarray
        :math:`y`-coordinate after rotation (copy of ``y``).
    zout : np.ndarray
        :math:`z`-coordinate after rotation.

    Notes
    -----
    The rotation matrix about :math:`+y` by angle :math:`\alpha` is:

    .. math::

       \begin{pmatrix} x' \\ y' \\ z' \end{pmatrix}
       =
       \begin{pmatrix}
            \cos\alpha & 0 & \sin\alpha \\
            0            & 1 & 0            \\
           -\sin\alpha & 0 & \cos\alpha
       \end{pmatrix}
       \begin{pmatrix} x \\ y \\ z \end{pmatrix}

    See Also
    --------
    :func:`rotate_position_about_x` : Rotation about the :math:`+x` axis.
    :func:`rotate_position_about_z` : Rotation about the :math:`+z` axis.

    Examples
    --------
    Rotate :math:`(1, 0, 0)` by :math:`+90°` about :math:`+y` to point along
    :math:`-z`:

    >>> x2, y2, z2 = rotate_position_about_y(1.0, 0.0, 0.0, 90.0)
    >>> float(x2), float(y2), float(z2)
    (0.0, 0.0, -1.0)
    """
    rad = np.deg2rad(angle)

    s = np.sin(rad)
    c = np.cos(rad)

    xout = c*x + s*z
    yout = y*0 + y  # enforce a copy that inherits the proper type
    zout = -s*x + c*z

    return xout, yout, zout


def rotate_position_about_z(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    angle: float,) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Rotate Cartesian position vectors about the :math:`+z` axis.

    Applies a right-handed rotation by ``angle`` degrees about :math:`+z`
    (solar north) to the input coordinates.

    Parameters
    ----------
    x : ArrayLike
        :math:`x`-coordinate. Broadcast together with ``y`` and ``z`` following
        NumPy broadcasting rules.
    y : ArrayLike
        :math:`y`-coordinate.
    z : ArrayLike
        :math:`z`-coordinate. Unchanged by this rotation.
    angle : float
        Rotation angle in **degrees**. Positive values follow the right-hand
        rule about :math:`+z`.

    Returns
    -------
    xout : np.ndarray
        :math:`x`-coordinate after rotation.
    yout : np.ndarray
        :math:`y`-coordinate after rotation.
    zout : np.ndarray
        :math:`z`-coordinate after rotation (copy of ``z``).

    Notes
    -----
    The rotation matrix about :math:`+z` by angle :math:`\alpha` is:

    .. math::

       \begin{pmatrix} x' \\ y' \\ z' \end{pmatrix}
       =
       \begin{pmatrix}
           \cos\alpha & -\sin\alpha & 0 \\
           \sin\alpha &  \cos\alpha & 0 \\
           0            &  0            & 1
       \end{pmatrix}
       \begin{pmatrix} x \\ y \\ z \end{pmatrix}

    See Also
    --------
    :func:`rotate_position_about_x` : Rotation about the :math:`+x` axis.
    :func:`rotate_position_about_y` : Rotation about the :math:`+y` axis.

    Examples
    --------
    Rotate :math:`(1, 0, 0)` by :math:`+90°` about :math:`+z` to point along
    :math:`+y`:

    >>> x2, y2, z2 = rotate_position_about_z(1.0, 0.0, 0.0, 90.0)
    >>> float(x2), float(y2), float(z2)
    (0.0, 1.0, 0.0)
    """
    rad = np.deg2rad(angle)

    s = np.sin(rad)
    c = np.cos(rad)

    xout = c*x - s*y
    yout = s*x + c*y
    zout = z*0 + z  # enforce a copy that inherits the proper type

    return xout, yout, zout


def clip_angle(angle: ArrayLike,
               max_value: float = 180) -> np.ndarray | float:
    r"""Wrap an angle in degrees to a half-open interval of width 360°.

    Reduces ``angle`` modulo :math:`360°` and then shifts the result into
    :math:`(\mathtt{max\_value} - 360°,\, \mathtt{max\_value}]`.
    The two common use-cases are:

    - ``max_value=180`` → interval :math:`(-180°, 180°]`
    - ``max_value=360`` → interval :math:`(0°, 360°]`

    Parameters
    ----------
    angle : ArrayLike
        Angle or array of angles in degrees. Plain floats or NumPy arrays are
        accepted; :mod:`astropy` ``Quantity`` objects are not supported.
    max_value : float, optional
        Upper bound (inclusive) of the output interval. The lower bound is
        ``max_value - 360``. Default is ``180``.

    Returns
    -------
    out : np.ndarray | float
        Wrapped angle(s) with the same shape as ``angle``. Returns a Python
        scalar when ``angle`` is scalar, otherwise an :class:`numpy.ndarray`.

    See Also
    --------
    :func:`los_rmin2angle` : Uses ``clip_angle`` internally to handle
        back-facing lines of sight.
    :func:`los_angle2rmin` : Inverse; also clips the input angle.

    Examples
    --------
    Wrap angles to :math:`(-180°, 180°]`:

    >>> clip_angle(270.0)
    -90.0

    Wrap to :math:`(0°, 360°]`:

    >>> clip_angle(-45.0, max_value=360)
    315.0
    """
    angle_ = np.asarray(angle)
    mod_angle = np.mod(angle_, 360.)
    clipped_angle = np.where(mod_angle > max_value, mod_angle - 360., mod_angle)
    return clipped_angle.item() if clipped_angle.ndim == 0 else clipped_angle


def thompson_sphere(elong: ArrayLike,
                    alt: ArrayLike,
                    obs_lon: ArrayLike,
                    obs_lat: ArrayLike,
                    r_obs_rs: ArrayLike,
                    obs_pangle: ArrayLike = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the 3-D intersection of a line of sight with the Thomson sphere.

    The `Thomson sphere
    <https://doi.org/10.1007/s11207-006-0030-x>`_ is the sphere of radius
    :math:`d_{obs}/2` centered on the midpoint between the Sun and the observer.
    Every point on it satisfies the condition that the angle between the
    Sun-to-point vector and the point-to-observer vector is exactly
    :math:`90°`, making it the locus of maximum Thomson-scattering efficiency
    for white-light coronagraph observations.

    A line of sight (LOS) is specified by its helioprojective angles
    (:math:`T_x`, :math:`T_y`) and the observer's Heliographic Carrington
    position.  The function returns the Cartesian position of the point where
    that LOS intersects the Thomson sphere, expressed in Carrington
    coordinates.

    Parameters
    ----------
    elong : ArrayLike
        Elongation (helioprojective longitude, :math:`T_x`) of the LOS from
        Sun-center in **degrees**, in :math:`[-180°, 180°]`.
    alt : ArrayLike
        Altitude (helioprojective latitude, :math:`T_y`) of the LOS from
        Sun-center in **degrees**, in :math:`[-90°, 90°]`.
    obs_lon : ArrayLike
        Carrington longitude of the observer in **degrees**,
        in :math:`[-180°, 180°]`.
    obs_lat : ArrayLike
        Carrington latitude of the observer in **degrees**,
        in :math:`[-90°, 90°]`.
    r_obs_rs : ArrayLike
        Observer distance from Sun-center in :math:`R_\odot`.
    obs_pangle : ArrayLike, optional
        Solar P-angle: rotation of the observer frame relative to solar north
        in **degrees**. A positive value means solar north appears
        ``obs_pangle`` degrees clockwise from the image up direction.
        Default is ``0.0``.

    Returns
    -------
    x : np.ndarray
        Carrington :math:`x`-coordinate of the Thomson-sphere intersection
        point(s), in :math:`R_\odot`.
    y : np.ndarray
        Carrington :math:`y`-coordinate, in :math:`R_\odot`.
    z : np.ndarray
        Carrington :math:`z`-coordinate, in :math:`R_\odot`.

    Notes
    -----
    Either the observer position *or* the LOS angles may be arrays — not both.
    Mixing array observer positions with array LOS angles is not supported.

    The algorithm:

    1. Constructs the reference point on the sphere of radius :math:`d_{obs}`
       centered at the observer corresponding to the given helioprojective
       angles.
    2. Finds the parameter :math:`t` along the LOS at which the distance to
       Sun-center is minimised (the Thomson-sphere intersection).
    3. Applies the P-angle, B\ :sub:`0`-angle (Carrington latitude), and
       Carrington longitude rotations sequentially via
       :func:`rotate_position_about_x`, :func:`rotate_position_about_y`, and
       :func:`rotate_position_about_z`.

    For LOSs with :math:`|T_x| > 90°` (pointing away from the Sun), the
    intersection is reflected to the back Thomson sphere.

    See Also
    --------
    :func:`los_rmin2angle` : Convert impact parameter :math:`r_{min}` to LOS angle.
    :func:`los_angle2rmin` : Convert LOS angle to impact parameter.
    :func:`clip_angle` : Angle wrapping used internally.

    Examples
    --------
    Point on the plane-of-sky at Sun-center elongation (:math:`T_x = 0`,
    :math:`T_y = 0`) for an observer at :math:`1\,\text{AU} \approx 215\,R_\odot`.

    >>> x, y, z = thompson_sphere(0.0, 0.0, 0.0, 0.0, 215.0)
    >>> float(z)
    0.0
    """
    # Create a Cartesian vector from the Sun to the observer.
    v_obs_x = r_obs_rs
    v_obs_y = 0.
    v_obs_z = 0.

    # Create a Cartesian vector from the Sun to the reference point.
    # - This is a point on a sphere of radius R_OBS centered at the observer,
    #   where R_OBS is the distance of the observer from the Sun.
    e_rad = np.deg2rad(elong)
    a_rad = np.deg2rad(alt)
    cos_a = np.cos(a_rad)
    v_ref_x = r_obs_rs*(1.0 - np.cos(e_rad)*cos_a)
    v_ref_y = r_obs_rs*np.sin(e_rad)*cos_a
    v_ref_z = r_obs_rs*np.sin(a_rad)

    # Find the point along this LOS that is closest to the Sun, v_r_min.
    # - This is the point on the LOS at which it intersects the Thomson sphere.
    # - in getpb, t is in relative s/r_obs coordinates, which is used to interpolate
    #   the position vector between the vector on the R_OBS sphere and the observer
    #   location vector. Here we just copy the solve for t at r_min (Thomson sphere).
    t_r_min = -(v_obs_x*(v_ref_x - v_obs_x)
                + v_obs_y*(v_ref_y - v_obs_y)
                + v_obs_z*(v_ref_z - v_obs_z))/r_obs_rs**2
    v_r_min_x = t_r_min*v_ref_x + (1.0 - t_r_min)*v_obs_x
    v_r_min_y = t_r_min*v_ref_y + (1.0 - t_r_min)*v_obs_y
    v_r_min_z = t_r_min*v_ref_z + (1.0 - t_r_min)*v_obs_z

    # Check if this position is looking backwards or not
    # - if so, we subtract 2x the vector pointing from
    #   the observer location to the inner thompson sphere location
    # - This puts the location on the equivalent thompson sphere
    #   BEHIND the observer.
    # - use np.iscalar to allow for array or scalar elongation angles
    elong = clip_angle(elong, max_value=180.)
    check = np.abs(elong) > 90.
    if np.any(check):
        if np.isscalar(elong):
            mask = 1.0
        else:
            mask = np.where(check, 1.0, 0.0)
    else:
        mask = 0.0

    # initialize coordinates for rotation
    x = v_r_min_x - 2*(v_r_min_x - v_obs_x)*mask
    y = v_r_min_y - 2*(v_r_min_y - v_obs_y)*mask
    z = v_r_min_z - 2*(v_r_min_z - v_obs_z)*mask

    # Apply the solar P angle transformation
    x, y, z = rotate_position_about_x(x, y, z, obs_pangle)

    # Apply the solar B0 angle (latitude) transformation.
    x, y, z = rotate_position_about_y(x, y, z, -obs_lat)

    # Apply the solar longitude transformation.
    x, y, z = rotate_position_about_z(x, y, z, obs_lon)

    return x, y, z


def los_rmin2angle(rmin_rs: ArrayLike,
                   d_obs_rs: float) -> np.ndarray | float:
    r"""Convert a LOS impact parameter :math:`r_{min}` to a helioprojective angle.

    The *impact parameter* :math:`r_{min}` is the distance of closest approach
    of a line of sight to Sun-center (in :math:`R_\odot`).  This function
    inverts the geometric relationship

    .. math::

       r_{min} = d_{obs} \sin\alpha

    where :math:`\alpha` is the helioprojective elongation angle, with a
    smooth extension beyond :math:`90°` for LOSs that point partly away from
    the Sun (see Notes).

    Parameters
    ----------
    rmin_rs : ArrayLike
        Impact parameter(s) in :math:`R_\odot`.  Negative values are supported
        and encode the sign of the corresponding angle (useful for 1-D
        coordinate sweeps).
    d_obs_rs : float
        Observer distance from Sun-center in :math:`R_\odot`.

    Returns
    -------
    angle_deg : np.ndarray | float
        Helioprojective elongation angle(s) in degrees.  Returns a scalar when
        ``rmin_rs`` is scalar, otherwise an :class:`numpy.ndarray`.

    Notes
    -----
    For :math:`|r_{min}| \leq d_{obs}` the standard relation
    :math:`\alpha = \arcsin(r_{min}/d_{obs})` applies.  For larger values
    (back-hemisphere LOSs), the impact parameter is defined by continuity as

    .. math::

       r_{min} = d_{obs}(2 - \sin\alpha), \quad \alpha \in (90°, 180°]

    so the inversion becomes :math:`\alpha = \pi - \arcsin(2 - r_{min}/d_{obs})`.

    See Also
    --------
    :func:`los_angle2rmin` : Inverse — angle to impact parameter.
    :func:`thompson_sphere` : Uses the same LOS geometry.

    Examples
    --------
    At :math:`\alpha = 90°` the LOS grazes the plane of sky; the impact
    parameter equals the observer distance:

    >>> import numpy as np
    >>> float(los_rmin2angle(215.0, 215.0))
    90.0
    """
    # this ratio should be no more than 2 given how i've defined rmin
    ratio = np.clip(np.abs(rmin_rs/d_obs_rs), a_min=0.0, a_max=2.0)

    # invert the impact parameter calculation to get the angle (see get_los_rmin)
    mask = ratio > 1.0
    angle_deg = np.rad2deg(np.where(mask, np.pi - np.arcsin(2 - ratio, where=mask), np.arcsin(ratio, where=~mask)))

    # re-sign it so that this can work for 1D coordinate arrays in an approximate sense by passing
    # a negative value for rmin
    angle_deg = np.sign(rmin_rs)*angle_deg

    # catch a scalar having been passed and remove the array indices
    if np.isscalar(rmin_rs):
        angle_deg = angle_deg[()]

    return angle_deg


def los_angle2rmin(angle_deg: ArrayLike,
                   d_obs_rs: float) -> np.ndarray | float:
    r"""Convert a helioprojective angle to a LOS impact parameter :math:`r_{min}`.

    The inverse of :func:`los_rmin2angle`.  Given the helioprojective
    elongation angle :math:`\alpha` (the angle between Sun-center and the
    point of closest approach as seen from the observer), returns the
    corresponding impact parameter :math:`r_{min}` in :math:`R_\odot`.

    Parameters
    ----------
    angle_deg : ArrayLike
        Helioprojective elongation angle(s) in degrees.  Negative values are
        supported and propagate their sign to ``rmin_rs`` (useful for 1-D
        coordinate sweeps).
    d_obs_rs : float
        Observer distance from Sun-center in :math:`R_\odot`.

    Returns
    -------
    rmin_rs : np.ndarray | float
        Impact parameter(s) in :math:`R_\odot`.  Returns a scalar when
        ``angle_deg`` is scalar, otherwise an :class:`numpy.ndarray`.

    Notes
    -----
    The forward relation is:

    .. math::

       r_{min} =
       \begin{cases}
           d_{obs} \sin\alpha, & |\alpha| \leq 90° \\
           d_{obs}(2 - \sin\alpha), & |\alpha| > 90°
       \end{cases}

    ensuring a smooth, monotone mapping from :math:`0°` to :math:`180°`.

    See Also
    --------
    :func:`los_rmin2angle` : Inverse — impact parameter to angle.
    :func:`thompson_sphere` : Uses the same LOS geometry.

    Examples
    --------
    At :math:`\alpha = 90°` the LOS grazes the plane of sky; the impact
    parameter equals the observer distance:

    >>> float(los_angle2rmin(90.0, 215.0))
    215.0
    """
    # clip the angle to only be within -180 and 180 because of periodicity, then take the abs value
    angle_deg_clipped = np.abs(clip_angle(angle_deg, max_value=180.))

    # get the impact parameter, but smoothly pad with the reverse above 90 degrees
    sin_angle = np.sin(np.deg2rad(angle_deg_clipped))
    rmin_rs = d_obs_rs*np.where(angle_deg_clipped > 90.0, 2.0 - sin_angle, sin_angle)

    # account for the possibility that the angle could be negative and over 90 degrees in magnitude
    # so it can work for 1D coordinate arrays or ranges by passing a negative value for the angle
    rmin_rs = np.sign(angle_deg)*rmin_rs

    # catch a scalar having been passed and remove the array indices added by np.where
    if np.isscalar(angle_deg):
        rmin_rs = rmin_rs[()]

    return rmin_rs


def query_horizons_ephemeris(body,
                             time='now',
                             frame='carrington',
                             observer='self',
                             coord_system='lonlat',
                             **kwargs):
    r"""Query the position of a spacecraft or planet from JPL Horizons.

    Returns an :class:`astropy.table.QTable` containing timestamps and spatial
    coordinates in the requested frame and representation.  The table can be
    saved for offline use, e.g. ``table.write('coords.ecsv')``.

    This is a convenience wrapper around
    :func:`sunpy.coordinates.get_horizons_coord` that adds frame-string
    shortcuts, coordinate-system normalisation, and metadata columns.

    .. note::

       Requires an active internet connection and the ``astroquery`` package.

    Parameters
    ----------
    body : str | int
        JPL Horizons body identifier, e.g. ``'SolO'`` or ``-144`` for Solar
        Orbiter.  See https://ssd.jpl.nasa.gov/horizons/app.html for a
        full list.
    time : str | list | dict, optional
        Time query accepted by :func:`sunpy.coordinates.get_horizons_coord`.
        Can be an ISO string, an :class:`astropy.time.Time` array, or a
        Horizons-style time-range dict:

        .. code-block:: python

            time = {
                'start': '2024-03-29T00:00:00.000',
                'stop':  '2024-03-31T00:00:00.000',
                'step':  '1d',
            }

        When passing a manual time array the Horizons API imposes a length
        limit of roughly 100 entries.  Default is ``'now'``.
    frame : str | astropy frame, optional
        Output coordinate frame.  Supported string aliases:

        - ``'carrington'`` — Heliographic Carrington (rotating, light-travel
          corrected; requires ``observer``).
        - ``'hpc'`` — Helioprojective Cartesian (requires ``observer``).
        - ``'hci'`` — Heliocentric Inertial.
        - ``'gse'`` — Geocentric Solar Ecliptic.
        - ``'stonyhurst'`` — Heliographic Stonyhurst.

        Alternatively pass an :mod:`astropy` or :mod:`sunpy` frame object
        directly. Default is ``'carrington'``.
    observer : str | SkyCoord, optional
        Observer used when the frame is rotating or perspective-dependent
        (Carrington, HPC).  ``'self'`` uses the body's own location, which is
        appropriate for in-situ comparisons.  ``'earth'`` is more appropriate
        for remote-sensing contexts.  A :class:`astropy.coordinates.SkyCoord`
        with matching timestamps can be passed for a moving observer.
        Default is ``'self'``.
    coord_system : str, optional
        Representation for the three spatial columns:

        - ``'lonlat'`` — :math:`(r, \text{lat}, \text{lon})`, angles in
          degrees, :math:`r` in AU.
        - ``'rtp'`` — :math:`(r, \theta, \phi)`, angles in radians,
          :math:`r` in :math:`R_\odot` (PSI/MAS convention).
        - ``'xyz'`` — Cartesian :math:`(x, y, z)` in AU.
        - ``'hpc'`` — :math:`(r, T_x, T_y)` with :math:`T_x, T_y` in
          arcseconds (standard imager sky frame).

        Default is ``'lonlat'``.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`sunpy.coordinates.get_horizons_coord`.

    Returns
    -------
    table : astropy.table.QTable
        Table with four columns: ``time`` plus the three spatial coordinates,
        each stored as an :class:`astropy.units.Quantity`.  Query metadata
        (source, body, frame, query date) is stored in ``table.meta``.

    See Also
    --------
    :func:`spacecraft_trajectory` : Thin wrapper returning an ``(3, nt)``
        NumPy array in Carrington :math:`(r, \theta, \phi)`.
    :func:`sunpy.coordinates.get_horizons_coord` : Underlying SunPy function.

    Examples
    --------
    Parker Solar Probe locations in Carrington :math:`(r, \theta, \phi)` over
    a three-day window at two-hour cadence:

    .. code-block:: python

        time_query = {
            'start': '2024-03-28T00:00:00.000',
            'stop':  '2024-03-31T00:00:00.000',
            'step':  '2h',
        }
        table = query_horizons_ephemeris('PSP', time_query,
                                         frame='carrington',
                                         coord_system='rtp')

    Solar Orbiter as seen from Earth in helioprojective coordinates:

    .. code-block:: python

        table = query_horizons_ephemeris('SolO', '2024-04-08T19:00:00',
                                          frame='hpc', observer='earth',
                                          coord_system='hpc')
    """
    # get the ephemeris from JP horizons, this will return in stonyhurst coordinates
    coord_stony = sun_coord.get_horizons_coord(body=body, time=time, **kwargs)

    # setup frame conversion. Use the string code or assume an astropy frame was passed directly
    if isinstance(frame, str):
        match frame.lower():
            case 'carrington' | 'heliographic_carrington':
                # Setup the Carrington frame. Because Carrington is a rotating frame and the light
                # travel time is considered, the observer must be specified.
                frame = sun_coord.frames.HeliographicCarrington(observer=observer)
            case 'hpc' | 'helioprojective':
                frame = sun_coord.frames.Helioprojective(observer=observer)
            case 'hci' | 'heliocentricinertial':
                frame = sun_coord.frames.HeliocentricInertial()
            case 'gse' | 'geocentricsolarecliptic':
                frame = sun_coord.frames.GeocentricSolarEcliptic()
            case 'stonyhurst' | 'heliographic_stonyhurst':
                frame = coord_stony.frame

    # convert to the frame of interest
    coord = coord_stony.transform_to(frame)

    match coord_system.lower():
        case 'lonlat' | 'lonlatau':
            names = ['time', 'r', 'lat', 'lon']
            positions = coord.represent_as(astro_coord.SphericalRepresentation)
            columns = [coord.obstime, positions.distance, positions.lat.to(u.deg), positions.lon.to(u.deg)]
        case 'rtp' | 'spherical':
            names = ['time', 'r', 't', 'p']
            positions = coord.represent_as(astro_coord.PhysicsSphericalRepresentation)
            columns = [coord.obstime, positions.r.to(u.R_sun), positions.theta.to(u.rad), positions.phi.to(u.rad)]
        case 'cartesian' | 'xyz':
            names = ['time', 'x', 'y', 'z']
            positions = coord.represent_as(astro_coord.CartesianRepresentation)
            columns = [coord.obstime, positions.x, positions.y, positions.z]
        case 'arcsec' | 'sky' | 'hpc':
            names = ['time', 'r', 'Tx', 'Ty']
            positions = coord.represent_as(astro_coord.SphericalRepresentation)
            columns = [coord.obstime, positions.distance, positions.lon.to(u.arcsec), positions.lat.to(u.arcsec)]
        case _:
            msg = f"Unknown coordinate system: {coord_system}. Must be one of 'lonlat', 'rtp', 'xyz', or 'hpc'."
            raise ValueError(msg)

    # metadata for the query
    meta = {
        'source': 'jpl_horizons',
        'query_date': astro_time.Time.now().strftime(timestamp_format_ms),
        'body': body,
        'frame_name': frame.name,
    }
    # include the observer choice as metadata if its a carrington frame
    if frame.name == 'heliographic_carrington' or frame.name == 'helioprojective':
        meta['observer'] = observer

    # account for the case when only one scalar time was asked for
    if coord.shape == ():
        columns = [[columns[0]], [columns[1]], [columns[2]], [columns[3]]]

    # now build an astropy table (use q table so values are Quantities with units when accessed)
    table = QTable(columns, names=names, meta=meta)

    return table


def fibonacci_lattice(
    n: int = 100,
    radius: float = 1.0,
    randomize: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate an approximately uniform set of points on a sphere via a Fibonacci lattice.

    Places ``n`` points on a sphere using the golden-angle increment
    :math:`\Delta\phi = \pi(3 - \sqrt{5})` (the *Fibonacci sphere* or
    *sunflower* algorithm).  The result is a low-discrepancy, quasi-uniform
    distribution that is often preferred over random sampling for deterministic
    coverage.  Coordinates are returned in the PSI spherical convention.

    Parameters
    ----------
    n : int, optional
        Number of sample points to generate.  Default is ``100``.
    radius : float, optional
        Constant radial distance assigned to every point, in whatever units
        the caller uses (typically :math:`R_\odot`).  Default is ``1.0``.
    randomize : bool, optional
        If ``True``, apply a random phase offset to the golden-angle sequence
        so that repeated calls with the same ``n`` produce different rotations
        of the point set.  The quasi-uniform spacing is preserved.
        Default is ``False``.
    seed : int | None, optional
        Seed for :func:`numpy.random.default_rng`, used only when
        ``randomize=True``.  Pass an integer for reproducibility.
        Default is ``None``.

    Returns
    -------
    r : np.ndarray
        Radial coordinates of shape ``(n,)``, all equal to ``radius``.
    t : np.ndarray
        Colatitude :math:`\theta \in [0, \pi]` of shape ``(n,)``.
    p : np.ndarray
        Longitude :math:`\phi \in [0, 2\pi)` of shape ``(n,)``.

    Notes
    -----
    Points are first placed on the unit sphere via the golden-angle recurrence

    .. math::

       \phi_i = \left[(i + \delta) \bmod n\right] \cdot \pi(3 - \sqrt{5}),
       \qquad
       y_i = \frac{2i}{n} - 1 + \frac{1}{n}

    where :math:`\delta = 0` (deterministic) or a uniform random offset in
    :math:`[0, n)` when ``randomize=True``.  Colatitude and longitude follow
    from :math:`\theta_i = \arccos(z_i)` and
    :math:`\phi_i = \operatorname{atan2}(y_i, x_i) \bmod 2\pi`.

    See Also
    --------
    :func:`cartesian_pointmesh` : Local neighbourhood sampling around one or
        more spherical centers.

    Examples
    --------
    Generate 1 000 points on a sphere of radius :math:`2\,R_\odot`:

    >>> r, t, p = fibonacci_lattice(n=1000, radius=2.0, randomize=True, seed=0)
    >>> r.shape, t.shape, p.shape
    ((1000,), (1000,), (1000,))
    """
    # index array
    i = np.arange(n, dtype=np.float64)

    # random phase
    rnd = 1.0
    if randomize:
        rng = np.random.default_rng(seed)
        rnd = rng.random() * n

    offset = 2.0 / n
    increment = np.pi * (3.0 - np.sqrt(5.0))

    # y in [-1, 1] (approximately), and radial factor in xz-plane
    y = (i * offset - 1.0) + (offset / 2.0)
    r_xz = np.sqrt(np.maximum(0.0, 1.0 - y * y))  # guard tiny negatives

    # golden-angle increment around
    phi = np.mod(i + rnd, n) * increment

    x = np.cos(phi) * r_xz
    z = np.sin(phi) * r_xz

    # radius (should be 1 for unit sphere)
    r = np.full_like(x, radius)

    # theta (co-latitude)
    t = np.arccos(np.clip(z, -1.0, 1.0))

    # p computed as atan2(y, x), wrapped to [0, 2pi)
    p = np.mod(np.arctan2(y, x), 2.0 * np.pi)

    return r, t, p


def cartesian_pointmesh(
    r: ArrayLike,
    t: ArrayLike,
    p: ArrayLike,
    angular_radius: float = 1.0,
    pts_per_direction: int = 2,
    dimensionality: tuple[int, int, int] = (1, 1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate a local point mesh around one or many spherical centers.

    For each input center :math:`(r, \theta, \phi)`, builds a small normalised
    Cartesian grid in the local spherical orthonormal basis
    :math:`(\hat{e}_r, \hat{e}_\theta, \hat{e}_\phi)`, retains only offsets
    within the unit ball, scales them by the arc-length
    :math:`\mathrm{d}s = r\,\alpha_{\mathrm{rad}}`, and maps back to global
    spherical coordinates.  The result is a set of launch or sample points
    in a neighbourhood around each center direction.

    Coordinate conventions follow the PSI/physics colatitude standard used
    throughout this module: :math:`\theta \in [0, \pi]` is colatitude from
    :math:`+z` (solar north) and :math:`\phi \in [0, 2\pi)` is azimuth.
    Basis vectors :math:`(\hat{e}_r, \hat{e}_\theta, \hat{e}_\phi)` are
    the orthonormal spherical basis at each center, obtained via
    :func:`spherical_to_cartesian_vec`.

    Parameters
    ----------
    r : ArrayLike
        Radial distance of the center(s) in :math:`R_\odot`. Broadcast together
        with ``t`` and ``p`` following NumPy broadcasting rules.
    t : ArrayLike
        Colatitude :math:`\theta` of the center(s) in :math:`[0, \pi]`.
    p : ArrayLike
        Longitude :math:`\phi` of the center(s) in :math:`[0, 2\pi)`.
    angular_radius : float, optional
        Angular radius of the neighbourhood in **degrees**.  Converted internally
        to an arc-length scale :math:`\mathrm{d}s = r\,\alpha_{\mathrm{rad}}`.
        Default is ``1.0``.
    pts_per_direction : int, optional
        Half-resolution of the local grid.  Along each enabled basis direction a
        1-D grid with :math:`2 \times \texttt{pts\_per\_direction} + 1` samples
        spanning :math:`[-1, 1]` is created.  Default is ``2``.
    dimensionality : tuple[int, int, int], optional
        Flags ``(use_r, use_t, use_p)`` controlling which local basis directions
        are included in the grid:

        - ``(1, 1, 1)`` — full 3-D ball in :math:`(r, \theta, \phi)` directions.
        - ``(0, 1, 1)`` — 2-D tangential disc (no radial offset).
        - ``(1, 0, 0)`` — radial line only.

        Disabled directions contribute a single offset of zero.
        Default is ``(1, 1, 1)``.

    Returns
    -------
    launch_points : np.ndarray
        Spherical coordinates :math:`(r, \theta, \phi)` of the generated
        points, shape ``(3, npts, *center_shape)``.  ``npts`` is the number of
        offsets surviving the unit-ball mask and depends on ``pts_per_direction``
        and ``dimensionality``.
    rho : np.ndarray
        Normalised radial distances of each kept offset from the local origin,
        shape ``(npts,)``.  Identical for all centers; useful as a distance
        weight or mask.

    Notes
    -----
    The local grid is generated once for all centers.  Offsets with
    :math:`\rho \leq 1` (plus a small tolerance) are retained, then applied
    to every broadcasted center by projecting normalised offsets into Cartesian
    space via the local spherical basis and scaling by :math:`\mathrm{d}s`.

    For large ``angular_radius`` the small-angle arc-length approximation may
    introduce geometric distortion; prefer small values (a few degrees) for
    accurate local sampling.

    See Also
    --------
    :func:`fibonacci_lattice` : Global quasi-uniform sphere sampling.
    :func:`spherical_to_cartesian_vec` : Basis-vector rotation used internally.

    Examples
    --------
    Tangential neighbourhood around a single direction on the unit sphere:

    >>> import numpy as np
    >>> lp, rho = cartesian_pointmesh(1.0, np.pi / 2, 0.0, angular_radius=2.0,
    ...                               pts_per_direction=2, dimensionality=(0, 1, 1))
    >>> lp.shape[0]  # first axis is always 3 (r, t, p)
    3

    Vectorised neighbourhoods around 10 centers along a meridian:

    >>> r = np.ones(10)
    >>> t = np.linspace(0, np.pi, 10)
    >>> p = np.zeros(10)
    >>> lp, rho = cartesian_pointmesh(r, t, p, angular_radius=1.0)
    >>> lp.shape[0], lp.shape[2]  # (3, npts, 10)
    (3, 10)
    """
    ang_rad = astro_coord.Angle(angular_radius, unit=u.deg).rad

    r, t, p = np.broadcast_arrays(r, t, p)
    center_shape = r.shape

    # --- Build normalized local grid once ---
    ugrid = np.linspace(-1.0, 1.0, 2*pts_per_direction + 1)
    zero = np.array([0.0])

    ui, uj, uk = np.meshgrid(*(ugrid if dim else zero for dim in dimensionality), indexing="ij")

    rho = np.sqrt(ui*ui + uj*uj + uk*uk)
    tol = 1e-6
    mask = rho <= (1.0 + tol)

    # Flatten the kept normalized offsets
    ui = ui[mask]              # (npts,)
    uj = uj[mask]
    uk = uk[mask]
    rho = rho[mask]
    npts = ui.size

    # --- Flatten centers so we can do (npts, ncenters) math ---
    r_flat = r.reshape(-1)
    t_flat = t.reshape(-1)
    p_flat = p.reshape(-1)

    # Center Cartesian
    x0, y0, z0 = spherical_to_cartesian(r_flat, t_flat, p_flat)  # expect (ncenters,)

    # Local spherical basis at each center
    rhat_x, rhat_y, rhat_z = spherical_to_cartesian_vec(1.0, 0.0, 0.0, t_flat, p_flat)
    that_x, that_y, that_z = spherical_to_cartesian_vec(0.0, 1.0, 0.0, t_flat, p_flat)
    phat_x, phat_y, phat_z = spherical_to_cartesian_vec(0.0, 0.0, 1.0, t_flat, p_flat)

    # Per-center scale
    ds = r_flat * ang_rad  # (ncenters,)

    # Project normalized offsets into Cartesian, then scale by ds per center
    # (npts, 1) * (1, ncenters) -> (npts, ncenters)
    x_off_unit = (ui[:, None] * rhat_x[None, :] +
                  uj[:, None] * that_x[None, :] +
                  uk[:, None] * phat_x[None, :])
    y_off_unit = (ui[:, None] * rhat_y[None, :] +
                  uj[:, None] * that_y[None, :] +
                  uk[:, None] * phat_y[None, :])
    z_off_unit = (ui[:, None] * rhat_z[None, :] +
                  uj[:, None] * that_z[None, :] +
                  uk[:, None] * phat_z[None, :])

    x_pts = x0[None, :] + ds[None, :] * x_off_unit
    y_pts = y0[None, :] + ds[None, :] * y_off_unit
    z_pts = z0[None, :] + ds[None, :] * z_off_unit

    # Back to spherical
    r_pts, t_pts, p_pts = cartesian_to_spherical(x_pts, y_pts, z_pts)  # each (npts, ncenters)

    # Reshape back to (3, npts, *center_shape)
    launch_points = np.stack([r_pts, t_pts, p_pts])  # (3, npts, ncenters)
    launch_points = launch_points.reshape((3, npts, *center_shape))

    return launch_points, rho


def spacecraft_trajectory(
    body,
    t0,
    t1,
    step: str = "1h",
    **kwargs,
) -> np.ndarray:
    """Query a spacecraft/body ephemeris from JPL Horizons and return an RTP trajectory.

    This is a thin convenience wrapper around :func:`query_horizons_ephemeris` that
    requests a trajectory in the Carrington frame for the target ``body`` over the
    interval ``[t0, t1]`` sampled at a fixed cadence. The returned array contains
    the spherical coordinates ``(r, t, p)`` stacked along axis 0.

    Parameters
    ----------
    body : Any
        Target identifier understood by :func:`query_horizons_ephemeris` / Horizons
        (e.g., a Horizons body ID, spacecraft name).
    t0, t1 : Any
        Start and stop times for the query. Types must be accepted by
        :func:`query_horizons_ephemeris` (commonly ISO strings, `datetime`, or
        `astropy.time.Time`).
    step : str, optional
        Sampling cadence passed to Horizons, typically in a Horizons-style string
        such as ``"1h"``, ``"30m"``, ``"1d"``.
    **kwargs
        Additional keyword arguments forwarded to :func:`query_horizons_ephemeris`

    Returns
    -------
    trajectory : np.ndarray
        Array of shape ``(3, nt)`` where ``nt`` is the number of returned samples.
        ``trajectory[0]`` is ``r``, ``trajectory[1]`` is ``t`` (theta/colatitude),
        and ``trajectory[2]`` is ``p`` (phi/azimuth), matching ``coord_system="rtp"``.

    Notes
    -----
    - This function hard-codes the ephemeris query options:

      - ``frame="carrington"``
      - ``observer="self"``
      - ``coord_system="rtp"``

      If you need different frames/observers/coordinates, call
      :func:`query_horizons_ephemeris` directly.

    - Units are whatever :func:`query_horizons_ephemeris` returns in its ``rtp``
      columns; this function strips units via ``.value`` before stacking. If you
      want to preserve units, return the columns directly or stack `Quantity`
      arrays instead.

    Examples
    --------
    Query PSP over two days at one-hour cadence (requires internet):

    .. code-block:: python

        traj = spacecraft_trajectory("psp", "2024-01-01", "2024-01-03", step="1h")
        print(traj.shape)   # (3, nt) where nt depends on the cadence
    """
    lps = query_horizons_ephemeris(body,
                                   time={'start': t0, 'stop': t1, 'step': step},
                                   frame='carrington',
                                   observer='self',
                                   coord_system='rtp',
                                   **kwargs)
    ephemeris = tuple(lps[col].value for col in 'rtp')
    return np.stack(ephemeris, axis=0)


def _norm(x: ArrayLike, eps: float = 1e-12):
    """Normalise a vector, returning ``None`` for near-zero inputs.

    Parameters
    ----------
    x : ArrayLike
        Input vector of arbitrary length.
    eps : float, optional
        Vectors with :func:`numpy.linalg.norm` below this threshold are
        treated as zero.  Default is ``1e-12``.

    Returns
    -------
    out : np.ndarray | None
        Unit vector with the same shape as ``x``, or ``None`` if ``x`` is
        effectively zero.
    """
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if n < eps:
        return None
    return x / n

def camera_roll_wrt_solar_north(
    position: tuple[float, float, float],
    focal_point: tuple[float, float, float],
    view_up: tuple[float, float, float],
    world_up: tuple[float, float, float] = SOLAR_NORTH,
    degrees: bool = True,) -> float:
    r"""Compute the camera roll angle relative to a world "up” direction.

    The roll is the signed rotation **about the view axis**
    (from ``position`` toward ``focal_point``) that would bring the projection
    of ``world_up`` onto the image plane into alignment with the projection of
    ``view_up``.  By default ``world_up`` is solar north
    :math:`(0, 0, 1)` (:data:`pyvisual.core.constants.SOLAR_NORTH`).

    Parameters
    ----------
    position : tuple[float, float, float]
        Camera position in world (Cartesian) coordinates,
        :math:`(x_c, y_c, z_c)`.
    focal_point : tuple[float, float, float]
        Camera look-at target in world coordinates.
    view_up : tuple[float, float, float]
        Camera "up” vector in world coordinates.  Need not be perfectly
        orthogonal to the view direction; it is projected into the view plane
        internally.
    world_up : tuple[float, float, float], optional
        Reference "up” direction in world coordinates.  Projected into the
        view plane before computing the roll.
        Default is :data:`~pyvisual.core.constants.SOLAR_NORTH`.
    degrees : bool, optional
        If ``True`` return the roll in degrees, otherwise in radians.
        Default is ``True``.

    Returns
    -------
    roll : float
        Signed roll angle about the view axis.  Positive sense follows the
        right-hand rule about
        :math:`\hat{v} = (\mathbf{f} - \mathbf{p}) / \|\mathbf{f} - \mathbf{p}\|`.
        Returns :data:`numpy.nan` for degenerate cases (see Notes).

    Notes
    -----
    Algorithm:

    1. Compute the normalised view direction
       :math:`\hat{v} = \operatorname{normalize}(\mathbf{f} - \mathbf{p})`.
    2. Project ``view_up`` (:math:`\hat{u}`) and ``world_up`` (:math:`\hat{w}`)
       onto the image plane (perpendicular to :math:`\hat{v}`):

       .. math::

          \mathbf{u}_\perp = \hat{u} - (\hat{u} \cdot \hat{v})\,\hat{v},
          \qquad
          \mathbf{w}_\perp = \hat{w} - (\hat{w} \cdot \hat{v})\,\hat{v}

    3. Compute the signed angle from :math:`\mathbf{w}_\perp` to
       :math:`\mathbf{u}_\perp` about :math:`\hat{v}`:

       .. math::

          \alpha = \arctan2\!\left(
              \hat{v} \cdot (\mathbf{w}_\perp \times \mathbf{u}_\perp),\;
              \mathbf{w}_\perp \cdot \mathbf{u}_\perp
          \right)

    The angle is undefined (returns :data:`numpy.nan`) when the projection of
    ``world_up`` or ``view_up`` onto the view plane has near-zero magnitude —
    most commonly when the view direction is parallel or anti-parallel to
    ``world_up`` (a gimbal-lock-like condition).

    See Also
    --------
    :meth:`pyvisual.core.mixins.ObserverMixin.observer_orientation` : Property
        that stores the result of this function for the active camera.

    Examples
    --------
    Observer on the :math:`+x` axis looking at the Sun with solar north as
    ``view_up`` — roll should be zero:

    >>> import numpy as np
    >>> roll = camera_roll_wrt_solar_north(
    ...     position=(10, 0, 0),
    ...     focal_point=(0, 0, 0),
    ...     view_up=(0, 0, 1),
    ... )
    >>> np.isfinite(roll)
    True
    """
    pos = np.asarray(position, dtype=float)
    foc = np.array(focal_point, dtype=float)
    u = np.array(view_up, dtype=float)
    w = np.array(world_up, dtype=float)

    v = _norm(foc - pos)          # view direction
    if v is None:
        return np.nan

    u = _norm(u)
    w = _norm(w)
    if u is None or w is None:
        return np.nan

    # project onto plane ⟂ v
    u_perp = u - np.dot(u, v) * v
    w_perp = w - np.dot(w, v) * v

    u_perp = _norm(u_perp)
    w_perp = _norm(w_perp)
    if u_perp is None or w_perp is None:
        # view direction parallel to world_up (or view_up): roll is not defined
        return np.nan

    # signed angle from w_perp -> u_perp about axis v
    # angle = atan2( v · (w×u), w · u )
    sin_term = np.dot(v, np.cross(w_perp, u_perp))
    cos_term = np.dot(w_perp, u_perp)
    ang = np.arctan2(sin_term, cos_term)

    return np.degrees(ang) if degrees else ang
