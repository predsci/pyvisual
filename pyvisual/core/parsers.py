r"""
Input-normalisation utilities and the ``@parse_mesh_params`` decorator.

This module provides the low-level API that translates user-supplied coordinate arrays,
frame strings, and mesh objects into the internal representations expected by the
mesh and mixin layers.

.. note::
   This module is principally concerned with parsing and validating coordinate
   transformations for spherical meshes. Although a certain (limited) set of additional
   coordinate frames will be supported in future iterations of **pyvisual**, the primary
   concern of this package is to work with PSI's native spherical-coordinate output.

   For those interested in working with a broader range of coordinate systems, it is
   recommended to consult SunPy's `sunkit-pyvista <https://docs.sunpy.org/projects/sunkit-pyvista/en/latest/>`_
   package, which provides a more general interface to PyVista with support for SunPy's
   coordinate frames and transformations.

Key responsibilities
--------------------
- **Frame resolution** — :func:`_normalize_frame` / :func:`fetch_canonical_frame`
  normalise any accepted alias to ``'spherical'`` or ``'cartesian'``.
- **Coordinate parsing** — :func:`parse_stack_mesh` and :func:`parse_grid_mesh`
  validate and optionally broadcast :math:`(r, \theta, \phi)` arrays to a common
  shape.
- **Data alignment** — :func:`parse_data` reshapes scalar arrays to match a target
  mesh shape, handling transposition, broadcasting, and flattening.
- **Mesh validation & transform** — :func:`validate_mesh_type` wraps arbitrary
  objects in a PyVista type; :func:`apply_mesh_transform` applies the appropriate
  coordinate-conversion function chain when the mesh frame differs from the plotter
  frame.
- **Decorator** — :func:`parse_mesh_params` wraps mixin methods so that scalar
  coordinate arguments are promoted to 1-D arrays via
  :func:`~pyvisual.utils.helpers.atleast_1dnull` before the method body runs.

See Also
--------
:mod:`pyvisual.core.constants`
    :data:`~pyvisual.core.constants.FRAME_ALIASES` and
    :data:`~pyvisual.core.constants.FRAME_SCALES` used here.
:func:`pyvisual.utils.helpers.atleast_1dnull`
    None-safe wrapper around :func:`numpy.atleast_1d`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce, wraps
from math import prod
from typing import Literal

import numpy as np
import pyvista as pv
from pyvista import algorithm_to_mesh_handler

from pyvisual.core._typing import PlottableType
from pyvisual.core.constants import FRAME_ALIASES
from pyvisual.utils.geometry import (
	cartesian_to_spherical,
	ij_meshgrid,
	moveaxis_to_start,
	spherical_to_cartesian,
)
from pyvisual.utils.helpers import atleast_1dnull

_FRAME_TRANSFORM_MAPPING = {
	("spherical", "cartesian"): spherical_to_cartesian,
	("cartesian", "spherical"): cartesian_to_spherical,
}


def _align_shape(values, mesh_shape):
	"""Broadcast a squeezed array to ``mesh_shape`` by matching axis sizes.

	Each axis of ``values`` is matched to the unique unmatched axis of
	``mesh_shape`` with the same size.  After all axes are matched the array is
	reshaped with size-1 dimensions inserted for unmatched mesh axes, then
	broadcast to the full ``mesh_shape``.

	Parameters
	----------
	values : np.ndarray
		The array to align.  Should already be squeezed (no size-1 dimensions
		unless they are genuinely degenerate).
	mesh_shape : tuple[int, ...]
		Target shape to align ``values`` against.

	Returns
	-------
	out : np.ndarray
		A view of ``values`` broadcast to ``mesh_shape``.

	Raises
	------
	ValueError
		If an axis of ``values`` has a size that does not match any remaining
		unmatched axis of ``mesh_shape``.

	Examples
	--------
	>>> import numpy as np
	>>> from pyvisual.core.parsers import _align_shape
	>>> v = np.ones((3, 5))
	>>> _align_shape(v, (3, 7, 5)).shape
	(3, 7, 5)
	"""
	mesh_shape = tuple(mesh_shape)
	val_shape = values.shape

	mesh_axes = []
	for size in val_shape:
		matches = [i for i, s in enumerate(mesh_shape) if s == size and i not in mesh_axes]
		if not matches:
			available = [mesh_shape[i] for i in range(len(mesh_shape)) if i not in mesh_axes]
			msg = (
				f"Cannot align values with shape {val_shape} to mesh shape {mesh_shape}: "
				f"values axis {len(mesh_axes)} has size {size}, but no unmatched mesh "
				f"axis has that size (remaining mesh axis sizes: {available})"
			)
			raise ValueError(msg)
		mesh_axes.append(matches[0])

	order = np.argsort(mesh_axes)
	values = values.transpose(order)
	sorted_axes = sorted(mesh_axes)

	new_shape = []
	val_iter = iter(values.shape)
	for i in range(len(mesh_shape)):
		new_shape.append(next(val_iter) if i in sorted_axes else 1)
	values = values.reshape(new_shape)

	return np.broadcast_to(values, mesh_shape)


def _reordered_ravel(
	arr: np.ndarray, axis: int | None, order: Literal["K", "A", "C", "F"] | None
) -> np.ndarray:
	"""Flatten ``arr``, optionally moving ``axis`` to the leading position first.

	Parameters
	----------
	arr : np.ndarray
	    Array to flatten.
	axis : int | None
	    If not ``None``, move this axis to position 0 before ravelling so that
	    elements along ``axis`` vary fastest in the output.
	order : {"K", "A", "C", "F"} | None
	    Memory layout order passed to :func:`numpy.ndarray.ravel`.

	Returns
	-------
	out : np.ndarray
	    1-D flattened array.
	"""
	if axis is None:
		return arr.ravel(order=order)
	return moveaxis_to_start(arr, axis).ravel(order=order)


def _normalize_frame(frame: str) -> str | None:
	"""Normalise a raw frame string to a consistent lowercase key.

	Strips whitespace, converts to lower-case, and replaces hyphens and spaces
	with underscores so that the result can be looked up in
	:data:`~pyvisual.core.constants.FRAME_ALIASES`.

	Parameters
	----------
	frame : str
	    Raw user-supplied frame string (e.g. ``'RTP'``, ``'Cartesian'``,
	    ``'spherical-psi'``).

	Returns
	-------
	out : str
	    Normalised frame string.
	"""
	return str(frame).strip().lower().replace("-", "_").replace(" ", "_")


def fetch_canonical_frame(frame: str) -> str:
	"""Return the canonical frame name for ``frame``, or ``None`` if unrecognised.

	Normalises ``frame`` via :func:`_normalize_frame` and looks it up in
	:data:`~pyvisual.core.constants.FRAME_ALIASES`.

	Parameters
	----------
	frame : str
	    Any accepted frame alias (e.g. ``'rtp'``, ``'polar'``, ``'xyz'``,
	    ``'cartesian'``).

	Returns
	-------
	out : str | None
	    The canonical name (``'spherical'`` or ``'cartesian'``), or ``None``
	    if the alias is not recognised.

	Examples
	--------
	>>> from pyvisual.core.parsers import fetch_canonical_frame
	>>> fetch_canonical_frame('polar')
	'spherical'
	>>> fetch_canonical_frame('XYZ')
	'cartesian'
	"""
	return FRAME_ALIASES.get(_normalize_frame(frame), None)


def parse_mesh_params(func: Callable):
	"""Promote positional coordinate arguments to 1-D arrays.

	Wraps each positional argument of the decorated method with
	:func:`~pyvisual.utils.helpers.atleast_1dnull`, converting scalar values
	to length-1 arrays while leaving ``None`` sentinels intact.

	Apply this decorator to any mixin method whose first positional arguments
	are ``r``, ``t``, ``p`` coordinate arrays so that downstream parsing code can
	always assume array inputs.

	Parameters
	----------
	func : Callable
	    The method to wrap.  Its positional arguments (after ``self``) will be
	    promoted.

	Returns
	-------
	out : Callable
	    The wrapped method with the same signature as ``func``.

	Examples
	--------
	>>> from pyvisual.core.parsers import parse_mesh_params
	>>> class Foo:
	...     @parse_mesh_params
	...     def bar(self, r, t, p):
	...         return r.ndim, t.ndim, p.ndim
	>>> Foo().bar(1.0, 2.0, 3.0)
	(1, 1, 1)
	"""

	@wraps(func)
	def decorator(self, *args, **kwargs):
		return func(self, *atleast_1dnull(*args), **kwargs)

	return decorator


def generate_transforms(mframe: str, pframe: str) -> list[Callable]:
	"""Return a list of coordinate-conversion functions from ``mframe`` to ``pframe``.

	Looks up the ``(canonical_mframe, canonical_pframe)`` pair in
	``_FRAME_TRANSFORM_MAPPING``.  Returns an empty list when no conversion is
	needed (same frame) or the pair is unrecognised.

	Parameters
	----------
	mframe : str
	    Coordinate frame of the source mesh — any alias accepted by
	    :func:`fetch_canonical_frame`.
	pframe : str
	    Target coordinate frame — any alias accepted by
	    :func:`fetch_canonical_frame`.

	Returns
	-------
	out : list[Callable]
	    Ordered list of transform callables.  Each callable accepts unpacked
	    coordinate arrays and returns a tuple of transformed arrays.  Pass
	    through :func:`functools.reduce` to apply the full chain.

	Examples
	--------
	>>> from pyvisual.core.parsers import generate_transforms
	>>> transforms = generate_transforms('spherical', 'cartesian')
	>>> len(transforms)
	1
	"""
	transforms = []
	frame_key = (fetch_canonical_frame(mframe), fetch_canonical_frame(pframe))
	if ftrans := _FRAME_TRANSFORM_MAPPING.get(frame_key):
		transforms.append(ftrans)
	return transforms


def validate_mesh_type(mesh: PlottableType) -> PlottableType:
	"""Ensure ``mesh`` is a valid PyVista dataset, wrapping it if necessary.

	Resolves any VTK algorithm objects via
	:func:`~pyvista.algorithm_to_mesh_handler`, then attempts to wrap non-PyVista
	objects with :func:`pyvista.wrap`.

	Parameters
	----------
	mesh : PlottableType
	    Any object accepted by PyVista's plotter — a :class:`pyvista.DataSet`,
	    a VTK algorithm, a NumPy array, etc.

	Returns
	-------
	out : pyvista.DataSet
	    A valid PyVista dataset.

	Raises
	------
	TypeError
	    If ``mesh`` cannot be converted to a PyVista dataset.
	"""
	mesh, *_ = algorithm_to_mesh_handler(mesh)
	if not pv.is_pyvista_dataset(mesh):
		mesh = pv.wrap(mesh) if mesh is not None else pv.PolyData()
		if not pv.is_pyvista_dataset(mesh):
			msg = f"Object type ({type(mesh)}) not supported for plotting in PyVista."
			raise TypeError(msg)
	return mesh


def apply_mesh_transform(mesh: PlottableType, mframe: str, pframe: str) -> PlottableType:
	"""Convert mesh point coordinates from ``mframe`` to ``pframe`` in-place on a copy.

	If the mesh stores a ``'MESH_FRAME'`` key in :attr:`pyvista.DataSet.user_dict`
	that value takes precedence over the ``mframe`` argument.  Composite types
	(:class:`pyvista.MultiBlock`, :class:`pyvista.PartitionedDataSet`) are handled
	recursively — each block is transformed individually.

	Grid meshes (:class:`pyvista.Grid` subclasses) are cast to a
	:class:`pyvista.StructuredGrid` before the point coordinates are modified,
	because :class:`pyvista.RectilinearGrid` stores its axes separately and cannot be
	mutated in-place.

	Parameters
	----------
	mesh : PlottableType
	    Source mesh to transform.
	mframe : str
	    Coordinate frame of ``mesh`` — any alias accepted by
	    :func:`fetch_canonical_frame`.  Overridden by ``mesh.user_dict['MESH_FRAME']``
	    when present.
	pframe : str
	    Target coordinate frame for the plotter.

	Returns
	-------
	out : pyvista.DataSet
	    A copy of ``mesh`` with points expressed in ``pframe`` coordinates.  Returns
	    ``mesh`` unchanged if the frames are identical or no transform is registered.
	"""
	mframe = mesh.user_dict.get("MESH_FRAME", mframe)
	match mesh:
		case pv.PartitionedDataSet():
			mesh = apply_mesh_transform(mesh.cast_to_multiblock(), mframe, pframe)
		case pv.MultiBlock():
			mesh = mesh.as_polydata_blocks(copy=True)
			for i, m in enumerate(mesh):
				mesh[i] = apply_mesh_transform(m, mframe, pframe)
	if not mesh.is_empty and mframe and (transforms := generate_transforms(mframe, pframe)):
		match mesh:
			case pv.Grid():
				mesh = mesh.cast_to_structured_grid()
			case pv.DataSet():
				mesh = mesh.copy()
		points = reduce(lambda v, f: f(*v), transforms, mesh.points.T)
		mesh.points = np.column_stack(points)
	return mesh


def parse_stack_mesh(
	r: np.ndarray, t: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Validate that three coordinate arrays share an identical shape.

	Used by :class:`~pyvisual.core.mixins.StackMeshMixin` methods to confirm that
	``r``, ``t``, ``p`` describe a single set of :math:`N`-D points (rather than
	independent 1-D axes).

	Parameters
	----------
	r : np.ndarray
	    Radial coordinate array.
	t : np.ndarray
	    Colatitude coordinate array.
	p : np.ndarray
	    Longitude coordinate array.

	Returns
	-------
	out : tuple[np.ndarray, np.ndarray, np.ndarray]
	    The input arrays unchanged as a 3-tuple ``(r, t, p)``.

	Raises
	------
	ValueError
	    If ``r``, ``t``, and ``p`` do not all have the same shape.
	"""
	if r.shape == t.shape == p.shape:
		return r, t, p
	msg = (
		f"All scales must be ND (with identical shapes), "
		f"got r: shape={r.shape}, t: shape={t.shape}, p: shape={p.shape}."
	)
	raise ValueError(msg)


def parse_grid_mesh(
	r: np.ndarray, t: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Validate and optionally broadcast coordinate arrays for a structured grid.

	Accepts two input conventions:

	- **1-D axes** — ``r``, ``t``, ``p`` are each 1-D arrays of independent axis
	  coordinates.  They are broadcast to a full 3-D meshgrid via
	  :func:`~pyvisual.utils.geometry.ij_meshgrid`.
	- **3-D full grid** — ``r``, ``t``, ``p`` are pre-broadcast 3-D arrays with
	  identical shape.  Returned unchanged.

	Parameters
	----------
	r : np.ndarray
	    Radial coordinate array — either 1-D axis values or a 3-D meshgrid.
	t : np.ndarray
	    Colatitude coordinate array.
	p : np.ndarray
	    Longitude coordinate array.

	Returns
	-------
	out : tuple[np.ndarray, np.ndarray, np.ndarray]
	    Fully broadcast 3-D ``(r, t, p)`` arrays.

	Raises
	------
	ValueError
	    If the arrays are neither all 1-D nor all 3-D with matching shapes.
	"""
	if 1 == r.ndim == t.ndim == p.ndim:
		return ij_meshgrid(r, t, p)
	if r.ndim == 3 and r.shape == t.shape == p.shape:
		return r, t, p
	msg = (
		f"All scales must be 1D (with arbitrary lengths), or 3D (with identical shapes), "
		f"got r: shape={r.shape}, t: shape={t.shape}, p: shape={p.shape}."
	)
	raise ValueError(msg)


def parse_data(
	data: np.ndarray,
	mesh_shape: tuple[int, ...],
	axis: int | None = None,
	order: Literal["K", "A", "C", "F"] | None = "C",
) -> np.ndarray:
	"""Reshape and flatten a scalar data array to match a target mesh shape.

	Handles five cases in order:

	1. ``data.shape == mesh_shape`` — ravel directly (optionally reordering ``axis``
	   to the leading position first).
	2. ``data.shape == mesh_shape[::-1]`` — transpose then ravel (Fortran vs C
	   storage order from the file reader).
	3. ``data.size == 1`` — broadcast to a constant array of length
	   ``prod(mesh_shape)``.
	4. ``data`` can be aligned to ``mesh_shape`` by axis-size matching after
	   squeezing — handled by :func:`_align_shape`.
	5. ``data.ndim > len(mesh_shape)`` — raises :class:`ValueError`.

	Parameters
	----------
	data : np.ndarray
	    Scalar field array read from a PSI HDF file or provided by the user.
	mesh_shape : tuple[int, ...]
	    Expected shape of the target mesh (e.g. ``(nr, nt, np)``).
	axis : int | None, optional
	    If not ``None``, move this axis to the leading position before ravelling,
	    so that the per-slice ordering matches the polydata connectivity.
	    Default is ``None``.
	order : {"K", "A", "C", "F"} | None, optional
	    Memory layout order for :func:`numpy.ndarray.ravel`.  Default is ``'C'``.

	Returns
	-------
	out : np.ndarray
	    1-D array of length ``prod(mesh_shape)`` ready to assign as a point-data
	    scalar array on a PyVista mesh.

	Raises
	------
	ValueError
	    If ``data.ndim > len(mesh_shape)``.

	Examples
	--------
	>>> import numpy as np
	>>> from pyvisual.core.parsers import parse_data
	>>> data = np.arange(6).reshape(2, 3)
	>>> parse_data(data, (2, 3)).shape
	(6,)
	"""
	mesh_ndim = len(mesh_shape)
	mesh_size = prod(mesh_shape)

	if data.ndim > mesh_ndim:
		msg = f"Values has {data.ndim} dimensions but mesh has only {mesh_ndim}."
		raise ValueError(msg)

	if data.shape == mesh_shape:
		return _reordered_ravel(data, axis, order)

	if data.shape == mesh_shape[::-1]:
		return _reordered_ravel(data.T, axis, order)

	if data.size == 1:
		return np.full(mesh_size, data.flat[0], dtype=data.dtype)

	values_ = _align_shape(np.squeeze(data), mesh_shape)
	return _reordered_ravel(values_, axis, order)
