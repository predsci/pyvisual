"""General-purpose helper utilities for pyvisual internals.

This module collects small, reusable helpers that do not belong to any single
subsystem.  The functions here are intentionally minimal and depend only on
the standard library and :mod:`numpy` — they exist to reduce boilerplate in
the core parsing and mesh-construction layers (see :mod:`pyvisual.core.parsers`
and :mod:`pyvisual.core.mesh3d`), and to resolve bundled package assets such as
the PyVista colour theme (see :func:`fetch_theme`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def atleast_1dnull(*args,
                   astuple: bool = False) -> np.ndarray | None | tuple[np.ndarray | None, ...]:
    r""":func:`numpy.atleast_1d` that passes ``None`` values through unchanged.

    A thin wrapper around :func:`numpy.atleast_1d` that treats ``None`` as a
    sentinel meaning "not provided" rather than raising or converting it.
    This is useful when optional coordinate arrays or scale factors may
    legitimately be absent, as is common in PSI mesh-parsing routines where
    individual :math:`(r, \theta, \phi)` coordinate arrays may be omitted.

    Parameters
    ----------
    *args : ArrayLike | None
        One or more values to promote to at least 1-D.  Any element that is
        ``None`` is returned as ``None`` without modification.
    astuple : bool, optional
        When ``False`` (default) and exactly one positional argument is given,
        return that single result directly (not wrapped in a :class:`tuple`).
        When ``True``, or when more than one argument is supplied, always
        return a :class:`tuple`.

    Returns
    -------
    out : np.ndarray | None | tuple[np.ndarray | None, ...]
        Promoted array(s), with ``None`` entries preserved.  The return type
        mirrors ``astuple``: a bare value for a single argument with
        ``astuple=False``, otherwise a :class:`tuple`.

    See Also
    --------
    :func:`numpy.atleast_1d` : The underlying NumPy function.

    Examples
    --------
    Promote a scalar to a 1-D array:

    >>> atleast_1dnull(1.0)
    array([1.])

    ``None`` is passed through without modification:

    >>> atleast_1dnull(None) is None
    True

    Multiple arguments always return a :class:`tuple`, with ``None`` preserved:

    >>> atleast_1dnull(1.0, None, astuple=True)
    (array([1.]), None)
    """
    if len(args) == 1 and not astuple:
        return np.atleast_1d(args[0]) if args[0] is not None else None
    return tuple(np.atleast_1d(arg) if arg is not None else None for arg in args)


def fetch_theme() -> Path:
    """Return the path to the bundled PyVisual PyVista theme file.

    Resolves ``pyvisual_theme.json`` from the package's ``_assets`` directory
    and returns its absolute :class:`~pathlib.Path`.  This file is loaded
    automatically at import time by :mod:`pyvisual.core` to apply the default
    PyVista colour theme.

    Returns
    -------
    filepath : pathlib.Path
        Absolute path to ``pyvisual_theme.json``.

    Raises
    ------
    FileNotFoundError
        If ``pyvisual_theme.json`` is not found at the expected location.
        This typically means the file was not included in the installation.

    Notes
    -----
    The path is resolved relative to this module's parent package directory:

    .. code-block:: python

        pkg_dir = Path(__file__).resolve().parent.parent
        filepath = pkg_dir / "_assets" / "pyvisual_theme.json"

    See Also
    --------
    :data:`pyvista.global_theme` : The PyVista theme this file configures.

    Examples
    --------
    >>> theme_path = fetch_theme()
    >>> theme_path.name
    'pyvisual_theme.json'
    """
    pkg_dir = Path(__file__).resolve().parent.parent
    filepath = pkg_dir / "_assets" / "pyvisual_theme.json"
    if not filepath.exists():
        msg = "Theme file `pyvisual_theme.json` not found."
        raise FileNotFoundError(msg)
    return filepath
