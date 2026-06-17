"""
Physical constants and coordinate-frame alias system for **pyvisual**.

This module defines the two canonical coordinate-frame names used throughout the
library — ``'spherical'`` and ``'cartesian'`` — together with the alias system that
maps the many user-facing spelling variants (e.g. ``'rtp'``, ``'polar'``, ``'cart'``,
any permutation of the axis letters) to one of those two canonical strings.

Frame alias system
------------------
Every function or class that accepts a ``frame`` argument calls
:func:`~pyvisual.core.parsers.fetch_canonical_frame`, which normalises the input
string and looks it up in :data:`FRAME_ALIASES`.  :data:`FRAMES` stores the inverse
mapping (canonical name → full set of accepted aliases), and :data:`FRAME_SCALES`
maps each canonical name to its ordered axis-letter key (used when constructing
:class:`~pyvisual.core.mesh3d.SphericalMesh` and
:class:`~pyvisual.core.mesh3d.CartesianMesh` from raw coordinate arrays).

Notes
-----
``RTP_PERMUTATIONS`` and ``XYZ_PERMUTATIONS`` are generated at import time and
included in the corresponding alias sets so that single-letter substrings (``'r'``,
``'x'``, ``'tr'``, etc.) are also valid aliases.
"""

from __future__ import annotations

from itertools import permutations
from types import MappingProxyType

import numpy as np

SOLAR_NORTH = np.array((0, 0, 1))
r"""Canonical solar-north unit vector in Cartesian coordinates :math:`(x, y, z)`.

Solar north is fixed at :math:`+\hat{z}` throughout the library.  It is used as
the default camera up-vector and as the reference direction for position-angle
calculations.

Examples
--------
>>> from pyvisual.core.constants import SOLAR_NORTH
>>> SOLAR_NORTH
array([0, 0, 1])
"""

TWOPI = 2 * np.pi
r"""Convenience constant :math:`2\pi`.

Examples
--------
>>> from pyvisual.core.constants import TWOPI
>>> import numpy as np
>>> np.isclose(TWOPI, 2 * np.pi)
True
"""

timestamp_format_ms = "%Y-%m-%dT%H:%M:%S.%f"
"""ISO 8601 timestamp format string with microsecond precision.

Used when parsing or formatting date-time strings associated with
in-situ spacecraft data or model output metadata.

Examples
--------
>>> from pyvisual.core.constants import timestamp_format_ms
>>> from datetime import datetime
>>> datetime.strptime('2023-01-15T12:00:00.000000', timestamp_format_ms)
datetime.datetime(2023, 1, 15, 12, 0)
"""

RTP_PERMUTATIONS = {
	"".join(perm) for i in range(1, len("rtp") + 1) for perm in permutations("rtp", i)
}
"""All non-empty substrings formed by permuting the letters ``r``, ``t``, ``p``.

These strings are added to the ``'spherical'`` alias set so that single-axis labels
(``'r'``, ``'t'``, ``'p'``) and two-axis labels (``'rt'``, ``'pr'``, etc.) are
recognised as spherical-frame aliases.
"""

XYZ_PERMUTATIONS = {
	"".join(perm) for i in range(1, len("xyz") + 1) for perm in permutations("xyz", i)
}
"""All non-empty substrings formed by permuting the letters ``x``, ``y``, ``z``.

Analogous to :data:`RTP_PERMUTATIONS` for the Cartesian frame.
"""

FRAMES = MappingProxyType(
	{
		"cartesian": {"xyz", "cartesian", "rectilinear"} | XYZ_PERMUTATIONS,
		"spherical": {"rtp", "spherical", "psi", "polar"} | RTP_PERMUTATIONS,
	}
)
"""Immutable mapping from canonical frame name to the full set of accepted aliases.

Keys are ``'cartesian'`` and ``'spherical'``.  Values are :class:`frozenset`-like
sets of normalised alias strings.

See Also
--------
:data:`FRAME_ALIASES` : Flat alias → canonical lookup derived from this mapping.
"""

FRAME_ALIASES = {alias: canonical for canonical, aliases in FRAMES.items() for alias in aliases}
"""Flat mapping from every accepted frame alias to its canonical name.

Built automatically from :data:`FRAMES` at import time.  Used by
:func:`~pyvisual.core.parsers.fetch_canonical_frame` to resolve user-supplied
``frame`` arguments.

Examples
--------
>>> from pyvisual.core.constants import FRAME_ALIASES
>>> FRAME_ALIASES['polar']
'spherical'
>>> FRAME_ALIASES['xyz']
'cartesian'
"""

FRAME_SCALES = {
	"cartesian": "xyz",
	"spherical": "rtp",
}
"""Mapping from canonical frame name to its ordered axis-letter key.

The axis-letter key is the string used as the default ``iformat`` when constructing
:class:`~pyvisual.core.mesh3d.SphericalMesh` (``'rtp'``) or
:class:`~pyvisual.core.mesh3d.CartesianMesh` (``'xyz'``) from raw arrays, and when
decoding the order in which coordinate arrays should be passed.

Examples
--------
>>> from pyvisual.core.constants import FRAME_SCALES
>>> FRAME_SCALES['spherical']
'rtp'
"""
