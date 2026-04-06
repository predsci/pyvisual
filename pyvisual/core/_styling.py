"""
Immutable default render-kwargs dictionaries for each **pyvisual** render type.

Every constant in this module is a :class:`~types.MappingProxyType` (read-only
view of a :class:`dict`), preventing accidental mutation of package-level defaults
at runtime.  Caller-supplied keyword arguments are merged on top of these defaults
using the ``|`` operator, so any key can be overridden without modifying the
originals.

The constants fall into two groups:

**Coloring defaults** — merged before geometry kwargs when data arrays are present
or absent:

- :data:`SOLID_COLOR_KWARGS` — used when *no* scalar data is supplied (solid color).
- :data:`COLORMAP_KWARGS` — used when scalar data *is* supplied (colormap render).

**Geometry defaults** — merged after the coloring defaults to fix render-style
flags that depend on the geometry type (points, splines, slices):

- :data:`PLOT1D_KWARGS`, :data:`PLOT2D_KWARGS`
- :data:`POINTS_KWARGS`, :data:`SPLINES_KWARGS`, :data:`SLICES_KWARGS`

**Fieldline-specific defaults**:

- :data:`FIELDLINE_KWARGS` — baseline colormap limits for fieldline scalars.
- :data:`FL_POLARITY_COLORING_DEFAULTS` — five-category polarity colormap.
- :data:`RANDOM_COLORING_DEFAULTS` — random per-fieldline hue assignment.
"""

from types import MappingProxyType


FL_STATE_ANNOTATIONS = MappingProxyType({
    -1.5: "Open (Br-)",
    -0.75: "Closed",
    0: "Error",
    0.75: "Disconnect",
    1.5: "Open (Br+)"
})
"""Scalar-bar annotations for the five magnetic fieldline polarity states.

Maps the numeric scalar value written on each fieldline to a human-readable
polarity label.  Used by :data:`FL_POLARITY_COLORING_DEFAULTS` to annotate the
scalar bar when rendering with ``coloring='polarity'``.

Keys are ``float`` sentinel values; values are display strings.
"""

SOLID_COLOR_KWARGS = MappingProxyType(dict(
    cmap=None,
    rgb=False
))
"""Base PyVista kwargs applied when rendering geometry without scalar data.

Disables the colormap (``cmap=None``) and multi-channel RGB mode (``rgb=False``),
so that the ``color`` kwarg (or PyVista's default) controls the mesh appearance.
"""

COLORMAP_KWARGS = MappingProxyType(dict(
    color=None,
    rgb=False
))
"""Base PyVista kwargs applied when rendering geometry *with* scalar data.

Clears any fixed ``color`` so that the colormap drives the appearance, and
disables RGB mode (``rgb=False``).
"""

FIELDLINE_KWARGS = MappingProxyType(dict(
    n_colors=5,
    clim=(-2, 2)
))
"""Default colormap discretisation and limits for fieldline scalar data.

``n_colors=5`` matches the five polarity categories; ``clim=(-2, 2)`` spans the
signed sentinel range used by :data:`FL_STATE_ANNOTATIONS`.
"""

FL_POLARITY_COLORING_DEFAULTS = MappingProxyType(dict(
    cmap=['blue', 'grey', 'black', 'green', 'red'],
    annotations=dict(FL_STATE_ANNOTATIONS),
    scalar_bar_args=dict(n_labels=0, label_font_size=11)
))
"""Full PyVista kwargs for polarity-colored fieldline rendering.

Provides a five-color discrete colormap aligned to the signed polarity sentinel
values, along with annotated scalar-bar configuration.  Applied when
``coloring='polarity'`` is passed to
:meth:`~pyvisual.core.mixins.StackMeshMixin.add_fieldlines`.
"""

RANDOM_COLORING_DEFAULTS = MappingProxyType(dict(
    cmap='hsv',
    show_scalar_bar=False,
))
"""PyVista kwargs for randomly hue-assigned fieldline rendering.

Each fieldline receives a distinct hue sampled from the circular ``'hsv'`` colormap.
The scalar bar is suppressed because the per-line integer index has no physical
meaning.  Applied when ``coloring='random'`` is passed to
:meth:`~pyvisual.core.mixins.StackMeshMixin.add_fieldlines`.
"""

PLOT1D_KWARGS = MappingProxyType(dict(
    render_lines_as_tubes=True,
    render_points_as_spheres=False,
    style='surface',
))
"""Default render-style kwargs for 1-D line geometry (tube rendering)."""

PLOT2D_KWARGS = MappingProxyType(dict(
    render_lines_as_tubes=False,
    render_points_as_spheres=False,
    style='surface',
))
"""Default render-style kwargs for 2-D surface geometry (flat surface rendering)."""

POINTS_KWARGS = MappingProxyType(dict(
    render_lines_as_tubes=False,
    render_points_as_spheres=True,
    style='points',
))
"""Default render-style kwargs for point-cloud geometry (sphere glyphs at each point)."""

SPLINES_KWARGS = MappingProxyType(dict(
    render_lines_as_tubes=True,
    render_points_as_spheres=False,
    style='surface',
))
"""Default render-style kwargs for spline/line geometry (tube rendering)."""

SLICES_KWARGS = MappingProxyType(dict(
    render_lines_as_tubes=False,
    render_points_as_spheres=False,
    style='surface',
))
"""Default render-style kwargs for 2-D slice geometry (flat surface, no tube/sphere glyphs)."""