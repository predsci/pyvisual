"""Pooch-based fetching and caching of PSI example datasets and package assets.

This module manages the downloading, verification, and local caching of HDF5
data files used in the pyvisual example gallery, and provides access to
bundled package assets (e.g. the PyVista color theme).

The example datasets are MHD output files from a PSI Thermo 2 steady-state
simulation for Carrington Rotation 2282 (CR 2282), hosted at
``https://www.predsci.com/doc/assets/``.  Both coronal (``cor``) and
heliospheric (``hel``) domain files are available, covering quantities such
as the magnetic field components :math:`(B_r, B_\\theta, B_\\phi)`, velocity
:math:`(v_r, v_\\theta, v_\\phi)`, density :math:`\\rho`, temperature :math:`T`,
and current density :math:`(j_r, j_\\theta, j_\\phi)`.

File integrity is verified on download via SHA256 hashes stored in
:data:`REGISTRY`.  The local cache location defaults to the OS-standard user
cache directory under a ``psi/`` subdirectory, and can be overridden by
setting the ``PYVISUAL_CACHE`` environment variable (see :data:`FETCHER`).
"""

from __future__ import annotations
from collections import namedtuple
from itertools import product
from pathlib import Path
from typing import Iterable

try:
    import pooch
except ImportError as e:
    raise ImportError(
        "Missing the optional 'pooch' dependency required for data fetching. "
        "Please install it via pip or conda to access the necessary datasets."
    ) from e

REGISTRY = {
    "cr2282-thermo2-cor/vr002.h5": "sha256:5dc25daff38a6e3663e3ad9c0889fb7bcc54e849a6ab98b0d0403cd927c93dd7",
    "cr2282-thermo2-cor/bp002.h5": "sha256:7e72a478554a8bd8b685d0081be988af4fecb58740a50a83a63a4ad1c02378a9",
    "cr2282-thermo2-cor/vt002.h5": "sha256:85311c63f6c0d1c14fccb70419a700a2346cd84084501a0c8f53f89a65b267b4",
    "cr2282-thermo2-cor/heat002.h5": "sha256:ab38dbbbf696724452d95f0a72cc46f4e6ed624e9fdde6bf44912ce2825acb01",
    "cr2282-thermo2-cor/rho002.h5": "sha256:ee7f72ac2ef8d40bd12fedee5627151c68619474b8e2ed2621e3c749f364da85",
    "cr2282-thermo2-cor/ep002.h5": "sha256:8ebfd3d4e1d6f5c656f5410c59bc630f62fe9a28b90996abd367e596c5ff6159",
    "cr2282-thermo2-cor/t002.h5": "sha256:b039c91af46b05382f459e5371de991c3fe8d05a52b4435ef05254ff6a49b3eb",
    "cr2282-thermo2-cor/em002.h5": "sha256:f715d598748f47d07e64abde7d0f502f121b919ceaf23cab0caa3a75782ca55f",
    "cr2282-thermo2-cor/br002.h5": "sha256:e37cd27a2b8d8953bbf604cec7a47c2d4d1dd85a6c941a4b6073547b1c797f12",
    "cr2282-thermo2-cor/vp002.h5": "sha256:ff77f21d58362d84c3bd59817c10fc461853277b9ab10e88796b8d98d0e38314",
    "cr2282-thermo2-cor/jp002.h5": "sha256:3eecab4b8d75b0b3f93fa0c477bca07985d1b92becb58fec88a061217b3508e9",
    "cr2282-thermo2-cor/bt002.h5": "sha256:d2339e4d57a5bbb66b3f3a08267e262c0810eab9fd167a8f169e3726f3ba8ce2",
    "cr2282-thermo2-cor/jt002.h5": "sha256:e779a1a24f8f785d020b203dab20085f98f7e74562eba887305724e52b2fc94e",
    "cr2282-thermo2-cor/ch_map002.h5": "sha256:ff066d9ae173436581e5d9c90f199a9996010189d4e3e15586dfc4c9f82eb8e5",
    "cr2282-thermo2-cor/jr002.h5": "sha256:cc10a1d639182c55b0224d678e7de5275c9415c8b2b183a2d6dc8fc70fbb1f41",
    "cr2282-thermo2-hel/vr002.h5": "sha256:ccfa6d497d20b7240b207819d6d3ff17e338fdc55f8ace68941dd8489bb74def",
    "cr2282-thermo2-hel/bp002.h5": "sha256:4e81c6afb5b12c866188113a02dfabcb8b1508f6434ba2f3bb4f24374c0ae110",
    "cr2282-thermo2-hel/vt002.h5": "sha256:cdbf5789fe9ccd0931a479e7a4a8f9af106a3869069cc725cd42ebdf3fb7165f",
    "cr2282-thermo2-hel/rho002.h5": "sha256:6834c5dc33a20a88111a6bc0e67a15fa1484d099241a38badbbe4f77f77dc73b",
    "cr2282-thermo2-hel/t002.h5": "sha256:6408c340047decb582dbd42702ea2a008688d999fa3c7b361889b9bac0d7d258",
    "cr2282-thermo2-hel/br002.h5": "sha256:2203cd2cc4540f6b3a0ccaf361bf1c91c6dc93756de05c2f7ad6639ffc8f42d8",
    "cr2282-thermo2-hel/vp002.h5": "sha256:916b8e0d045f16920d54419a855f2bef816b1bfabaf3f08dce8a260c10386827",
    "cr2282-thermo2-hel/jp002.h5": "sha256:762bcf4b09a2c3f43ff02b6c475a84c1e80396f57d649056f65ece0e323e7d80",
    "cr2282-thermo2-hel/bt002.h5": "sha256:04f805b4a76def07687d7473062b84452e7ace61b842e7fae23effe8851533e9",
    "cr2282-thermo2-hel/jt002.h5": "sha256:818e677d8e1c3500d08dce05be94fb2c5ae10e7694c694b37dad917dc3409037",
    "cr2282-thermo2-hel/jr002.h5": "sha256:834c1d615349f343f8d94f37780485c41b9cdad82466b062e1d6395e85690bcc",
}
"""Registry of available CORHEL data files with corresponding SHA256 hashes.

This registry is used by the pooch fetcher to verify the integrity of
downloaded files, and is primarily intended for building sphinx-gallery
examples that require MHD data files.

The files listed here correspond to a Thermo 2 steady-state run for
Carrington Rotation 2282 – both the coronal and heliospheric domains.
"""

BASE_URL = "https://www.predsci.com/doc/assets/"
"""Base URL for the PSI documentation asset server hosting example data files."""

FETCHER = pooch.create(
    path=pooch.os_cache("psi"),
    base_url=BASE_URL,
    registry=REGISTRY,
    env="PYVISUAL_CACHE",
)
"""Pooch fetcher for downloading and caching magnetic field files.

.. note::
    The cache directory can be overridden by setting the ``PYVISUAL_CACHE``
    environment variable to a desired path. Otherwise, the default cache
    directory is platform-dependent, as determined by :func:`pooch.os_cache`.

.. note::
    The default (os-dependent) cache directory stores assets under a
    subdirectory named ``psi``. The reason for this naming choice – as opposed
    to ``pyvisual`` – is to maintain consistency with other PredSci packages
    that utilize the same asset hosting and caching mechanism.
"""

def fetch_datasets(domains: Iterable[str] | str = 'cor',
                   variables: Iterable[str] | str = 'br') -> namedtuple:
    """Download (or load from cache) one or more CR 2282 CORHEL datasets.

    A thin convenience wrapper around :data:`FETCHER` that fetches the
    requested combination of domain(s) and variable(s) and returns their
    local file paths as a :class:`~collections.namedtuple`.  Intended
    primarily for Sphinx-gallery examples and tutorials that need a small,
    version-pinned set of PSI MHD data files.

    Each field of the returned namedtuple is named ``"{domain}_{variable}"``
    (e.g. ``cor_br``, ``hel_vr``).

    Parameters
    ----------
    domains : Iterable[str] | str, optional
        One or more domain identifiers:

        - ``'cor'`` — coronal domain (:math:`1\\text{–}30\\,R_\\odot`).
        - ``'hel'`` — heliospheric domain (:math:`30\\text{–}230\\,R_\\odot`).

        A bare string is treated as a one-element list.  Default is ``'cor'``.
    variables : Iterable[str] | str, optional
        One or more MHD variable identifiers (without the ``002`` suffix).
        Available quantities include:

        - Magnetic field: ``'br'``, ``'bt'``, ``'bp'``
          (:math:`B_r, B_\\theta, B_\\phi`)
        - Velocity: ``'vr'``, ``'vt'``, ``'vp'``
          (:math:`v_r, v_\\theta, v_\\phi`)
        - Thermodynamic: ``'rho'`` (:math:`\\rho`), ``'t'`` (:math:`T`),
          ``'ep'``, ``'em'``, ``'heat'``
        - Current density: ``'jr'``, ``'jt'``, ``'jp'``
          (:math:`j_r, j_\\theta, j_\\phi`)
        - Coronal-hole map: ``'ch_map'`` (coronal domain only)

        A bare string is treated as a one-element list.  Default is ``'br'``.

    Returns
    -------
    filepaths : namedtuple
        A :func:`~collections.namedtuple` whose fields are named
        ``"{domain}_{variable}"`` for each requested ``(domain, variable)``
        pair.  Each value is the local filesystem path (:class:`str`) to the
        fetched HDF5 file.  Field order matches
        ``itertools.product(domains, variables)`` — domains vary slowest.

    Raises
    ------
    ValueError
        If a ``(domain, variable)`` combination is not present in
        :data:`REGISTRY`, or produces an invalid Python identifier.
    RequestException
        If a download is required and the network request fails.
    HashMismatchError
        If a downloaded file's SHA256 hash does not match :data:`REGISTRY`.
    FileNotFoundError
        If the file is absent from the cache and cannot be downloaded.

    Notes
    -----
    Files are fetched from URLs of the form:

    .. code-block:: text

        https://www.predsci.com/doc/assets/cr2282-thermo2-{domain}/{variable}002.h5

    and verified against :data:`REGISTRY` on every call.  The cache location
    is determined by :data:`FETCHER` and can be overridden with the
    ``PYVISUAL_CACHE`` environment variable.

    See Also
    --------
    :data:`REGISTRY` : Complete list of available files and their SHA256 hashes.
    :data:`FETCHER` : Underlying :class:`pooch.Pooch` instance.

    Examples
    --------
    Fetch the radial magnetic field for the coronal domain:

    >>> fp = fetch_datasets("cor", "br")
    >>> fp.cor_br  # doctest: +SKIP
    '/.../cr2282-thermo2-cor/br002.h5'

    Fetch multiple variables from both domains simultaneously:

    >>> fp = fetch_datasets(domains=("cor", "hel"), variables=("br", "bt"))
    >>> fp.cor_br, fp.cor_bt, fp.hel_br, fp.hel_bt  # doctest: +SKIP
    (..., ..., ..., ...)
    """
    if isinstance(domains, str):
        domains = [domains]
    if isinstance(variables, str):
        variables = [variables]
    req_pairs = tuple(product(domains, variables))
    Filepaths = namedtuple("Filepaths", [f"{dom}_{var}" for dom, var in req_pairs])
    return Filepaths(*(FETCHER.fetch(f"cr2282-thermo2-{dom}/{var}002.h5") for dom, var in req_pairs))


def fetch_theme():
    """Return the path to the bundled PyVisual PyVista theme file.

    Resolves ``pyvisual_theme.json`` from the package's ``_assets`` directory
    and returns its absolute :class:`~pathlib.Path`.  This file is loaded
    automatically at import time by :mod:`pyvisual.core` to apply the default
    PyVista color theme.

    Returns
    -------
    filepath : pathlib.Path
        Absolute path to ``pyvisual_theme.json``.

    Raises
    ------
    FileNotFoundError
        If ``pyvisual_theme.json`` is not found at the expected location.
        This typically means the file was not included in the installation
        (e.g. a partial checkout without Git LFS assets).

    Notes
    -----
    The path is resolved as:

    .. code-block:: python

        core_dir = Path(__file__).resolve().parent.parent
        filepath = core_dir / "_assets" / "pyvisual_theme.json"

    Examples
    --------
    >>> theme_path = fetch_theme()
    >>> theme_path.name
    'pyvisual_theme.json'
    """
    core_dir = Path(__file__).resolve().parent.parent
    filepath = core_dir / '_assets' / 'pyvisual_theme.json'
    if not filepath.exists():
        raise FileNotFoundError(f"Theme file `pyvisual_theme.json` not found. "
                                f"Please ensure the file exists in the `pyvisual/assets/templates` directory, "
                                f"or run git-lfs pull to fetch the file from the repository.")
    return filepath