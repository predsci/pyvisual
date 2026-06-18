# noqa: INP001
"""
MHDweb Integration Part I
=========================

This is the first of a two-part series demonstrating how to use the
`MHDweb REST API <https://predsci.com/mhdweb2/api>`_ to retrieve
Predictive Science MAS model output and spacecraft connectivity data.

`MHDweb <https://predsci.com/mhdweb2>`_ provides programmatic retrieval
of MAS run HDF5 data files, querying of MHDweb's *MAS Run Database* /
*Spacecraft Database*, and generation of various data products.

.. attention::
   Access requires a PSI-issued API key (passed as ``Authorization: Api-Key <key>`` in the
   header of every request).

   **To request a key, visit the link:** `MHDweb API Key <https://predsci.com/mhdweb2/api>`_.


Three endpoints are used here:

- ``mas-run-db`` — searches the MAS run catalog by time, model, and variable.
- ``mas-run-db/{id}/{domain}/{state}/{variable}`` — streams a ZIP archive of
  HDF5 field files for the requested domain and state.
- ``spacecraft-mapping/{id}/{sc_id}`` — returns the magnetic connectivity
  mapping for a named spacecraft, serialized as an
  `Astropy ECSV <https://docs.astropy.org/en/stable/io/ascii/ecsv.html>`_
  byte stream.

The downloaded files are consumed in
:ref:`sphx_glr_gallery_99_advanced_plots_p11_integrating_mhdweb_p2.py`.

.. note::

   This example requires a valid ``API_KEY`` environment variable (or a
   ``.env`` file in the working directory). Run the script locally with
   your own key to reproduce the downloads.

   **To safeguard your API key, do not commit it to version control or share
   it publicly. Use environment variables or secure vaults to manage your credentials.**

.. seealso::

   :ref:`sphx_glr_gallery_99_advanced_plots_p11_integrating_mhdweb_p2.py`
      Part II — loads the files downloaded here, traces magnetic
      connectivity, and produces four visualizations.
   `MHDweb References <https://predsci.com/mhdweb2/references>`_
      A collections of resources for learning more about MHDweb, the MAS model, and related topics.
"""

# sphinx_gallery_thumbnail_path = '_static/psi_logo.png'

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from pprint import pprint

import requests
from astropy.table import Table

# %%
# Configure Output Paths and Authentication
# -----------------------------------------
#
# ``STATIC_ASSETS`` is an optional environment variable that redirects output
# to a shared asset directory used by the Sphinx pre-build pipeline.  When
# unset, files are written to the current working directory.
#
# Authentication follows the `MHDweb API key scheme
# <https://predsci.com/mhdweb2/api>`_: the key is passed as
# ``Authorization: Api-Key <key>`` on every request.

OUTPUT_DIR = Path(os.environ.get("STATIC_ASSETS", "")).resolve()
COR_OUTPUT_DIR = OUTPUT_DIR / "cor_mag_field"
HEL_OUTPUT_DIR = OUTPUT_DIR / "hel_mag_field"
BASE_URL = "https://www.predsci.com/mhdweb2_bu/v2/api"
API_KEY = os.environ.get("API_KEY")
AUTH = {"Authorization": f"Api-Key {API_KEY}"}

if not COR_OUTPUT_DIR.exists():
	COR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not HEL_OUTPUT_DIR.exists():
	HEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Query the MAS Run Database
# --------------------------
#
# The ``mas-run-db`` endpoint searches the MAS run catalog and returns a list
# of runs ranked by proximity to the requested time of interest (``toi``).
# Key query parameters:
#
# - ``toi`` — ISO-8601 timestamp; the API selects all runs with a run time
#   range encompassing the requested time and ranks them by proximity to it.
# - ``model`` — MAS model variant *e.g.* ``'thermo_2'``, ``'thermo_1'``, ``'poly'``.
# - ``type`` — run type; ``'ss'`` is a steady-state solution.
# - ``domain`` — list of domains to require: ``'cor'`` (corona,
#   :math:`1\text{–}30\,R_\odot`) and ``'hel'`` (heliosphere, extending
#   to :math:`\sim 1\,\mathrm{AU}`).
# - ``variables`` — field components needed for tracing:
#   :math:`(B_r, B_\theta, B_\phi)`.
#
# The first result is taken; its ``id`` field (``cor_id``) is the run
# identifier used in all subsequent requests.

db_query_params = {
	"toi": "2024-05-09T12:00:00",
	"model": "thermo_2",
	"type": "ss",
	"domain": ["cor", "hel"],
	"variables": ["br", "bt", "bp"],
}

db_response = requests.get(
	f"{BASE_URL}/mas-run-db",
	headers={"Accept": "application/json"} | AUTH,
	params=db_query_params,
	timeout=(3, 10),
)
db_response.raise_for_status()

runs = db_response.json()
pprint(runs)		# noqa: T203

try:
	run = runs[0]
except IndexError as e:
	msg = "No results found in MAS Run DB for the specified parameters."
	raise IndexError(msg) from e

cor_id = run["id"]

# %%
# Inspect Run Metadata
# --------------------
#
# Fetching ``mas-run-db/{id}`` returns a metadata record for the run,
# including per-domain ``states`` lists.  For steady-state runs there is
# a single state (index ``0``); time-dependent runs have multiple states,
# each corresponding to one simulation snapshot.

dbmeta_response = requests.get(
	f"{BASE_URL}/mas-run-db/{cor_id}",
	headers={"Accept": "application/json"} | AUTH,
	timeout=(3, 10),
)
dbmeta_response.raise_for_status()

run_meta = dbmeta_response.json()
pprint(run_meta["cor"])		# noqa: T203
len(run_meta["cor"]["states"])

pprint(run_meta["hel"])		# noqa: T203
len(run_meta["hel"]["states"])

print(len(run_meta["omas"]))		# noqa: T201

# %%
# Download Magnetic Field Files
# ------------------------------
#
# The endpoint ``mas-run-db/{cor_id}/{domain}/{state}/{variable}`` streams a
# ZIP archive containing HDF5 files for the requested field components.
# ``state='0'`` selects the first (and for steady-state runs, only) snapshot.
# Requesting ``variable='br,bt,bp'`` as a comma-separated string fetches all
# three components in a single archive.
#
# The coronal and heliospheric archives are downloaded and written separately
# since they will be extracted to different directories in
# :ref:`sphx_glr_gallery_99_advanced_plots_p11_integrating_mhdweb_p2.py`.

cor_files_params = {"cor_id": str(cor_id), "domain": "cor", "state": "0", "variable": "br,bt,bp"}

print("Fetching coronal magnetic field files...")		# noqa: T201
cor_files_response = requests.get(
	f"{BASE_URL}/mas-run-db/" + "/".join(cor_files_params.values()),
	headers=AUTH,
	stream=True,
	timeout=(3, 10),
)
cor_files_response.raise_for_status()

print("Saving coronal magnetic field files...")		# noqa: T201
with Path(COR_OUTPUT_DIR / "cor_mag_field.zip").open("wb") as f:
	f.writelines(cor_files_response.iter_content(chunk_size=8192))

hel_files_params = {"cor_id": str(cor_id), "domain": "hel", "state": "0", "variable": "br,bt,bp"}

print("Fetching heliospheric magnetic field files...")		# noqa: T201
hel_files_response = requests.get(
	f"{BASE_URL}/mas-run-db/" + "/".join(hel_files_params.values()),
	headers=AUTH,
	stream=True,
	timeout=(3, 10),
)
hel_files_response.raise_for_status()

print("Saving heliospheric magnetic field files...")		# noqa: T201
with Path(HEL_OUTPUT_DIR / "hel_mag_field.zip").open("wb") as f:
	f.writelines(hel_files_response.iter_content(chunk_size=8192))

# %%
# Download the Spacecraft Mapping
# --------------------------------
#
# The ``spacecraft-mapping/{cor_id}/{sc_id}`` endpoint returns the magnetic
# connectivity mapping for a named spacecraft over the run's time range.
# Here ``sc_id='solo'`` selects Solar Orbiter.
#
# The response is an `Astropy ECSV
# <https://docs.astropy.org/en/stable/io/ascii/ecsv.html>`_ byte stream.
# Reading it into an :class:`astropy.table.Table` via :class:`~io.BytesIO` preserves
# column units and metadata without writing a temporary file. In this case – for
# continuity with :ref:`sphx_glr_gallery_99_advanced_plots_p11_integrating_mhdweb_p2.py` –
# the table is written to disk as an ECSV file for use in Part II.
#
# Each row of the table corresponds to one time step and contains three sets
# of :math:`(r, \theta, \phi)` coordinates (in :math:`R_\odot` and radians):
#
# - ``sc_pos_{r,t,p}`` — actual spacecraft position in the Carrington frame.
# - ``r1_pos_{r,t,p}`` — position ballistically mapped inward to the outer
#   coronal boundary (:math:`r_1 \approx 30\,R_\odot`).
# - ``r0_pos_{r,t,p}`` — magnetic footpoint traced to the inner boundary
#   (:math:`r_0 = 1\,R_\odot`).

sc_id = "solo"

response = requests.get(
	f"{BASE_URL}/spacecraft-mapping/{cor_id}/{sc_id}",
	headers=AUTH,
	stream=True,
	timeout=(3, 10)
)
response.raise_for_status()

spacecraft_mapping = Table.read(BytesIO(response.content), format="ascii.ecsv")
print(spacecraft_mapping.info(out=None))		# noqa: T201

spacecraft_mapping.write(
	OUTPUT_DIR / "spacecraft_mapping.ecsv", format="ascii.ecsv", overwrite=True
)
