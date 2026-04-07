"""
TODO add description
"""

from pyvisual import Plot3d
from pyvisual.core.mesh3d import SphericalMesh
from pyvisual.utils.data import fetch_datasets
from psi_io import read_hdf_data

data = fetch_datasets("cor", ["br", "ch_map"])
mesh1 = SphericalMesh(data.cor_br, iformat='rtp')
mesh2 = SphericalMesh(data.cor_ch_map, iformat='pt', r=3)

# chmap, p, t = read_hdf_data(data.cor_ch_map)


plotter = Plot3d()
plotter.add_mesh(mesh1, cmap='seismic', clim=(-1e-1, 1e-1), show_scalar_bar=False, opacity=0.5)
plotter.add_mesh(mesh2, cmap='seismic', clim=(-1e-1, 1e-1), show_scalar_bar=False, opacity=1)
plotter.show()