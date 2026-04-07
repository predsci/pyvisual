"""
TODO add description
"""

from pyvisual import Plot3d
from pyvisual.utils.data import fetch_datasets
from psi_io import read_hdf_by_index

data = fetch_datasets("cor", ["br", "ch_map"])
br_rslice, *scales_rslice = read_hdf_by_index(data.cor_br, 50, None, None)
br_tslice, *scales_tslice = read_hdf_by_index(data.cor_br, None, 50, None)
br_pslice, *scales_pslice = read_hdf_by_index(data.cor_br, None, None, 50)

kwargs = dict(clim=(-1e-1, 1e-1), cmap="seismic", show_scalar_bar=False)

plotter = Plot3d()
plotter.add_sun()

plotter.add_2d_slice(*scales_rslice, br_rslice, **kwargs)
plotter.add_2d_slice(*scales_tslice, br_tslice, **kwargs)
plotter.add_2d_slice(*scales_pslice, br_pslice, **kwargs)
plotter.show()
