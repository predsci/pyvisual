"""
TODO add description
"""

from mapflpy.tracer import Tracer

from pyvisual import Plot3d
from pyvisual.utils.data import fetch_datasets
from psi_io import read_hdf_data

data_files = fetch_datasets("cor", ["br", "bt", "bp", "ch_map"])

tracer = Tracer(data_files.cor_br, data_files.cor_bt, data_files.cor_bp)
fieldlines = tracer.trace_fwd(r=1., n=256)


chmap, *chmap_scales = read_hdf_data(data_files.cor_ch_map)
br, *br_scales = read_hdf_data(data_files.cor_br)

plotter = Plot3d()

plotter.add_longitudinal_lines()
plotter.add_latitudinal_lines()

plotter.add_2d_slice(1, *reversed(chmap_scales), chmap.T,
                     cmap="seismic", show_scalar_bar=False)
plotter.add_contour(*br_scales, br,
                    opacity=0.5, color='white', show_scalar_bar=False)
plotter.add_fieldlines(fieldlines.geometry,
                       line_width=2,
                       coloring='random', cmap='hsv', name='fieldlines')
plotter.show()