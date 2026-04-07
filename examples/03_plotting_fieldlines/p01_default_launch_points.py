"""
TODO add description
"""


from pyvisual import Plot3d
from pyvisual.utils.data import fetch_datasets
from psi_io import np_interpolate_slice_from_hdf
from mapflpy.tracer import Tracer

data = fetch_datasets("cor", ["br", "bt", "bp"])
br, *scales = np_interpolate_slice_from_hdf(data.cor_br, 1, None, None)
br_t, br_p = scales

tracer = Tracer(*data)
fieldlines = tracer.trace_fwd()

plotter = Plot3d()

plotter.add_2d_slice(1, br_t, br_p, br,
                     clim=(-1, 1),
                     cmap="seismic",
                     show_scalar_bar=False)

plotter.add_fieldlines(fieldlines.geometry,
                       coloring='random',
                       cmap='hsv',
                       name='fieldlines')
plotter.show()
