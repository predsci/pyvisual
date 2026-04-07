"""
TODO add description
"""


import numpy as np

from pyvisual import Plot3d
from pyvisual.utils.data import fetch_datasets
from psi_io import np_interpolate_slice_from_hdf
from mapflpy.tracer import Tracer
from mapflpy.utils import get_fieldline_polarity

data = fetch_datasets("cor", ["br", "bt", "bp"])
br, *scales = np_interpolate_slice_from_hdf(data.cor_br, 1, None, None)
br_t, br_p = scales

tracer = Tracer(*data)
fieldlines = tracer.trace_fwd(r=1., n=256)
fl_polarity = get_fieldline_polarity(1, 30, data.cor_br, fieldlines)

plotter = Plot3d()

plotter.add_2d_slice(1, br_t, br_p, br,
                     clim=(-1, 1),
                     cmap="seismic",
                     show_scalar_bar=False)

plotter.add_fieldlines(*np.swapaxes(fieldlines.geometry, 0, 1), fl_polarity,
                       coloring='polarity',
                       name='fieldlines')
plotter.show()
