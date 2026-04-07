"""
TODO add description
"""


import numpy as np
from sunpy.coordinates.sun import carrington_rotation_time
from psi_io import np_interpolate_slice_from_hdf
from mapflpy.scripts import inter_domain_tracing
from mapflpy.utils import combine_and_pad_fieldlines

from pyvisual import Plot3d
from pyvisual.utils.data import fetch_datasets
from pyvisual.utils.geometry import spacecraft_trajectory

if __name__ == "__main__":
    data = fetch_datasets(["cor", "hel"], ["br", "bt", "bp"])
    br, *scales = np_interpolate_slice_from_hdf(data.cor_br, 1, None, None)
    br_t, br_p = scales

    t0, t1 = carrington_rotation_time([2282, 2283])
    trajectory = spacecraft_trajectory("psp", t0, t1)

    fieldlines, *_ = inter_domain_tracing(data.cor_br, data.cor_bt, data.cor_bp,
                         data.hel_br, data.hel_bt, data.hel_bp,
                                      launch_points=trajectory)
    nan_padded_fl = combine_and_pad_fieldlines(fieldlines)

    plotter = Plot3d()

    plotter.add_points(trajectory, point_size=5)

    plotter.add_2d_slice(1, br_t, br_p, br,
                         clim=(-1, 1),
                         cmap="seismic",
                         show_scalar_bar=False)

    plotter.add_2d_slice(30, br_t, br_p, br,
                         clim=(-1, 1),
                         cmap="seismic",
                         opacity=0.2,
                         show_scalar_bar=False)

    plotter.add_fieldlines(nan_padded_fl,
                           coloring='random',
                           cmap='hsv',
                           name='fieldlines')
    plotter.show()