"""
This example shows how to create a simple scene with a sun and display it.
"""

from math import pi

from pyvisual import Plot3d
from psi_io import np_interpolate_slice_from_hdf
import pyvista as pv

plotter = Plot3d()
plotter.add_sun()
# x = plotter.add_shell(opacity=0.2, inner_radius=1.1, outer_radius=3, color='red')
# plotter.add_disc(normal=(0,1,0), inner_radius=1, outer_radius=2, color='white')
sphere = pv.SolidSphere(outer_radius=0.00917, center=(215, 0, 0))
plotter.camera.focal_point = (215,0,0)

plotter.add_mesh(sphere, color='blue')
plotter.show()
# observer_position = plotter.observer_position
# plotter.add_point(observer_position)
# # plotter.add_thompson_sphere(opacity=0.2)
# plotter.show()