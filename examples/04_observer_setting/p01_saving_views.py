"""
TODO add description
"""

from pyvisual import Plot3d

plotter = Plot3d()
plotter.add_sun()
plotter.add_latitudinal_lines()
plotter.add_longitudinal_lines()
plotter.show_axes()
plotter.add_camera_update()
plotter.show()

orientation = plotter.observer_orientation
print(orientation)
plotter.observer_orientation = orientation
print(plotter.observer_orientation)
plotter.show()
plotter.observer_orientation = orientation
plotter.show()
