"""
Instantiating a Plot3d Object
=============================

This example demonstrates how to create an instance of the Plot3d class,
add and remove actors, and display results in an interactive visualization window.
"""

from pyvisual import Plot3d

# %%
# .. note::
#    As stated throughout the **pyvisual** documentation site, it is *highly* recommended to explore the
#    `PyVista <https://docs.pyvista.org/>`_ documentation to learn about the extensive capabilities
#    provided by the PyVista and VTK libraries. **pyvisual** is but a thin wrapper around a small
#    subset of these capabilities, and thus, learning about the underlying libraries will allow you to
#    make the most of **pyvisual**.

# %%
# The Plot3d class inherits from :class:`~pyvista.Plotter` and, therefore, can be instantiated
# with the arguments articulated in the linked documentation, *e.g.* applying a colored border
# to the visualization window, setting the lighting scheme, or using off-screen rendering.

# %%
# Once instantiated, the Plot3d object can be used to add actors/meshes to the scene. Here we
# add axes (to display the coordinate system) and a sun (a "reference" sphere, centered
# at the origin, with radius 1 :math:`R_\odot`). By calling the :func:`~pyvisual.core.plot3d.Plot3d.show`
# method, we can display the scene in an interactive window.

plotter = Plot3d(
    off_screen=True,
    border=True,
    border_color='white',
    lighting='three lights',
    window_size=(500, 500)
)

plotter.show_axes()
sun_actor = plotter.add_sun()
plotter.show()

# %%
# By capturing the returned result of the :func:`~pyvisual.Plot3d.add_sun` method, we can
# manipulate the sun actor after it has been added to the scene. For example, we can remove it
# from the scene and then add it back again.

# sphinx_gallery_start_ignore
# NOTE: The following code block can be removed when run outside the sphinx gallery pipeline.
#       for each code block (defined by "# %%"), the plotter is destroyed; therefore, it is
#       necessary to re-instantiate it. In practice, multiple ``show()`` calls can be executed
#       without re-instantiating the plotter.
plotter = Plot3d(
    off_screen=True,
    border=True,
    border_color='white',
    lighting='three lights',
    window_size=(500, 500)
)

plotter.show_axes()
sun_actor = plotter.add_sun()
# sphinx_gallery_end_ignore

plotter.remove_actor(sun_actor)
plotter.show()