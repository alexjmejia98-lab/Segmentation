import pyvista as pv

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vtk

# Create server (explicitly use vue3)
server = get_server(client_type="vue3")

# Create PyVista scene
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(pv.Sphere())

# Build UI
with SinglePageLayout(server) as layout:
    layout.title.set_text("PyVista + Trame + Codespaces")

    with layout.content:
        vtk.VtkRemoteView(plotter.ren_win)

# Start
server.start()
