# the fixed function viewer for pyglet<2 which nothing imports by default:
# use `trimesh.viewer.pyglet1.viewer.SceneViewer` explicitly
import warnings

warnings.warn(
    "`trimesh.viewer.pyglet1` is the viewer for `pyglet<2` and will be removed after "
    + "April 2028: `trimesh.viewer.SceneViewer` works with every pyglet version",
    category=DeprecationWarning,
    stacklevel=2,
)
