# deprecated fixed function viewer for pyglet<2, only imported explicitly
import warnings

warnings.warn(
    "`trimesh.viewer.pyglet1` is the viewer for `pyglet<2` and will be removed after "
    + "April 2028: `trimesh.viewer.opengl.viewer.SceneViewer` works with every pyglet",
    category=DeprecationWarning,
    stacklevel=2,
)
