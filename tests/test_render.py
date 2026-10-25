try:
    from . import generic as g
except BaseException:
    import generic as g


class RenderTest(g.unittest.TestCase):
    """
    Test to make sure we can convert trimesh objects
    to pyglet objects for rendering.
    """

    def test_args(self):
        # not imported to trimesh by __init__
        # so we don't have a pyglet import attempt
        # unless we explicitly ask for it (like during
        # a viewer show() call)
        from trimesh import rendering

        files = ["featuretype.STL", "fuze.obj", "points_bin.ply"]
        for file_name in files:
            m = g.get_mesh(file_name)
            args = rendering.convert_to_vertexlist(m)

            if isinstance(m, g.trimesh.Trimesh):
                # try turning smoothing off and on
                rendering.mesh_to_vertexlist(m, smooth_threshold=0)
                rendering.mesh_to_vertexlist(m, smooth_threshold=g.np.inf)

                P30 = m.section(plane_normal=[0, 0, 1], plane_origin=m.centroid)
                args = rendering.path_to_vertexlist(P30)
                args_auto = rendering.convert_to_vertexlist(P30)
                assert len(args) == 6
                assert len(args_auto) == len(args)

                P20, _T = P30.to_planar()
                args = rendering.path_to_vertexlist(P20)
                args_auto = rendering.convert_to_vertexlist(P20)
                assert len(args) == 6
                assert len(args_auto) == len(args)

        P21 = g.get_mesh("2D/wrench.dxf")
        args = rendering.path_to_vertexlist(P21)
        args_auto = rendering.convert_to_vertexlist(P21)
        assert len(args) == 6
        assert len(args_auto) == len(args)

        P22 = g.random((100, 2))
        args = rendering.points_to_vertexlist(P22)
        args_auto = rendering.convert_to_vertexlist(P22)
        assert len(args) == 6
        assert len(args_auto) == len(args)

        P31 = g.random((100, 3))
        args = rendering.points_to_vertexlist(P31)
        args_auto = rendering.convert_to_vertexlist(P31)
        assert len(args) == 6
        assert len(args_auto) == len(args)

        P32 = g.trimesh.points.PointCloud(P31)
        args = rendering.convert_to_vertexlist(P32)
        assert len(args) == 6
        assert len(args_auto) == len(args)

    def test_deprecated(self):
        # `trimesh.viewer.pyglet1` and its `trimesh.rendering` alias
        # keep working and warn the caller until April 2028
        # in a fresh interpreter so each module is imported for the first time
        code = """
import json, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import trimesh.viewer.pyglet1
    from trimesh import rendering
    from trimesh.viewer.pyglet1 import conversion
    function = rendering.convert_to_vertexlist
print(json.dumps({
    "forwarded": function is conversion.convert_to_vertexlist,
    "caught": [(w.category.__name__, str(w.message), w.filename)
               for w in caught if "April 2028" in str(w.message)],
}))
"""
        result = g.json.loads(
            g.subprocess.run(
                [g.sys.executable, "-c", code], capture_output=True, check=True, text=True
            ).stdout
        )
        caught = result["caught"]
        # one warning from each module, both pointing at the importing code
        checks = {
            "two warnings": len(caught) == 2,
            "deprecation": all(kind == "DeprecationWarning" for kind, _, _ in caught),
            "stacklevel": all(filename == "<string>" for _, _, filename in caught),
            "pyglet1": "trimesh.viewer.pyglet1`" in caught[0][1],
            "rendering": "trimesh.rendering`" in caught[-1][1],
            "forwarded": result["forwarded"],
        }
        for name, ok in checks.items():
            assert ok, name

        # `windowed` is main's viewer with one warning on pyglet<2, and
        # the OpenGL 3.3 viewer with none on every newer pyglet
        g.require_rendering()
        code = """
import io, json, sys, warnings
import numpy, pyglet, trimesh
from PIL import Image
pyglet1 = int(pyglet.version.split(".")[0]) < 2
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from trimesh.viewer import windowed
png = trimesh.creation.box().scene().save_image(resolution=(64, 48))
image = Image.open(io.BytesIO(png))
if pyglet1:
    from trimesh.viewer.pyglet1.viewer import SceneViewer, render_scene
    other = "trimesh.viewer.opengl.viewer"
else:
    from trimesh.viewer.opengl.viewer import SceneViewer
    from trimesh.viewer.render import render_scene
    other = "trimesh.viewer.pyglet1"
print(json.dumps({
    "pyglet1": pyglet1,
    "viewer": windowed.SceneViewer is SceneViewer,
    "lazy viewer": trimesh.viewer.SceneViewer is SceneViewer,
    "render_scene": windowed.render_scene is render_scene,
    "lazy render_scene": trimesh.viewer.render_scene is render_scene,
    "other viewer never imported": other not in sys.modules,
    "png": png.startswith(b"\\x89PNG") and image.format == "PNG",
    "size": list(image.size),
    "not blank": bool(numpy.asarray(image.convert("RGB")).std() > 0),
    "caught": [(w.category.__name__, str(w.message), w.filename)
               for w in caught if "April 2028" in str(w.message)],
}))
"""
        env = dict(g.os.environ)
        if g.sys.platform.startswith("linux") and not env.get("DISPLAY"):
            # main's viewer has no EGL fallback of its own
            env["PYGLET_HEADLESS"] = "1"
        result = g.json.loads(
            g.subprocess.run(
                [g.sys.executable, "-c", code],
                capture_output=True,
                check=True,
                text=True,
                env=env,
            ).stdout
        )
        caught = result.pop("caught")
        checks = {
            "one warning only on pyglet<2": len(caught) == int(result.pop("pyglet1")),
            "deprecation": all(kind == "DeprecationWarning" for kind, _, _ in caught),
            "stacklevel": all(filename == "<string>" for _, _, filename in caught),
            "says upgrade pyglet": all("pyglet" in message for _, message, _ in caught),
            "size": result.pop("size") == [64, 48],
            **result,
        }
        for name, ok in checks.items():
            assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
