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
    # what `inspect` and `pytest` probe every module for doesn't warn
    hasattr(rendering, "__wrapped__")
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
        assert len(caught) == 2, "two warnings"
        assert all(kind == "DeprecationWarning" for kind, _, _ in caught), "deprecation"
        assert all(filename == "<string>" for _, _, filename in caught), "stacklevel"
        assert "trimesh.viewer.pyglet1`" in caught[0][1], "pyglet1"
        assert "trimesh.rendering`" in caught[-1][1], "rendering"
        assert result["forwarded"], "forwarded"

        # `windowed` is the OpenGL 3.3 viewer on every pyglet with no warning,
        # and main's `render_scene` keywords for a window warn and are dropped
        g.require_rendering()
        code = """
import io, json, sys, warnings
import numpy, pyglet, trimesh
from PIL import Image
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from trimesh.viewer import windowed
    legacy = trimesh.viewer.render_scene(
        trimesh.creation.box().scene(), (64, 48), visible=True, resizable=False
    )
png = trimesh.creation.box().scene().save_image(resolution=(64, 48))
image = Image.open(io.BytesIO(png))
from trimesh.viewer.opengl.viewer import SceneViewer
from trimesh.viewer.render import render_scene
print(json.dumps({
    "viewer": windowed.SceneViewer is SceneViewer,
    "lazy viewer": trimesh.viewer.SceneViewer is SceneViewer,
    "render_scene": windowed.render_scene is render_scene,
    "lazy render_scene": trimesh.viewer.render_scene is render_scene,
    "no pyglet1": "trimesh.viewer.pyglet1" not in sys.modules,
    "png": png.startswith(b"\\x89PNG") and image.format == "PNG",
    "legacy png": bool(numpy.abs(
        numpy.asarray(Image.open(io.BytesIO(legacy)), dtype=int) - numpy.asarray(image)
    ).max() <= 1),
    "size": list(image.size),
    "not blank": bool(numpy.asarray(image.convert("RGB")).std() > 0),
    "caught": [(w.category.__name__, str(w.message), w.filename)
               for w in caught if "April 2028" in str(w.message)],
}))
"""
        env = dict(g.os.environ)
        if g.sys.platform.startswith("linux") and not env.get("DISPLAY"):
            # where pyglet 3 prints its EGL version before the results
            env["PYGLET_HEADLESS"] = "1"
        result = g.json.loads(
            g.subprocess.run(
                [g.sys.executable, "-c", code],
                capture_output=True,
                check=True,
                text=True,
                env=env,
            ).stdout.splitlines()[-1]
        )
        caught = result.pop("caught")
        assert len(caught) == 1, "one warning for the window keywords"
        assert all(kind == "DeprecationWarning" for kind, _, _ in caught), "deprecation"
        assert all(filename == "<string>" for _, _, filename in caught), "stacklevel"
        assert all(
            "visible" in message and "resizable" in message for _, message, _ in caught
        ), "names the keywords"
        assert result.pop("size") == [64, 48], "size"
        for name, ok in result.items():
            assert ok, name

    def test_unchanged(self):
        from pathlib import Path

        # importing trimesh loads no renderer, encoder or pyglet, checked
        # in a fresh interpreter where nothing else imported them
        # nor does converting geometry for the GPU, which Blender renders use
        code = "import sys, trimesh, trimesh.viewer.conversion; "
        code += "print(' '.join(sorted(sys.modules)))"
        loaded = g.subprocess.run(
            [g.sys.executable, "-c", code], capture_output=True, check=True, text=True
        ).stdout.split()
        lazy = ["trimesh.viewer.render", "trimesh.interfaces.ffmpeg", "pyglet"]
        for name in lazy:
            assert name not in loaded, f"{name} not imported"
        assert "trimesh.scene.scene" in loaded, "trimesh imported"

        # a still to a path, where `test_every_scene` checks scenes are unchanged
        g.require_rendering()
        scene = g.get_mesh("duck.dae")
        png = scene.save_image(resolution=(64, 48))
        with g.TemporaryDirectory() as directory:
            path = g.os.path.join(directory, "still.png")
            written = scene.render(Path(path), resolution=(64, 48))
            with open(path, "rb") as file_obj:
                stored = file_obj.read()
        assert g.png(png).shape == (48, 64, 4), "png"
        assert written is None, "a path returns nothing"
        g.assert_images(g.png(stored), g.png(png), "a path holds the same png", 2)
        # what isn't a viewer option raises rather than being dropped
        with g.pytest.raises(TypeError, match="bogus"):
            scene.render(resolution=(64, 48), bogus=True)

    def test_keywords(self):
        # every `Scene.render` default is the viewer's, passing the defaults
        # draws the default image, and changing any of them changes it
        from inspect import signature

        from trimesh.viewer.conversion import RENDER_DEFAULTS
        from trimesh.viewer.render import render_scene

        g.require_rendering()
        defaults = {
            key: item.default
            for key, item in signature(g.trimesh.Scene.render).parameters.items()
            if key in [*RENDER_DEFAULTS, "quality", "background", "samples"]
        }
        scene_render = signature(render_scene).parameters
        assert defaults == {
            **RENDER_DEFAULTS,
            **{
                key: scene_render[key].default
                for key in ("quality", "background", "samples")
            },
        }, "one set of defaults"
        box = g.trimesh.creation.box()
        box.apply_translation([0, 0, 1])
        line = g.trimesh.load_path([[-1, -1, 0.5], [1, 1, 0.5]])
        points = g.trimesh.PointCloud([[0.7, -0.7, 0.5], [-0.7, 0.7, 0.5]])
        scene = g.trimesh.Scene([box, line, points])
        resolution = (96, 72)
        default = g.png(scene.render(resolution=resolution))
        changed = {
            "quality": False,
            "exposure": 2.0,
            "bloom": 0.0,
            "ground_shadow": False,
            "show_environment": True,
            "background": [255, 0, 0, 255],
            "line_width": 6.0,
            "point_size": 12.0,
        }
        for key, value in [*defaults.items(), *changed.items()]:
            if key == "samples":
                continue
            image = g.png(scene.render(resolution=resolution, **{key: value}))
            if value is defaults[key]:
                g.assert_images(image, default, f"default {key}")
            else:
                assert g.np.abs(image.astype(int) - default).max() > 8, f"{key} drawn"
        # bloom is part of quality
        fast = g.png(scene.render(resolution=resolution, quality=False))
        g.assert_images(
            g.png(scene.render(resolution=resolution, quality=False, bloom=0.0)),
            fast,
            "no bloom without quality",
        )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
