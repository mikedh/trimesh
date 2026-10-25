"""
test_blender.py
-----------------

Everything which runs Blender, which `trimesh-setup --matrix` runs on
every supported Blender: its renders agree with the OpenGL viewer through
the same camera.
"""

import re

try:
    from . import generic as g
    from .test_viewer import (
        SUN_POSE,
        box_shadow,
        clear_glass,
        lighting_cases,
        lit_scene,
    )
except BaseException:
    import generic as g
    from test_viewer import (
        SUN_POSE,
        box_shadow,
        clear_glass,
        lighting_cases,
        lit_scene,
    )

from trimesh.interfaces import blender
from trimesh.scene.lighting import DirectionalLight, PointLight
from trimesh.scene.metadata import Environment, Softbox
from trimesh.viewer.render import render_frames, turntable

tf = g.trimesh.transformations

if not blender.exists:
    if g.all_dependencies:
        raise ValueError("blender is required!")
    g.pytest.skip("blender is not installed", allow_module_level=True)

# the size every engine comparison renders at
RESOLUTION = [160, 120]
# the color of a glTF light which doesn't say, where trimesh's is gray
WHITE = [255, 255, 255, 255]


def render_png(scene, engine):
    """A still of a scene at `RESOLUTION` from an engine, as float RGB."""
    exported = scene.render(file_type="png", resolution=RESOLUTION, engine=engine)
    return g.png(exported)[..., :3].astype(g.np.float64)


def engine_checks(images, prefix):
    """Named checks that the OpenGL and Blender renders of a scene agree."""
    from scipy import ndimage

    drawn = [(image < 250).any(axis=-1) for image in images]
    inside = ndimage.binary_erosion(drawn[0] & drawn[1], iterations=2)
    difference = g.np.abs(images[0] - images[1])
    # mid tones are where falloffs like a spot cone's soft edge are,
    # which a mean over everything would hide, away from silhouettes the
    # engines antialias differently: Blender's fit of glTF's spot edge
    # differs by 17 here, every other case by at most 6, and a spot cone
    # 5% too wide by 28 and 20% by 55, which the test checks fails
    gray = g.np.array([image.mean(axis=-1) for image in images])
    middle = ((gray > 30) & (gray < 200)).any(axis=0) & inside
    return {
        f"{prefix} engines agree": g.iou(*drawn) > 0.95,
        f"{prefix} engines shade alike": difference[inside].mean() < 6.0,
        f"{prefix} engines compared": inside.sum() > 2000,
        f"{prefix} engines shade mid tones alike": difference[middle].mean() < 25.0,
    }


def light_reach(scene, engine, **frames):
    """
    Where the only light of a scene changes the last frame rendered,
    from it on and at zero intensity.
    """
    images = []
    for scale in (1.0, 0.0):
        copy = scene.copy()
        copy.lights[0].intensity *= scale
        rendered = list(
            render_frames(copy, resolution=RESOLUTION, engine=engine, **frames)
        )
        images.append(rendered[-1][..., :3].astype(int))
    return g.np.abs(images[0] - images[1]).max(axis=-1) > 2


def test_engines():
    """Blender and OpenGL render the same exported scene alike, and animations have every frame."""
    from PIL import Image, ImageSequence

    g.require_rendering()
    scene = g.trimesh.creation.box(extents=[3, 1, 2]).scene()
    scene.camera_transform = tf.euler_matrix(0.4, 0.3, 0.1) @ tf.translation_matrix(
        [0, 0, 6]
    )
    settings = g.trimesh.interfaces.ffmpeg.VideoSettings(fps=5)

    checks = {}
    engines = ["opengl", "blender"]
    # every frame of an animation agrees, which a turntable turning 120
    # degrees a frame makes very different frame to frame, without the
    # floor whose soft shadows the engines blur differently, and the
    # turn loops in a video twice as long
    floorless = g.set_render(scene.copy(), ground_shadow=False)
    floorless.animations.append(turntable(floorless, 0.6, 5))
    # and the box turns too in a second animation, as every animation
    # plays at once
    box = floorless.graph.nodes_geometry[0]
    times = g.np.arange(4) * 0.2
    spin = tf.rotation_matrix(times * 2.0 * g.np.pi / 0.6, [1, 0, 0])
    floorless.animations.append(
        g.trimesh.scene.animation.Animation(
            [(None, box)], times, spin[:, None], name="spin"
        )
    )
    animation = {"resolution": [80, 60], "duration": 1.2, "settings": settings}
    frames = []
    for engine in engines:
        gif = floorless.render(file_type="gif", engine=engine, **animation)
        sequence = ImageSequence.Iterator(Image.open(g.trimesh.util.wrap_as_stream(gif)))
        frames.append(
            [(g.np.array(frame.convert("RGB")) < 250).any(-1) for frame in sequence]
        )
    checks["frames agree"] = len(frames[0]) == len(frames[1]) == 6 and all(
        g.iou(first, second) > 0.9 for first, second in zip(*frames)
    )
    for engine, drawn in zip(engines, frames):
        # a GIF merges repeated frames, so a held pose is fewer of them
        checks[f"{engine} loops"] = (
            len(drawn) == 6
            and all(g.iou(drawn[index], drawn[index + 3]) > 0.99 for index in range(3))
            and all(g.iou(drawn[index], drawn[index + 1]) < 0.9 for index in range(2))
        )

    # a saved `.blend` renders the frames Blender renders, here the
    # second time around the loop
    with g.tempfile.TemporaryDirectory() as directory:
        path = g.os.path.join(directory, "scene.blend")
        floorless.export(path, resolution=[80, 60], duration=1.2, fps=5)
        g.subprocess.run(
            [blender._blender_executable, "--background", path]
            + ["--render-output", g.os.path.join(directory, "frame_")]
            + ["--render-frame", "4"],
            check=True,
            capture_output=True,
        )
        reopened = Image.open(g.os.path.join(directory, "frame_0004.png"))
        white = Image.new("RGBA", reopened.size, (255, 255, 255, 255))
        reopened = g.np.asarray(
            Image.alpha_composite(white, reopened.convert("RGBA"))
        ).astype(int)
    direct = list(
        render_frames(
            floorless, resolution=[80, 60], engine="blender", duration=1.2, fps=5
        )
    )[4].astype(int)
    checks["blend renders the same"] = g.np.abs(reopened - direct).max() <= 2

    # an environment seen behind the scene, which both engines draw alike
    wide = Softbox(direction=[0.3, 0.8, 0.5], sharpness=6.0, color=[2] * 3)
    shown = Environment(sky=[0.6, 0.5, 0.4], ground=[0.05, 0.08, 0.1], softboxes=[wide])
    # saturated sRGB colors, which glTF stores as linear
    colored = g.trimesh.creation.icosphere(subdivisions=4)
    colored.visual.vertex_colors = g.trimesh.visual.color.interpolate(
        colored.vertices[:, 0], color_map="viridis"
    )
    faced = g.trimesh.creation.box(extents=[2.0, 1.5, 1.0])
    faced.visual.face_colors = g.np.random.default_rng(4).integers(
        0, 255, (len(faced.faces), 3)
    )
    # the shared cases, where Blender draws a spot light's soft edge with
    # a different curve than glTF, fit as close as it can, and a thin
    # sheet lit through from behind
    cases = lighting_cases()
    shared = {
        name: cases[name]
        for name in [
            "directional",
            "spot",
            "autolight",
            "custom environment",
            "diffuse transmission",
        ]
    }
    # Blender 4.5 ignores a material's slab thickness, so it draws thin
    # surfaces lit from behind darker than glTF says
    if blender.version()[:2] == (4, 5):
        shared.pop("diffuse transmission")
    # a camera whose fields of view don't match the image's shape
    # has pixels which aren't square, and both engines keep them
    stretched = scene.copy()
    stretched.camera.fov = [70, 40]
    # and whose far plane cuts the box in half, which both engines clip
    clipped = scene.copy()
    clipped.camera.z_far = 6.0
    # the lights and environment reach Blender through the glTF in the
    # same units, so the shading agrees too and not just shape
    for name, lit in {
        "silhouette": scene,
        "stretched pixels": stretched,
        "far plane": clipped,
        # both engines cast the same shadow from the same light
        "shadowed": box_shadow(DirectionalLight(name="sun", intensity=3.0), SUN_POSE),
        "environment shown": g.set_render(
            lit_scene(environment=shown, metallic=0.5, roughness=0.3),
            show_environment=True,
        ),
        # no material or colors at all, which isn't in the file
        "default color": g.trimesh.creation.icosphere(subdivisions=4).scene(),
        "vertex colors": colored.scene(),
        # a different color on every triangle, which glTF only has
        # on vertices
        "face colors": faced.scene(),
        # clear glass over a colored sphere, which both engines see through
        "glass": g.trimesh.Scene([colored, clear_glass(0.02, 1.5, 0.3)]),
        # one stop brighter, which Blender's exposure is in
        "exposure": g.set_render(cases["directional"].copy(), exposure=2.0),
        **shared,
    }.items():
        images = [render_png(lit, engine) for engine in engines]
        checks.update(engine_checks(images, name))

    # the thresholds bite: a spot cone 20% wider in Blender alone, which
    # moves the soft edge a few pixels, shades mid tones differently
    wider = shared["spot"].copy()
    wider.lights[0].outerConeAngle *= 1.2
    wider.lights[0].innerConeAngle *= 1.2
    images = [render_png(shared["spot"], "opengl"), render_png(wider, "blender")]
    control = engine_checks(images, "wider spot")
    checks["a wider spot cone fails the mid tones"] = not control[
        "wider spot engines shade mid tones alike"
    ]

    # points and lines, which Blender draws as spheres and tubes as
    # many pixels across as the viewer, under nodes which scale them
    rng = g.np.random.default_rng(3)
    hues = g.np.column_stack([rng.random(200), g.np.ones((200, 2))])
    cloud = g.trimesh.PointCloud(
        rng.uniform(-1, 1, (200, 3)) * [0.5, 0.33, 0.33],
        colors=g.trimesh.visual.color.hsv_to_rgba(hues),
    )
    angle = g.np.linspace(0, 12, 200)
    helix = g.np.column_stack([g.np.cos(angle), g.np.sin(angle), angle / 6 - 1])
    segments = g.trimesh.Scene()
    segments.add_geometry(cloud, transform=tf.scale_matrix(3))
    segments.add_geometry(g.trimesh.load_path(helix / 2), transform=tf.scale_matrix(2))
    segments.camera_transform = scene.camera_transform
    images = [render_png(segments, engine) for engine in engines]
    drawn = [(image < 250).any(axis=-1) for image in images]
    checks["segments engines agree"] = g.iou(*drawn) > 0.9
    # Blender renders glass over its floor the same every time
    glassy = g.trimesh.Scene([colored, clear_glass(0.02, 1.5, 0.3)])
    checks["blender repeats exactly"] = g.np.array_equal(
        *[
            g.png(glassy.render(file_type="png", resolution=RESOLUTION, engine="blender"))
            for _ in range(2)
        ]
    )
    # the spheres are lit the way the viewer lights its points
    means = [image[mask].mean(axis=0) for image, mask in zip(images, drawn)]
    checks["segments engines color alike"] = g.np.abs(means[0] - means[1]).max() < 16

    # a light reaches the same pixels in both engines: far and dim, which
    # EEVEE culls by default, and exactly dark past a glTF `range`
    far = lit_scene(
        [PointLight(name="bulb", color=WHITE, intensity=1.0)],
        [tf.translation_matrix([0, 0, 3.0])],
        plane=True,
    )
    bulb = tf.translation_matrix([0, 0, 2.0])
    ranged = lit_scene(
        [PointLight(name="bulb", color=WHITE, intensity=6.0, radius=3.0)],
        [bulb],
        plane=True,
        environment=Environment(sky=[0.1] * 3, ground=[0.1] * 3),
    )
    # and on the last frame of a box sliding past where the light reached
    # with the box where it starts, which lights reach in every frame
    sliding = g.set_render(g.dolly_scene("box", duration=3.0), ground_shadow=False)
    sliding.lights = [PointLight(name="lamp", color=WHITE, intensity=40.0)]
    sliding.graph["lamp"] = tf.translation_matrix([-3.0, 0.0, 2.0])
    reaches = {}
    for name, lit, frames in [
        ("far point", far, {"still": True}),
        ("point range", ranged, {"still": True}),
        ("sliding box", sliding, {"duration": 3.0, "fps": 1.0}),
    ]:
        reaches[name] = [light_reach(lit, engine, **frames) for engine in engines]
        checks[f"{name} reaches the same pixels"] = g.iou(*reaches[name]) > 0.9
    # the pixels whose camera ray hits the plane within the light's range
    ranged.camera.fit_resolution(RESOLUTION)
    hits = g.camera_hits(ranged)
    within = g.np.zeros(reaches["point range"][0].shape, dtype=bool)
    near = g.np.linalg.norm(hits.points - bulb[:3, 3], axis=1) < 3.0
    within[hits.rows[near], hits.columns[near]] = True
    for engine, reach in zip(engines, reaches["point range"]):
        checks[f"{engine} point range dark past the range"] = not (reach & ~within).any()
        checks[f"{engine} point range lit inside the range"] = (
            reach.sum() > 0.5 * within.sum()
        )

    # a camera node with a scale, which files exported from 3ds Max carry,
    # has its planes in its own units as glTF says, which Blender's aren't
    glossy = lit_scene(metallic=0.5, roughness=0.3)
    pose = glossy.camera_transform
    renders = []
    for clip, transform in [(0.1, pose), (0.1 / 0.025, pose @ tf.scale_matrix(0.025))]:
        glossy.camera.z_near = clip
        glossy.camera_transform = transform
        exported = glossy.render(file_type="png", resolution=[80, 60], engine="blender")
        renders.append(g.png(exported)[..., :3].astype(int))
    checks["camera scale is glTF's"] = g.np.abs(renders[0] - renders[1]).max() <= 1

    for name, ok in checks.items():
        assert ok, name


def test_minimum_version():
    """
    Blender older than the template supports raises naming the version,
    and an executable which isn't Blender raises naming the executable.
    """
    scene = g.trimesh.creation.box().scene()
    with g.pytest.MonkeyPatch.context() as patch:
        patch.setitem(blender._BLENDER_VERSION, blender._blender_executable, (4, 2, 0))
        with g.pytest.raises(ValueError, match=r"4\.3.*4\.2\.0"):
            scene.render(file_type="png", resolution=[16, 12], engine="blender")
    with g.pytest.MonkeyPatch.context() as patch:
        # Python answers `--version` with something other than Blender's
        patch.setattr(blender, "_blender_executable", g.sys.executable)
        patch.delitem(blender._BLENDER_VERSION, g.sys.executable, raising=False)
        with g.pytest.raises(RuntimeError, match=re.escape(g.sys.executable)):
            scene.render(file_type="png", resolution=[16, 12], engine="blender")
