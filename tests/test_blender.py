"""
test_blender.py
-----------------

Everything which runs Blender, which `trimesh-setup --matrix` runs on
every supported Blender: its renders have the OpenGL viewer's size, frame
count and silhouettes, which are what the two engines share.
"""

import glob
import re

try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.interfaces import blender
from trimesh.viewer.render import stage

tf = g.trimesh.transformations
PBR = g.trimesh.visual.material.PBRMaterial

if not blender.exists:
    if g.all_dependencies:
        raise ValueError("blender is required!")
    g.pytest.skip("blender is not installed", allow_module_level=True)

RESOLUTION = [160, 120]
FPS = 5


def mixed_scene():
    """A PBR box, a textured mesh, a path, points and glass side by side."""
    box = g.trimesh.creation.box()
    box.visual = g.trimesh.visual.TextureVisuals(
        material=PBR(baseColorFactor=[200, 60, 40, 255], roughnessFactor=0.4)
    )
    glass = g.trimesh.creation.icosphere()
    glass.visual = g.trimesh.visual.TextureVisuals(
        material=PBR(transmissionFactor=1.0, roughnessFactor=0.0, metallicFactor=0.0)
    )
    fuze = g.get_mesh("fuze.obj")
    fuze.apply_scale(1.0 / fuze.extents.max())
    angle = g.np.linspace(0, 12, 100)
    helix = g.np.column_stack([g.np.cos(angle), g.np.sin(angle), angle / 6 - 1])
    cloud = g.trimesh.PointCloud(g.np.random.default_rng(3).uniform(-0.5, 0.5, (50, 3)))
    scene = g.trimesh.Scene()
    for x, geometry in enumerate(
        [box, fuze, g.trimesh.load_path(helix / 2), cloud, glass]
    ):
        scene.add_geometry(geometry, transform=tf.translation_matrix([1.5 * x - 3, 0, 0]))
    scene.camera_transform = tf.translation_matrix([0, 0, 8])
    return scene


def test_engines():
    """Blender frames are the viewer's size and count with its silhouettes."""
    from PIL import Image
    from scipy import ndimage

    g.require_rendering()
    dolly = g.dolly_scene("box", 0.8)
    for name, scene, still in [("mixed", mixed_scene(), True), ("dolly", dolly, False)]:
        staged, count = stage(scene, resolution=RESOLUTION, fps=FPS, still=still)
        with g.tempfile.TemporaryDirectory() as directory:
            blender.build(staged, directory, count=count, fps=FPS, samples=16)
            paths = sorted(glob.glob(g.os.path.join(directory, "frame_*.png")))
            frames = [g.np.array(Image.open(path)) for path in paths]
        # the viewer's segmentation of each frame, posed as that frame is
        renders = []
        for index in range(count):
            posed = scene.copy()
            if not still:
                posed.animate(index / FPS)
            renders.append(
                g.render(posed, include_segmentation=True, resolution=RESOLUTION)
            )
        opengl = g.np.array([render.segmentation >= 0 for render in renders])
        # half a pixel covered, as blends and antialiasing make edges
        blend = g.np.array([frame[..., 3] >= 128 for frame in frames])
        # the engines' edges differ by the pixel of antialiasing and the
        # pixel their projections are off, either side of the viewer's edge
        band = ndimage.binary_dilation(opengl, iterations=2)
        band &= ~ndimage.binary_erosion(opengl, iterations=2)
        assert len(frames) == count, f"{name} frames"
        assert all(f.shape == (120, 160, 4) for f in frames), f"{name} size"
        assert opengl.any(axis=(1, 2)).all() and not ((blend ^ opengl) & ~band).any(), (
            f"{name} silhouettes"
        )
        if name == "dolly":
            # the box slides the same way in both engines
            columns = [
                (mask * g.np.arange(160)).sum(axis=(1, 2)) / mask.sum(axis=(1, 2))
                for mask in (opengl, blend)
            ]
            assert g.np.ptp(columns[0]) > 5.0, "dolly slides"
            assert g.np.allclose(*columns, atol=2.0), "dolly slides alike"


def test_minimum_version():
    """An unknown option, an old Blender and a non-Blender raise clearly."""
    scene = g.trimesh.creation.box().scene()
    with g.pytest.raises(TypeError, match="bogus"):
        scene.render(file_type="png", resolution=[16, 12], engine="blender", bogus=1)
    with g.tempfile.TemporaryDirectory() as directory:
        # the template run by a Python whose `bpy` says it is Blender 4.2
        with open(g.os.path.join(directory, "bpy.py"), "w") as f:
            f.write("from types import SimpleNamespace\n")
            f.write("app = SimpleNamespace(version=(4, 2, 0), version_string='4.2.0')\n")
        old = g.subprocess.run(
            [
                g.sys.executable,
                "-c",
                g.trimesh.resources.get_string("templates/blender_scene.py"),
            ],
            capture_output=True,
            text=True,
            env={**g.os.environ, "PYTHONPATH": directory},
        )
    assert old.returncode != 0 and re.search(r"4\.3.*4\.2\.0", old.stderr), old.stderr
    with g.pytest.MonkeyPatch.context() as patch:
        # Python fails on Blender's arguments, and the error names it
        patch.setattr(blender, "_blender_executable", g.sys.executable)
        with g.pytest.raises(RuntimeError, match=re.escape(g.sys.executable)):
            scene.render(file_type="png", resolution=[16, 12], engine="blender")
