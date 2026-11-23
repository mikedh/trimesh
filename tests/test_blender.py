"""
test_blender.py
-----------------

Blender renders have the OpenGL viewer's size, frame count and
silhouettes, which are what the two engines share.
"""

import glob

try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.interfaces import blender
from trimesh.viewer.conversion import RenderSettings
from trimesh.viewer.render import stage

if not blender.exists:
    if g.all_dependencies:
        raise ValueError("blender is required!")
    g.pytest.skip("blender is not installed", allow_module_level=True)

RESOLUTION = [160, 120]
FPS = 5


def test_engines():
    """Blender frames are the viewer's size and count with its silhouettes."""
    from PIL import Image
    from scipy import ndimage

    g.require_rendering()
    with g.pytest.raises(TypeError, match="bogus"):
        g.trimesh.creation.box().scene().render(
            file_type="png", engine="blender", bogus=1
        )
    # render cases which are stills, and an animation
    scenes = {
        case["name"]: (g.case_scene(case), True)
        for case in g.data["render_cases"]
        if "blender" in case["checks"]
    }
    scenes["dolly"] = (g.dolly_scene("box", 0.8), False)
    for name, (scene, still) in scenes.items():
        staged, count = stage(scene, resolution=RESOLUTION, fps=FPS, still=still)
        with g.tempfile.TemporaryDirectory() as directory:
            blender.build(staged, directory, count, RenderSettings(fps=FPS, samples=16))
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
        assert opengl.any(axis=(1, 2)).all(), f"{name} drawn"
        assert not ((blend ^ opengl) & ~band).any(), f"{name} silhouettes"
