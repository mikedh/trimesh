"""
render.py
-----------

Render a scene to an image or a video with OpenGL or Blender.

```
trimesh.load_scene("model.glb").render("model.mp4")
```

Videos play `scene.animations`, or orbit the camera if there are none.
"""

import glob
import os
import shutil
import tempfile
import warnings

import numpy as np

from .. import util
from ..interfaces.ffmpeg import VIDEO_TYPES, write_video
from ..scene.animation import Animation
from ..scene.cameras import current_view
from ..scene.scene import Scene
from ..transformations import rotation_matrix
from ..typed import Integer, Iterable, NDArray, Number, Sequence, Stream
from ..visual.color import to_rgba
from .conversion import RenderSettings, render_lights

# file types `render_scene` writes: Pillow formats then ffmpeg videos
RENDER_TYPES = ("png", "gif", "webp", "avif", *VIDEO_TYPES)
# seconds per orbit of a scene without animations
TURNTABLE_DURATION = 4.0
# deprecated keywords for the window a render no longer opens
WINDOW_KEYWORDS = ("visible", "resizable", "start_loop", "fullscreen")


def render_scene(
    scene: Scene,
    resolution: Sequence[Integer] | NDArray | None = None,
    file_obj: str | os.PathLike | Stream | None = None,
    file_type: str | None = None,
    **kwargs,
) -> bytes | None:
    """
    Render a scene to a PNG of its current pose, or to an animation.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render, not changed.
    resolution : (2,) int or None
      Size in pixels, defaults to `scene.camera.resolution`.
    file_obj : None, str, pathlib.Path or file-like
      Where to write the render, or None to return it.
    file_type : str or None
      One of `RENDER_TYPES`, defaults to the extension or "png".
    kwargs : dict
      `conversion.RenderSettings` fields, the rest passed to `SceneViewer`.

    Returns
    ------------
    data : bytes or None
      Encoded render if `file_obj` is None.
    """
    from PIL import Image

    if isinstance(file_obj, (str, os.PathLike)):
        file_obj = os.path.abspath(os.path.expanduser(file_obj))
        file_type = file_type or file_obj.split(".")[-1]
    file_type = (file_type or "png").strip().lower().lstrip(".")
    if file_type not in RENDER_TYPES:
        raise ValueError(f"`{file_type}` is not one of {RENDER_TYPES}")
    ignored = [key for key in WINDOW_KEYWORDS if key in kwargs]
    if len(ignored) > 0:
        message = f"`render_scene` ignores {ignored}, which will raise after April 2028"
        warnings.warn(message, category=DeprecationWarning, stacklevel=2)
    names = RenderSettings.__dataclass_fields__.keys()
    settings = RenderSettings(**{key: kwargs.pop(key) for key in names & kwargs.keys()})
    viewer = {key: v for key, v in kwargs.items() if key not in WINDOW_KEYWORDS}
    if settings.engine not in ("opengl", "blender"):
        raise ValueError(f"engine `{settings.engine}` is not `opengl` or `blender`")
    # an explicit zero duration is a still in any format
    still = file_type == "png" or settings.duration == 0
    scene, count = stage(scene, resolution, settings.duration, settings.fps, still)
    target = file_obj or util.BytesIO()
    with tempfile.TemporaryDirectory() as directory:
        frames = render_frames(scene, count, settings, directory, still, **viewer)
        if file_type in VIDEO_TYPES:
            # ffmpeg needs a seekable file on disk
            path = file_obj
            if not isinstance(file_obj, str):
                path = os.path.join(directory, f"render.{file_type}")
            write_video(frames, path, fps=settings.fps)
            if path != file_obj:
                with open(path, "rb") as written:
                    shutil.copyfileobj(written, target)
        else:
            images = [Image.fromarray(frame) for frame in frames]
            # round frame ends to the tick: 30 fps GIF is 30, 40, 30 ms
            # rather than 30 ms, which plays 11% fast
            animation = {}
            if len(images) > 1:
                # GIF counts hundredths of a second, the rest milliseconds
                tick = {"gif": 10}.get(file_type, 1)
                ends = np.round(
                    np.arange(len(images) + 1) * 1000.0 / (settings.fps * tick)
                )
                animation = {
                    "save_all": True,
                    "append_images": images[1:],
                    "duration": (np.diff(ends).astype(np.int64) * tick).tolist(),
                    "loop": 0,
                }
            images[0].save(target, format=file_type, **animation)
    if file_obj is None:
        return target.getvalue()


def render_frames(
    scene: Scene,
    count: Integer,
    settings: RenderSettings,
    directory: str,
    still: bool = False,
    **kwargs,
) -> Iterable[NDArray]:
    """
    Every frame of a staged scene.

    Parameters
    ------------
    scene : trimesh.Scene
      From `stage`.
    count : int
      Frames to render.
    settings : RenderSettings
      How to draw the scene.
    directory : str
      Temporary directory for Blender.
    still : bool
      Render one frame as posed.
    kwargs : dict
      Passed to `SceneViewer`.

    Yields
    ------------
    frame : (height, width, 3 or 4) uint8
      Pixels with the first row at the top.
    """
    from PIL import Image

    if settings.engine == "blender":
        from ..interfaces import blender

        # Blender writes transparent PNG frames to put on the background
        blender.build(scene, directory, count, settings, **kwargs)
        color = (255,) * 4
        if settings.background is not None:
            color = tuple(to_rgba(settings.background).tolist())
        backdrop = Image.new("RGBA", tuple(scene.camera.resolution), color)
        for name in sorted(glob.glob(os.path.join(directory, "frame_*.png"))):
            with Image.open(name) as image:
                yield np.asarray(Image.alpha_composite(backdrop, image.convert("RGBA")))
        return

    from .opengl.viewer import SceneViewer

    # look through the scene camera, which may be animated
    viewer = SceneViewer(
        scene,
        visible=False,
        start_loop=False,
        flags={"follow_camera": True},
        **vars(settings),
        **kwargs,
    )
    try:
        for index in range(count):
            if not still:
                # loops shorter animations without posing the scene
                viewer.animation_time = index / settings.fps
            viewer.draw(changed=index == 0)
            yield viewer.read_pixels("resolved")
    finally:
        viewer.close()


def stage(
    scene: Scene,
    resolution: Sequence[Integer] | NDArray | None = None,
    duration: Number | None = None,
    fps: Number = 30.0,
    still: bool = False,
) -> tuple[Scene, int]:
    """
    Copy a scene with explicit camera, lights and animation for every engine.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render, not changed.
    resolution : (2,) int or None
      Size in pixels, defaults to `scene.camera.resolution`.
    duration : float or None
      Seconds, defaults to `scene.duration` or `TURNTABLE_DURATION`.
    fps : float
      Frames per second.
    still : bool
      Render one frame as posed.

    Returns
    ------------
    staged : trimesh.Scene
      Copy with a camera, lights and animation.
    count : int
      Frames to render.
    """
    # share geometry so caches like `smooth_shaded` outlive the copy
    camera, camera_transform = current_view(scene, resolution=resolution)
    original = scene
    scene = Scene(graph=original.graph.copy(), metadata=original.metadata)
    scene.geometry.update(original.geometry)
    # own lists, as a camera or turntable may be added
    scene.cameras = [camera, *original.cameras[1:]]
    scene.animations = list(original.animations)
    scene.skins = original.skins
    # default lights get nodes, the scene's own keep theirs
    lights, transforms = render_lights(original, camera_transform)
    scene.lights = lights
    for light, transform in zip(lights, transforms):
        if light.name not in scene.graph.nodes:
            scene.graph.update(frame_to=light.name, matrix=transform)

    count = 1
    if not still:
        duration = duration or scene.duration or TURNTABLE_DURATION
        count = max(round(duration * fps), 1)
    turntable = not still and scene.duration == 0.0
    if turntable or not original.has_camera:
        # an existing camera keeps its node unless a turntable drives it
        scene.camera_transform = camera_transform
    if turntable:
        # orbit the world axis nearest camera up so the horizon stays level
        nearest = np.abs(camera_transform[:3, 1]).argmax()
        axis = np.eye(3)[nearest] * np.sign(camera_transform[nearest, 1])
        times = np.arange(count + 1) / fps
        orbit = rotation_matrix(2.0 * np.pi * times / duration, axis, scene.centroid)
        matrices = orbit @ camera_transform
        scene.animations.append(Animation("turntable", [camera.name], times, matrices))
    return scene, count
