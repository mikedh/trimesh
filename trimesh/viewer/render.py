"""
render.py
-----------

Render a scene to an image or a video with OpenGL or Blender.

```
trimesh.load_scene("model.glb").render("model.mp4")
```

A video plays `scene.animations`, looping them if they are shorter, or
orbits the camera around a scene which isn't animated. Every engine writes
every file type.
"""

import contextlib
import glob
import os
import shutil
import tempfile
import warnings

import numpy as np

from .. import util
from ..interfaces.ffmpeg import VIDEO_TYPES, VideoWriter
from ..scene.animation import Animation
from ..scene.cameras import current_view
from ..scene.scene import Scene
from ..transformations import rotation_matrix
from ..typed import Integer, NDArray, Number, Sequence, Stream
from .conversion import render_lights

# every file type `render_scene` can write: images and looping animations
# Pillow writes with no ffmpeg, then videos, then the scene Blender renders
RENDER_TYPES = ("png", "gif", "webp", "avif", *VIDEO_TYPES)
# seconds a turntable takes to orbit a scene which isn't animated
TURNTABLE_DURATION = 4.0
# what `render_scene` and `Scene.save_image` took on `main` for the window
# they rendered in, which a render no longer opens
WINDOW_KEYWORDS = ("visible", "resizable", "start_loop", "fullscreen")


def window_keywords(kwargs: dict, name: str) -> dict:
    """
    Drop the keywords in `WINDOW_KEYWORDS` from a render's keywords
    with a warning, which will raise after April 2028.

    Parameters
    ------------
    kwargs : dict
      Keywords passed to `name` by its caller.
    name : str
      The function they were passed to, for the warning.

    Returns
    ------------
    kept : dict
      Every other keyword.
    """
    ignored = [key for key in WINDOW_KEYWORDS if key in kwargs]
    if len(ignored) > 0:
        message = f"`{name}` ignores {ignored}, which will raise after April 2028"
        warnings.warn(message, category=DeprecationWarning, stacklevel=3)
    return {key: value for key, value in kwargs.items() if key not in WINDOW_KEYWORDS}


def render_scene(
    scene: Scene,
    resolution: Sequence[Integer] | NDArray | None = None,
    file_obj: str | os.PathLike | Stream | None = None,
    file_type: str | None = None,
    duration: Number | None = None,
    fps: Number = 30.0,
    engine: str = "opengl",
    **kwargs,
) -> bytes | None:
    """
    Render a scene to a PNG of its current pose, or to an animation.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render, which is not changed.
    resolution : (2,) int or None
      Size in pixels, defaults to `scene.camera.resolution`.
    file_obj : None, str, pathlib.Path or file-like
      Where to write the render, or None to return it.
    file_type : str or None
      "png" for a still, or "gif", "webp", or a video like "mp4" for the
      animation of the scene or a turntable if it isn't animated, from
      the extension of `file_obj` if not passed, or "png" without either.
    duration : float or None
      Seconds of animation, defaults to `scene.duration` or
      `TURNTABLE_DURATION` if it isn't animated.
    fps : float
      Frames per second of an animation.
    engine : str
      "opengl" for the built in renderer or "blender".
    kwargs : dict
      Passed to the OpenGL `SceneViewer` or to `blender.build`, where
      `WINDOW_KEYWORDS` are deprecated and dropped.

    Returns
    ------------
    data : bytes or None
      The encoded image or video if `file_obj` is None.
    """
    from PIL import Image

    if isinstance(file_obj, (str, os.PathLike)):
        file_obj = os.path.abspath(os.path.expanduser(file_obj))
        # a file name's extension is its type unless one was passed
        file_type = file_type or file_obj.split(".")[-1]
    file_type = (file_type or "png").strip().lower().lstrip(".")
    if file_type not in RENDER_TYPES:
        raise ValueError(f"`{file_type}` is not one of {RENDER_TYPES}")
    if engine not in ("opengl", "blender"):
        raise ValueError(f"engine `{engine}` is not `opengl` or `blender`")
    kwargs = window_keywords(kwargs, "render_scene")
    still = file_type == "png"
    scene, count = stage(scene, resolution, duration=duration, fps=fps, still=still)
    # frames are added to a video as they are drawn, or to the images
    # Pillow writes, which are a still or a looping animation
    target, images = file_obj or util.BytesIO(), []
    with tempfile.TemporaryDirectory() as directory:
        # ffmpeg writes a file itself, seeking around it
        path = file_obj
        if not isinstance(file_obj, str):
            path = os.path.join(directory, f"render.{file_type}")
        with contextlib.ExitStack() as stack:
            add = images.append
            if file_type in VIDEO_TYPES:
                add = stack.enter_context(VideoWriter(path, fps=fps)).write
            if engine == "opengl":
                from .opengl.viewer import SceneViewer

                # a render looks through the scene's camera, which may be animated
                kwargs = {"follow_camera": True, "visible": False, **kwargs}
                viewer = SceneViewer(scene, start_loop=False, **kwargs)
                stack.callback(viewer.close)
                for index in range(count):
                    if not still:
                        # drawn without posing the scene, looping shorter animations
                        viewer.animation_time = index / fps
                    viewer.draw(changed=index == 0)
                    add(viewer.read_pixels("resolved"))
            else:
                from ..interfaces import blender

                # every frame as a PNG over a transparent background
                blender.build(scene, directory, count, fps, **kwargs)
                white = Image.new("RGBA", tuple(scene.camera.resolution), (255,) * 4)
                for name in sorted(glob.glob(os.path.join(directory, "frame_*.png"))):
                    with Image.open(name) as image:
                        frame = Image.alpha_composite(white, image.convert("RGBA"))
                    add(np.asarray(frame))
        if file_type in VIDEO_TYPES:
            if path != file_obj:
                with open(path, "rb") as written:
                    shutil.copyfileobj(written, target)
        else:
            images = [Image.fromarray(frame) for frame in images]
            # one image is a still and more a looping animation whose frames
            # end within half a tick of when they should, so 30 fps in a GIF
            # is 30, 40 and 30 ms rather than 30 which plays 11% fast
            animation = {}
            if len(images) > 1:
                # the milliseconds durations are counted in, where GIF's
                # are hundredths of a second and the rest milliseconds
                tick = {"gif": 10}.get(file_type, 1)
                ends = np.round(np.arange(len(images) + 1) * 1000.0 / (fps * tick))
                animation = {
                    "save_all": True,
                    "append_images": images[1:],
                    "duration": (np.diff(ends).astype(np.int64) * tick).tolist(),
                    "loop": 0,
                }
            images[0].save(target, format=file_type, **animation)
    if file_obj is None:
        return target.getvalue()


def stage(
    scene: Scene,
    resolution: Sequence[Integer] | NDArray | None = None,
    duration: Number | None = None,
    fps: Number = 30.0,
    still: bool = False,
) -> tuple[Scene, int]:
    """
    The copy of a scene every engine renders, with its camera, lights and
    animation made explicit so every engine draws exactly the same ones.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render, which is not changed.
    resolution : (2,) int or None
      Size in pixels, defaults to `scene.camera.resolution`.
    duration : float or None
      Seconds of animation, defaults to `scene.duration` or
      `TURNTABLE_DURATION` if it isn't animated.
    fps : float
      Frames per second.
    still : bool
      One frame of the scene as it is posed.

    Returns
    ------------
    staged : trimesh.Scene
      A copy with a camera, lights ending with an `EnvironmentLight`,
      and an animation unless `still`.
    count : int
      Frames to render.
    """
    # everything below changes a copy, so rendering never changes a scene,
    # which shares the geometry rendering only reads so values cached on
    # it like `smooth_shaded` are computed once rather than every render
    camera, camera_transform = current_view(scene, resolution=resolution)
    original = scene
    scene = Scene(graph=original.graph.copy(), metadata=original.metadata)
    scene.geometry.update(original.geometry)
    # lists of their own, as a camera or a turntable may be added
    scene.cameras = [camera, *original.cameras[1:]]
    scene.animations = list(original.animations)
    scene.skins = original.skins
    # the default lights are made explicit in the copy, fixed where the
    # camera starts, and the scene's own lights stay on their nodes
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
        # a new camera stays where it starts framing the scene, and the
        # scene's own camera on its node which may be moving, unless a
        # turntable drives its node from the base frame
        scene.camera_transform = camera_transform
    if turntable:
        # a keyframe every frame orbiting the scene's center once around
        # the world axis nearest the camera's up so the horizon stays level
        nearest = np.abs(camera_transform[:3, 1]).argmax()
        axis = np.eye(3)[nearest] * np.sign(camera_transform[nearest, 1])
        times = np.arange(count + 1) / fps
        orbit = rotation_matrix(2.0 * np.pi * times / duration, axis, scene.centroid)
        matrices = orbit @ camera_transform
        scene.animations.append(Animation("turntable", [camera.name], times, matrices))
    return scene, count
