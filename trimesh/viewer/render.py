"""
render.py
-----------

Render a scene to an image or a video with OpenGL or Blender.

```
trimesh.load_scene("model.glb").render("model.mp4")
```

A video plays `scene.animations`, looping them if they are shorter, or
orbits the camera around a scene which isn't animated. Rendering an
engine's frames is kept apart from encoding them, so every engine writes
every file type, and a `.blend` is the scene Blender renders.
"""

import contextlib
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
from ..typed import Number
from .conversion import render_lights

# every file type `render_scene` can write: images and looping animations
# Pillow writes with no ffmpeg, then videos
RENDER_TYPES = ("png", "gif", "webp", "avif", *VIDEO_TYPES)
# the milliseconds an animated image format counts frame durations in,
# where GIF's are hundredths of a second and the rest whole milliseconds
DURATION_TICKS = {"gif": 10}
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
    scene,
    resolution=None,
    file_obj=None,
    file_type: str = "png",
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
      Scene to render.
    resolution : (2,) int or None
      Size in pixels, defaults to `scene.camera.resolution`.
    file_obj : None, str or file-like
      Where to write the render, or None to return it.
    file_type : str
      "png" for a still, or "gif", "webp", or a video like "mp4" for the
      animation of the scene or a turntable if it isn't animated.
    duration : float or None
      Seconds of animation, defaults to `scene.duration` or
      `TURNTABLE_DURATION` if it isn't animated.
    fps : float
      Frames per second of an animation.
    engine : str
      "opengl" for the built in renderer or "blender".
    kwargs : dict
      Passed to the OpenGL `SceneViewer` or to `blender.render`, where
      `WINDOW_KEYWORDS` are deprecated and dropped.

    Returns
    ------------
    data : bytes or None
      The encoded image or video if `file_obj` is None.
    """
    if file_type not in RENDER_TYPES:
        raise ValueError(f"`{file_type}` is not one of {RENDER_TYPES}")
    kwargs = window_keywords(kwargs, "render_scene")
    still = file_type == "png"
    frames = render_frames(scene, resolution, engine, duration, fps, still, **kwargs)
    # the generator frees its viewer or Blender even if encoding fails
    with contextlib.closing(frames):
        return encode(frames, file_obj=file_obj, file_type=file_type, fps=fps)


def encode(frames, file_obj, file_type: str, fps: Number) -> bytes | None:
    """
    Encode (height, width, 3 or 4) uint8 frames as a file.

    Parameters
    ------------
    frames : iterable
      Frames in order.
    file_obj : None, str or file-like
      Where to write the file, or None to return it.
    file_type : str
      One of `RENDER_TYPES`.
    fps : float
      Frames per second of an animation.

    Returns
    ------------
    data : bytes or None
      The encoded file if `file_obj` is None.
    """
    # write the file, or bytes which are returned
    target = file_obj or util.BytesIO()
    if file_type in VIDEO_TYPES:
        with tempfile.TemporaryDirectory() as directory:
            # ffmpeg writes a file itself, seeking around it to finish it
            path = file_obj
            if not isinstance(file_obj, str):
                path = os.path.join(directory, f"render.{file_type}")
            with VideoWriter(path, fps=fps) as video:
                for frame in frames:
                    video.write(frame)
            if path != file_obj:
                with open(path, "rb") as video:
                    shutil.copyfileobj(video, target)
    else:
        from PIL import Image

        images = [Image.fromarray(frame) for frame in frames]
        # one image is a still and more a looping animation whose frames
        # end within half a tick of when they should, so 30 fps in a GIF
        # is 30, 40 and 30 ms rather than 30 which plays 11% fast
        animation = {}
        if len(images) > 1:
            tick = DURATION_TICKS.get(file_type, 1)
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


def render_frames(
    scene,
    resolution=None,
    engine: str = "opengl",
    duration: Number | None = None,
    fps: Number = 30.0,
    still: bool = False,
    **kwargs,
):
    """
    Render the frames of a scene's animation, or a turntable if it
    isn't animated, or one still of its current pose, without encoding
    them: every engine draws the same camera, lights and times.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render, which is not changed.
    resolution : (2,) int or None
      Size in pixels, defaults to `scene.camera.resolution`.
    engine : str
      "opengl" for the built in renderer or "blender".
    duration : float or None
      Seconds of animation, defaults to `scene.duration` or
      `TURNTABLE_DURATION` if it isn't animated.
    fps : float
      Frames per second.
    still : bool
      Render one frame of the scene as it is posed.
    kwargs : dict
      Passed to the OpenGL `SceneViewer` or to `blender.render`.

    Yields
    ------------
    frame : (height, width, 3 or 4) uint8
      Every image in order.
    """
    if engine not in ("opengl", "blender"):
        raise ValueError(f"engine `{engine}` is not `opengl` or `blender`")
    scene, count = stage(scene, resolution, duration=duration, fps=fps, still=still)
    if engine == "blender":
        from ..interfaces import blender

        yield from blender.render(scene, count=count, fps=fps, **kwargs)
        return

    from .opengl.viewer import SceneViewer

    # a render looks through the scene's camera, which may be animated
    kwargs = {"follow_camera": True, "visible": False, "start_loop": False, **kwargs}
    viewer = SceneViewer(scene, resolution=scene.camera.resolution, **kwargs)
    try:
        for index in range(count):
            if count > 1:
                # drawn without posing the scene, looping shorter animations
                viewer.animation_time = index / fps
            viewer.draw(changed=index == 0)
            yield viewer.read_pixels("resolved")
    finally:
        viewer.close()


def stage(
    scene,
    resolution=None,
    duration: Number | None = None,
    fps: Number = 30.0,
    still: bool = False,
):
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
    # a list of its own, as the camera is added to the copy
    scene.cameras = list(original.cameras)
    scene.animations = list(original.animations)
    scene.skins = original.skins
    scene.camera = camera
    if not original.has_camera:
        # a new camera framing the scene stays where it starts, and the
        # scene's own camera stays on its node which may be moving
        scene.camera_transform = camera_transform
    # the default lights are made explicit in the copy, fixed where the
    # camera starts, and the scene's own lights stay on their nodes
    lights, transforms = render_lights(original, camera_transform)
    scene.lights = lights
    for light, transform in zip(lights, transforms):
        if light.name not in scene.graph.nodes:
            scene.graph.update(frame_to=light.name, matrix=transform)

    if duration is None:
        duration = scene.duration or TURNTABLE_DURATION
    count = 1
    if not still:
        count = max(round(duration * fps), 1)
    if not still and scene.duration == 0.0:
        # a turntable driving the camera's node from the base frame with
        # a keyframe every frame, orbiting the scene center once around
        # the world axis nearest the camera's up so the horizon stays level
        start = scene.camera_transform
        scene.graph.update(frame_to=scene.camera.name, matrix=start)
        times = np.arange(count + 1) / fps
        nearest = np.abs(start[:3, 1]).argmax()
        axis = np.eye(3)[nearest] * np.sign(start[nearest, 1])
        orbit = rotation_matrix(2.0 * np.pi * times / duration, axis, scene.centroid)
        nodes = [scene.camera.name]
        turntable = Animation("turntable", nodes, times=times, matrices=orbit @ start)
        scene.animations.append(turntable)
    return scene, count
