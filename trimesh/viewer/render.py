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
import tempfile

import numpy as np

from .. import util
from ..interfaces.ffmpeg import VIDEO_TYPES, VideoSettings, VideoWriter, frame_durations
from ..scene.cameras import current_view
from ..scene.metadata import RenderSettings
from ..scene.scene import Scene
from ..typed import Number

# written by Pillow with no ffmpeg: a still image, or a looping animation
PILLOW_TYPES = ("png", "gif", "webp", "avif")

# every file type `render_scene` can write
RENDER_TYPES = (*PILLOW_TYPES, *VIDEO_TYPES)


def render_scene(
    scene,
    resolution=None,
    file_type: str = "png",
    engine: str = "opengl",
    duration: Number | None = None,
    settings: VideoSettings | None = None,
    **kwargs,
) -> bytes:
    """
    Render a scene to a PNG of its current pose, or to an animation.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render.
    resolution : (2,) int or None
      Size in pixels, defaults to `scene.camera.resolution`.
    file_type : str
      "png" for a still, or "gif", "webp", or a video like "mp4" for the
      animation of the scene or a turntable if it isn't animated.
    engine : str
      "opengl" for the built in renderer or "blender".
    duration : float or None
      Seconds of animation, defaults to `scene.duration` or 4.0.
    settings : VideoSettings or None
      Frame rate and encoder for animations.
    kwargs : dict
      Passed to the OpenGL `SceneViewer`.

    Returns
    ------------
    data : bytes
      The encoded image or video.
    """
    if file_type not in RENDER_TYPES:
        raise ValueError(f"`{file_type}` is not one of {RENDER_TYPES}")
    if settings is None:
        settings = VideoSettings()
    frames = render_frames(
        scene,
        resolution=resolution,
        engine=engine,
        duration=duration,
        fps=settings.fps,
        still=file_type == "png",
        **kwargs,
    )
    # the generator frees its viewer or Blender even if encoding fails
    with contextlib.closing(frames):
        return encode(frames, file_type=file_type, settings=settings)


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
      Seconds of animation, defaults to `scene.duration` or 4.0.
    fps : float
      Frames per second.
    still : bool
      Render one frame of the scene as it is posed.
    kwargs : dict
      Passed to the OpenGL `SceneViewer`.

    Returns
    ------------
    frames : iterable
      (height, width, 3 or 4) uint8 images in order.
    """
    if engine not in ("opengl", "blender"):
        raise ValueError(f"engine `{engine}` is not `opengl` or `blender`")
    scene, studio, count = stage(
        scene, resolution=resolution, duration=duration, fps=fps, still=still
    )
    if engine == "blender":
        from ..interfaces import blender

        return blender.render(scene, studio=studio, count=count, fps=fps)
    return opengl_frames(scene, count=count, fps=fps, **kwargs)


def blend_scene(
    scene,
    resolution=None,
    duration: Number | None = None,
    fps: Number = 30.0,
    samples: int = 64,
) -> bytes:
    """
    The scene as a `.blend`, which Blender renders the same frames as
    `render_frames(engine="blender")` from, with its animation, cameras,
    lights, environment and images.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to save, which is not changed.
    resolution : (2,) int or None
      Render size in pixels, defaults to `scene.camera.resolution`.
    duration : float or None
      Seconds in the frame range, defaults to `scene.duration` or 4.0.
    fps : float
      Frames per second.
    samples : int
      Antialiasing samples per pixel for rendering.

    Returns
    ------------
    blend : bytes
      The `.blend` file.
    """
    from ..interfaces import blender

    scene, studio, count = stage(scene, resolution=resolution, duration=duration, fps=fps)
    return blender.save_blend(scene, studio=studio, count=count, fps=fps, samples=samples)


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
      Seconds of animation, defaults to `scene.duration` or 4.0.
    fps : float
      Frames per second.
    still : bool
      One frame of the scene as it is posed.

    Returns
    ------------
    staged : trimesh.Scene
      A copy with a camera, lights and an animation unless `still`.
    studio : trimesh.scene.metadata.Studio
      Every render default filled in for where the camera starts.
    count : int
      Frames to render.
    """
    # everything below changes a copy, so rendering never changes a scene,
    # which shares the geometry rendering only reads so values cached on
    # it like `smooth_shaded` are computed once rather than every render
    framed = not scene.has_camera
    camera, camera_transform = current_view(scene, resolution=resolution)
    original = scene
    scene = Scene(graph=original.graph.copy(), metadata=original.metadata)
    scene.geometry.update(original.geometry)
    scene.cameras = original.cameras
    scene.animations = list(original.animations)
    scene.skins = original.skins
    if original.has_lights:
        scene.lights = list(original.lights)
    scene.camera = camera
    if framed:
        # a new camera framing the scene stays where it starts, and the
        # scene's own camera stays on its node which may be moving
        scene.camera_transform = camera_transform
    studio = RenderSettings.from_metadata(scene.metadata).resolve(scene, camera_transform)
    if not scene.has_lights:
        # the default lights are made explicit in the copy, fixed where
        # the camera starts
        scene.lights = studio.lights
        for light, transform in zip(studio.lights, studio.transforms):
            scene.graph.update(frame_to=light.name, matrix=transform)

    count = 1
    if not still:
        if scene.duration == 0.0:
            scene.animations.append(turntable(scene, duration or 4.0, fps))
        count = max(round((duration or scene.duration) * fps), 1)
    return scene, studio, count


def opengl_frames(scene, count: int, fps: Number, **kwargs):
    """
    Render frames of a scene with the built in OpenGL renderer.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render at `scene.camera.resolution`.
    count : int
      One frame of the current pose, or this many frames of
      `scene.animations` every `1 / fps` seconds.
    fps : float
      Frames per second of animation.
    kwargs : dict
      Passed to `SceneViewer`.

    Returns
    ------------
    frames : generator
      (height, width, 4) uint8 RGBA images.
    """
    from .opengl.viewer import SceneViewer

    # a render is always offscreen so these don't apply
    for key in ("visible", "start_loop", "fullscreen", "resizable"):
        kwargs.pop(key, None)
    viewer = SceneViewer(
        scene,
        resolution=scene.camera.resolution,
        visible=False,
        start_loop=False,
        **kwargs,
    )
    from .opengl import gl, read_framebuffer_finish, read_framebuffer_start

    # two pixel buffers so one frame is copied off the GPU while the
    # next is drawn, rather than waiting for every frame
    buffers = (gl.GLuint * 2)()
    gl.glGenBuffers(2, buffers)
    framebuffer = viewer.resolve_framebuffer
    try:
        previous = None
        for index in range(count):
            if count > 1:
                # an animation shorter than the video loops
                scene.animate(index / fps, loop=True)
                # a render looks through the scene's camera, which may be animated
                viewer.place_camera(scene.camera_transform)
            # animation only moves the scene graph, which every frame reads
            viewer.render(changed=index == 0)
            read_framebuffer_start(framebuffer, buffers[index % 2])
            if previous is not None:
                yield read_framebuffer_finish(framebuffer, previous)
            previous = buffers[index % 2]
        yield read_framebuffer_finish(framebuffer, previous)
    finally:
        viewer.window.switch_to()
        gl.glDeleteBuffers(2, buffers)
        viewer.close()


def encode(frames, file_type: str, settings: VideoSettings) -> bytes:
    """
    Encode (height, width, 3 or 4) uint8 frames as a file.

    Parameters
    ------------
    frames : iterable
      Frames in order.
    file_type : str
      One of `RENDER_TYPES`.
    settings : VideoSettings
      Frame rate and encoder.

    Returns
    ------------
    data : bytes
      The encoded file.
    """
    if file_type in VIDEO_TYPES:
        # ffmpeg seeks around the file to finish some containers
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, f"render.{file_type}")
            with VideoWriter(path, settings) as video:
                for frame in frames:
                    video.write(frame)
            with open(path, "rb") as file_obj:
                return file_obj.read()

    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    with util.BytesIO() as buffer:
        # one image is a still and more is a looping animation, whose
        # frames end on time in the units each format stores
        animation = {}
        if len(images) > 1:
            animation = {
                "save_all": True,
                "append_images": images[1:],
                "duration": frame_durations(len(images), settings.fps, file_type),
                "loop": 0,
            }
        images[0].save(buffer, format=file_type, **animation)
        return buffer.getvalue()


def turntable(scene, duration: Number, fps: Number):
    """
    A camera animation orbiting the scene once around the world axis
    nearest the camera's up, so the horizon stays level, with a
    keyframe for every frame.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to orbit, using its current camera.
    duration : float
      Seconds for one full orbit.
    fps : float
      Keyframes per second.

    Returns
    ------------
    animation : trimesh.scene.animation.Animation
      Drives the camera node from the base frame.
    """
    from ..scene.animation import Animation
    from ..transformations import rotation_matrix

    start = scene.camera_transform
    times = np.arange(max(round(duration * fps), 1) + 1) / fps
    # the world axis nearest the camera's up, pointing the same way
    nearest = np.abs(start[:3, 1]).argmax()
    axis = np.eye(3)[nearest] * np.sign(start[nearest, 1])
    # a full turn around it through the scene center over the duration
    orbit = rotation_matrix(2.0 * np.pi * times / duration, axis, scene.centroid)
    return Animation(
        edges=[(None, scene.camera.name)],
        times=times,
        matrices=orbit @ start,
        name="turntable",
    )
