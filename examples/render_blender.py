"""
render_blender.py
-------------------

Render an animated glTF file headlessly with Blender.

This is a `bpy` script and is not run by python directly, it is
passed to Blender which supplies the `bpy` module:

    blender -b -P examples/render_blender.py -- cycloidal.glb out.mp4

Produces an MP4 if the output name ends in `.mp4`, otherwise a
numbered PNG sequence. Generate the input with
`examples/animation_cycloidal.py` or `examples/animation_robot.py`.
"""

import math
import os
import sys

import bpy
import mathutils

# how much empty space to leave around the subject. note that the camera
# is fit to every pose in the animation, so a subject which sweeps a large
# volume will look small: that space is where it is about to move
MARGIN = 1.1


def clear():
    """
    Delete everything in the startup scene.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)


def duration():
    """
    Find the last frame of any imported animation.

    Returns
    ----------
    end : int
      Final frame, or 1 if nothing is animated.
    """
    end = 1.0
    for action in bpy.data.actions:
        # `frame_range` is in frames, already converted from the
        # glTF times using the scene frame rate at import
        end = max(end, action.frame_range[1])
    return math.ceil(end)


def look_at(subjects, frames=None, samples=24):
    """
    Add a camera framing every passed object, and some lights.

    Parameters
    ------------
    subjects : list
      Blender objects to fit in the frame.
    frames : (2,) int or None
      First and last frame, so the camera can be fit to the whole
      animation rather than just the pose the file happens to open in.
    samples : int
      How many frames to check when fitting.

    Returns
    ----------
    camera : bpy.types.Object
      The created camera.
    """
    scene = bpy.context.scene
    if frames is None:
        moments = [scene.frame_current]
    else:
        # an animated subject sweeps out much more space than its rest
        # pose, and fitting to the rest pose alone will crop the motion
        step = max(1, (frames[1] - frames[0]) // max(1, samples - 1))
        moments = sorted(set(range(frames[0], frames[1] + 1, step)) | {frames[1]})

    corners = []
    for moment in moments:
        scene.frame_set(moment)
        corners.extend(
            obj.matrix_world @ mathutils.Vector(corner)
            for obj in subjects
            for corner in obj.bound_box
        )
    scene.frame_set(moments[0])

    low = mathutils.Vector(map(min, zip(*corners)))
    high = mathutils.Vector(map(max, zip(*corners)))

    center = (low + high) / 2.0
    radius = max((high - low).length / 2.0, 1e-4)

    # an empty at the center for the camera and lights to aim at
    target = bpy.data.objects.new("target", None)
    bpy.context.collection.objects.link(target)
    target.location = center

    camera_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.collection.objects.link(camera)

    # the direction the camera sits in, relative to the subject
    view = mathutils.Vector((0.62, -0.72, 0.31)).normalized()

    # the camera's own axes, so width and height can be fit separately
    # against their own field of view rather than a single worst case
    forward = -view
    right = forward.cross(mathutils.Vector((0.0, 0.0, 1.0))).normalized()
    up = right.cross(forward).normalized()

    offsets = [corner - center for corner in corners]
    half_width = max(abs(o.dot(right)) for o in offsets)
    half_height = max(abs(o.dot(up)) for o in offsets)
    along = max(abs(o.dot(view)) for o in offsets)

    # `angle` covers the larger image dimension, the other is narrower
    render = bpy.context.scene.render
    wide = camera_data.angle
    narrow = 2.0 * math.atan(
        math.tan(wide / 2.0)
        * min(render.resolution_x, render.resolution_y)
        / max(render.resolution_x, render.resolution_y)
    )
    if render.resolution_x >= render.resolution_y:
        fov_x, fov_y = wide, narrow
    else:
        fov_x, fov_y = narrow, wide

    # back off far enough for both axes to fit, plus the subject depth
    distance = along + MARGIN * max(
        half_width / math.tan(fov_x / 2.0), half_height / math.tan(fov_y / 2.0)
    )
    camera.location = center + view * distance
    # clip planes have to bracket a subject of any scale
    camera_data.clip_start = max(distance - along - radius, radius * 1e-3)
    camera_data.clip_end = distance + along + radius * 2.0

    _aim(camera, target)
    bpy.context.scene.camera = camera

    # a key light and a softer fill, both aimed at the subject
    for name, offset, energy in (
        ("key", (1.0, -0.8, 1.4), 5.0),
        ("fill", (-1.2, -0.4, 0.5), 2.0),
    ):
        light_data = bpy.data.lights.new(name, type="SUN")
        light_data.energy = energy
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = center + mathutils.Vector(offset) * distance
        _aim(light, target)

    return camera


def _aim(obj, target):
    """
    Point an object at a target using a track-to constraint.

    Parameters
    ------------
    obj : bpy.types.Object
      Object to aim.
    target : bpy.types.Object
      What to aim it at.
    """
    constraint = obj.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"


def engine():
    """
    Pick a real-time render engine which exists in this Blender.

    Returns
    ----------
    name : str
      An identifier valid for `scene.render.engine`.
    """
    # EEVEE has been renamed more than once between versions
    available = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if name in available:
            return name
    return next(iter(available))


def render(path, output, resolution=(1280, 720), fps=30, samples=32):
    """
    Import an animated glTF file and render it.

    Parameters
    ------------
    path : str
      Source `.glb` or `.gltf` file.
    output : str
      Where to write, `.mp4` for a video or a prefix for PNG frames.
    resolution : (2,) int
      Width and height in pixels.
    fps : int
      Frames per second.
    samples : int
      Anti-aliasing samples per pixel.
    """
    clear()

    scene = bpy.context.scene
    # the frame rate has to be set before importing as the glTF
    # importer converts keyframe times from seconds into frames
    scene.render.fps = fps

    bpy.ops.import_scene.gltf(filepath=path)

    subjects = [o for o in scene.objects if o.type == "MESH"]
    if len(subjects) == 0:
        raise ValueError(f"no meshes found in `{path}`!")

    scene.frame_start = 1
    scene.frame_end = duration()

    # the resolution has to be set before framing as the camera is
    # fit against the aspect ratio of the image it will render
    scene.render.resolution_x, scene.render.resolution_y = resolution

    # fit the camera to the whole animation, not just the opening pose
    look_at(subjects, frames=(scene.frame_start, scene.frame_end))

    # a plain gray world so nothing renders against pure black
    world = bpy.data.worlds.new("world")
    if world.node_tree is None:
        # `use_nodes` is deprecated in newer versions where it is the default
        world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)
    scene.world = world

    scene.render.engine = engine()
    scene.render.film_transparent = False
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples

    video = output.lower().endswith(".mp4")
    if video:
        try:
            # not every Blender ships with video support compiled in, and
            # the advertised enum doesn't always match what is accepted
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format = "MPEG4"
            scene.render.ffmpeg.codec = "H264"
            scene.render.ffmpeg.constant_rate_factor = "HIGH"
        except TypeError:
            print("this Blender has no FFMPEG support, writing a PNG sequence instead")
            print(
                f"assemble it with: ffmpeg -framerate {fps}"
                + f" -i {output[:-4]}_%04d.png {output}"
            )
            output = f"{output[:-4]}_"
            video = False

    if not video:
        scene.render.image_settings.file_format = "PNG"

    # a relative path would be resolved against the blend file, which
    # doesn't exist when running headless from a script
    scene.render.filepath = os.path.abspath(output)

    print(f"rendering {scene.frame_end} frames to `{output}`")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    # blender passes script arguments after a bare `--`
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []

    if len(argv) < 1:
        raise ValueError(
            "usage: blender -b -P render_blender.py -- input.glb [output.mp4]"
        )

    render(argv[0], argv[1] if len(argv) > 1 else "render.mp4")
