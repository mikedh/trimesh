"""
render_blender.py
-------------------

Render an animated glTF file headlessly with Blender.

Run it with python and it finds Blender and re-launches itself inside it:

    uv run examples/render_blender.py cycloidal.glb out.mp4

which is exactly the same as handing the file over yourself:

    blender -b -P examples/render_blender.py -- cycloidal.glb out.mp4

The half that runs inside Blender deliberately does not import trimesh:
Blender ships its own Python and there is no reliable way to install into
it. Everything a render needs is in the glTF file already, so that half
only has to set up the render. Generate an input with
`examples/animation_cycloidal.py`.

Produces an MP4 if the output name ends in `.mp4`, otherwise a
numbered PNG sequence.
"""

import math
import os
import subprocess
import sys
from dataclasses import dataclass

# note `bpy` is imported inside each function that needs it rather than
# here: it only exists inside Blender, and the launcher at the bottom of
# this file has to be importable by a normal python to do its job


@dataclass
class Preset:
    """
    How much time to spend on a render.

    Almost everything here is close to free and only `samples` really
    costs: at 4K a frame is 9.7 seconds at 128 samples and 37.8 at 512,
    for a mean pixel difference of 0.04 out of 255. Turning fast GI off
    entirely saves one second in forty.

    Attributes
    ------------
    resolution : (2,) int
      Width and height in pixels.
    samples : int
      Anti-aliasing samples per pixel, which is where the time goes.
    trace_scale : str
      Resolution divisor for ray tracing, "1" being full resolution.
    fast_gi : bool
      Fill the shadowed side of surfaces with approximate bounce light.
    shadow_rays : int
      Rays per shadow, up to 4.
    shadow_steps : int
      Steps along each shadow ray, up to 16.
    """

    resolution: tuple = (3840, 2160)
    samples: int = 128
    trace_scale: str = "1"
    fast_gi: bool = True
    shadow_rays: int = 4
    shadow_steps: int = 16


# what to hand someone: 4K with everything which is nearly free turned up
FINAL = Preset()

# and one to iterate on, which trades everything for about half a second
# a frame so a six second shot previews in a minute and a half
PREVIEW = Preset(
    resolution=(1920, 1080),
    samples=16,
    trace_scale="2",
    fast_gi=False,
    shadow_rays=1,
    shadow_steps=4,
)

PRESETS = {"final": FINAL, "preview": PREVIEW}


def clear():
    """
    Delete everything in the startup scene.
    """
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)


def duration():
    """
    Find the last frame of any imported animation.
    """
    import bpy

    end = 1.0
    for action in bpy.data.actions:
        # `frame_range` is in frames, already converted from the
        # glTF times using the scene frame rate at import
        end = max(end, action.frame_range[1])
    return math.ceil(end)


def prefer(target, key, *values):
    """
    Set an enum to the first of `values` this Blender actually has.

    Enum spellings move between versions and some are missing entirely,
    so asking for a list in preference order and taking what is there
    beats hard-coding one name and crashing on the version which renamed
    it. Leaves the property alone if none of them exist.
    """
    prop = target.bl_rna.properties.get(key)
    if prop is None:
        return None

    available = {item.identifier for item in getattr(prop, "enum_items", ())}
    for value in values:
        if value in available:
            setattr(target, key, value)
            return value
    return None


def engine():
    """
    Pick a real-time render engine which exists in this Blender.
    """
    import bpy

    # EEVEE has been renamed more than once between versions
    available = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if name in available:
            return name
    return next(iter(available))


def quality(scene, preset=FINAL):
    """
    Turn on the settings which make glossy materials look like glossy
    materials, skipping any this version of Blender doesn't have.
    """
    scene.render.engine = engine()
    scene.render.film_transparent = False

    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        # without ray tracing a smooth metal has nothing to reflect
        for key, value in (
            ("taa_render_samples", preset.samples),
            ("use_raytracing", True),
            ("use_shadows", True),
            ("shadow_ray_count", preset.shadow_rays),
            ("shadow_step_count", preset.shadow_steps),
            ("shadow_resolution_scale", 1.0),
            # fast GI is what fills the shadowed side of a part
            ("use_fast_gi", preset.fast_gi),
            ("fast_gi_ray_count", 16),
            ("fast_gi_step_count", 64),
            ("fast_gi_quality", 1.0),
        ):
            if hasattr(eevee, key):
                setattr(eevee, key, value)

        # the enums separately, as their spellings move between versions
        prefer(eevee, "ray_tracing_method", "SCREEN")
        prefer(eevee, "fast_gi_method", "GLOBAL_ILLUMINATION")
        prefer(eevee, "shadow_pool_size", "1024", "512")
        # the resolution divisors: "1" is full resolution
        prefer(eevee, "fast_gi_resolution", "1")

        tracing = getattr(eevee, "ray_tracing_options", None)
        if tracing is not None:
            for key, value in (
                ("use_denoise", True),
                ("denoise_spatial", True),
                ("denoise_temporal", True),
                ("denoise_bilateral", True),
                ("screen_trace_quality", 1.0),
                # keep tracing all the way into rough surfaces rather
                # than falling back to a probe halfway
                ("trace_max_roughness", 1.0),
            ):
                if hasattr(tracing, key):
                    setattr(tracing, key, value)
            # the default is half resolution, which a final render can
            # afford to turn off and a preview very much cannot
            prefer(tracing, "resolution_scale", preset.trace_scale)

    # a filmic view transform is the difference between "rendered in a
    # spreadsheet" and "photographed", but the color management config
    # isn't loadable in every install so none of this can be fatal
    for key, value in (("view_transform", "AgX"), ("look", "AgX - Punchy")):
        try:
            setattr(scene.view_settings, key, value)
        except TypeError:
            pass


def render(path, output, preset=FINAL, fps=30):
    """
    Import an animated glTF file and render it.
    """
    import bpy

    clear()

    scene = bpy.context.scene
    # the frame rate has to be set before importing as the glTF
    # importer converts keyframe times from seconds into frames
    scene.render.fps = fps
    scene.render.resolution_x, scene.render.resolution_y = preset.resolution

    bpy.ops.import_scene.gltf(filepath=path)

    # the file carries its own camera, which is the whole point: it was
    # placed and animated against the same scene graph the geometry is in
    camera = next((o for o in scene.objects if o.type == "CAMERA"), None)
    if camera is None:
        raise ValueError(
            f"`{path}` has no camera! generate one with `animation_cycloidal.py`"
        )
    scene.camera = camera

    if not any(o.type == "LIGHT" for o in scene.objects):
        print("warning: file has no lights, the render will be black")

    scene.frame_start = 1
    scene.frame_end = duration()

    quality(scene, preset=preset)

    # tag the file with what it was rendered at, so a preview and a final
    # sit next to each other for comparison rather than overwriting
    root, extension = os.path.splitext(output)
    output = f"{root}_{preset.resolution[0]}x{preset.resolution[1]}{extension}"

    video = output.lower().endswith(".mp4")
    if video:
        settings = scene.render.image_settings
        # Blender 5.0 only offers a video format once the settings are
        # switched over to video, and older versions have no such
        # property: without this `file_format` rejects "FFMPEG" and
        # looks exactly like a build with no video support at all
        prefer(settings, "media_type", "VIDEO")
        settings.file_format = "FFMPEG"

        # say as little as possible past here. the container and codec
        # do have to be named as the defaults are MPEG1 and no codec,
        # but the rest are quality knobs Blender can pick for itself
        prefer(scene.render.ffmpeg, "format", "MPEG4", "MKV", "QUICKTIME")
        prefer(scene.render.ffmpeg, "codec", "H264", "MPEG4")
        prefer(scene.render.ffmpeg, "constant_rate_factor", "PERC_LOSSLESS", "HIGH")
    else:
        scene.render.image_settings.file_format = "PNG"

    # a relative path would be resolved against the blend file, which
    # doesn't exist when running headless from a script
    scene.render.filepath = os.path.abspath(output)

    print(f"rendering {scene.frame_end} frames to `{output}`")
    bpy.ops.render.render(animation=True)


def launch(source, output, preset="final"):
    """
    Find Blender and run this script inside it.
    """
    # trimesh already knows where Blender puts itself on every platform,
    # which is more than `shutil.which` will find on Windows or macOS.
    # this import is safe here as the launcher runs in a normal python
    from trimesh.interfaces.blender import _blender_executable

    if _blender_executable is None:
        raise ValueError("no `blender` found, is it installed and in your PATH?")

    subprocess.check_call(
        [_blender_executable, "--background", "--python", os.path.abspath(__file__)]
        # Blender hands everything after a bare `--` to the script
        + ["--", os.path.abspath(source), os.path.abspath(output), preset]
    )


if __name__ == "__main__":
    # Blender puts a bare `--` before the arguments meant for the script,
    # so its absence means this was run by python and has no Blender yet
    inside = "--" in sys.argv
    argv = sys.argv[sys.argv.index("--") + 1 :] if inside else sys.argv[1:]

    if len(argv) < 1:
        raise ValueError(
            "usage: python render_blender.py input.glb [output.mp4] "
            + f"[{'|'.join(PRESETS)}]"
        )

    chosen = argv[2] if len(argv) > 2 else "final"
    if chosen not in PRESETS:
        raise ValueError(f"`{chosen}` is not one of {sorted(PRESETS)}")

    source = argv[0]
    output = argv[1] if len(argv) > 1 else "render.mp4"

    if inside:
        render(source, output, preset=PRESETS[chosen])
    else:
        launch(source, output, chosen)
