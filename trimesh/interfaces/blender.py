import glob
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import asdict

import numpy as np

from .. import util
from ..constants import log
from ..resources import get_string
from ..typed import BooleanOperationType, Iterable, Number
from .generic import MeshScript

if platform.system() == "Windows":
    # try to find Blender install on Windows: the PATH first, then every
    # `Blender Foundation/Blender X.Y` install, newest first
    _search_path = [i for i in os.environ.get("PATH", "").split(";") if len(i) > 0]
    installs = []
    for program_files in [r"C:\Program Files", r"C:\Program Files (x86)"]:
        foundation = os.path.join(program_files, "Blender Foundation")
        if os.path.exists(foundation):
            installs.extend(
                os.path.join(foundation, name)
                for name in os.listdir(foundation)
                if "Blender" in name
            )
    installs.sort(
        key=lambda path: [int(number) for number in re.findall(r"\d+", path)],
        reverse=True,
    )
    # without duplicates, keeping the order
    _search_path = ";".join(dict.fromkeys(_search_path + installs))
    log.debug("searching for blender in: %s", _search_path)
elif platform.system() == "Darwin":
    # try to find Blender on Mac OSX
    _search_path = [i for i in os.environ.get("PATH", "").split(":") if len(i) > 0]
    _search_path.extend(
        [
            "/Applications/blender.app/Contents/MacOS",
            "/Applications/Blender.app/Contents/MacOS",
            "/Applications/Blender/blender.app/Contents/MacOS",
        ]
    )
    # without duplicates, keeping the order
    _search_path = ":".join(dict.fromkeys(_search_path))
    log.debug("searching for blender in: %s", _search_path)
else:
    _search_path = os.environ.get("PATH", "")

_blender_executable = util.which("blender", path=_search_path)
exists = _blender_executable is not None

# a map that translates:
# `trimesh.BooleanOperationType` -> blender value
_blender_bool = {
    "union": "UNION",
    "difference": "DIFFERENCE",
    "intersection": "INTERSECT",
}


def boolean(
    meshes: Iterable,
    operation: BooleanOperationType = "difference",
    use_exact: bool = True,
    use_self: bool = False,
    debug: bool = False,
    check_volume: bool = True,
):
    """
    Run a boolean operation with multiple meshes using Blender.

    Parameters
    -----------
    meshes
      List of mesh objects to be operated on
    operation
      Type of boolean operation ("difference", "union", "intersect").
    use_exact
      Use the "exact" mode as opposed to the "fast" mode.
    use_self
      Whether to consider self-intersections.
    debug
      Provide additional output for troubleshooting.
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.

    Returns
    ----------
    result
      The result of the boolean operation on the provided meshes.
    """
    if not exists:
        raise ValueError("No blender available!")
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")

    # conversions from the trimesh `BooleanOperationType` to the blender option
    key = operation.lower().strip()
    if key not in _blender_bool:
        raise ValueError(
            f"`{operation}` is not a valid boolean: `{_blender_bool.keys()}`"
        )

    if use_exact:
        solver_options = "EXACT"
    else:
        solver_options = "FAST"

    # get the template from our resources folder
    template = get_string("templates/blender_boolean.py.tmpl")
    # use string substitutions rather than `string.Template` as we aren't going
    # to be filling in all the values here, `MeshScript` is going to be
    # the source of `$MESH_PRE`, etc.
    script = (
        template.replace("$OPERATION", _blender_bool[key])
        .replace("$SOLVER_OPTIONS", solver_options)
        .replace("$USE_SELF", f"{use_self}")
    )
    with MeshScript(meshes=meshes, script=script, debug=debug) as blend:
        result = blend.run(_blender_executable + " --background --python $SCRIPT")

    result = util.make_sequence(result)
    for m in result:
        # blender returns actively incorrect face normals
        m.face_normals = None

    return util.concatenate(result)


def unwrap(
    mesh, angle_limit: float = 66.0, island_margin: float = 0.0, debug: bool = False
):
    """
    Run an unwrap operation using blender.
    """
    if not exists:
        raise ValueError("No blender available!")

    # get the template from our resources folder
    template = get_string("templates/blender_unwrap.py.template")
    script = template.replace("$ANGLE_LIMIT", f"{angle_limit:.6f}").replace(
        "$ISLAND_MARGIN", f"{island_margin:.6f}"
    )

    with MeshScript(meshes=[mesh], script=script, exchange="obj", debug=debug) as blend:
        result = blend.run(_blender_executable + " --background --python $SCRIPT")

    for m in util.make_sequence(result):
        # blender returns actively incorrect face normals
        m.face_normals = None

    return result


# the oldest Blender which `templates/blender_scene.py` runs in
MINIMUM_VERSION = (4, 3)

# what `blender --version` reported, by executable
_BLENDER_VERSION = {}

# Blender fades every light out towards its cutoff distance by
# `(1 - (distance / cutoff)^4)^2`, which is glTF's window for a `range`.
# A light without a `range` reaches everything, so its cutoff is far
# enough past the farthest point it reaches that the fade there is under
# half of an 8 bit level: `1 - (1 - x)^2 ~= 2x <= 1 / 510` for
# `x = (reach / cutoff)^4`, so `cutoff >= reach * 1020^(1/4)`
CUTOFF_PER_REACH = 1020.0**0.25

# the most bytes in a Blender name, where 4.3 stores 64 with a terminator
NAME_BYTES = 63


def version() -> tuple:
    """
    The version of the Blender being used, asked once.

    Returns
    ------------
    version : tuple
      Like `(4, 5, 15)`.
    """
    try:
        return _BLENDER_VERSION[_blender_executable]
    except KeyError:
        pass
    try:
        output = subprocess.run(
            [_blender_executable, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
    except OSError as E:
        raise RuntimeError(f"`{_blender_executable}` didn't run: {E}") from E
    match = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", output)
    if match is None:
        raise RuntimeError(
            f"`{_blender_executable} --version` didn't report a Blender "
            + f"version:\n{output}"
        )
    found = tuple(int(part) for part in match.groups())
    _BLENDER_VERSION[_blender_executable] = found
    return found


def spot_blend_scale() -> float:
    """
    How many times steeper Blender's spot light edge has to be than
    glTF's for the two to match closest. In terms of how far a direction's
    cosine is from the outer cone's to the inner's, glTF's falloff is
    `ramp^2`, and Blender's is a smoothstep of a ramp starting at the
    outer cone, which this scales to be closest to it: the scale with
    the smallest worst difference anywhere.

    Returns
    ------------
    scale : float
      The fit to a thousandth, 0.797.
    """
    # past `1 / 0.5` both are one for every scale tried
    ramp = np.linspace(0.0, 2.0, 2001)[:, None]
    scales = np.linspace(0.5, 1.0, 501)
    steeper = np.clip(ramp * scales, 0.0, 1.0)
    smoothstep = steeper * steeper * (3.0 - 2.0 * steeper)
    difference = np.abs(smoothstep - np.clip(ramp, 0.0, 1.0) ** 2)
    return float(scales[difference.max(axis=0).argmin()])


def viewer_glb(scene, count: int, fps: Number, directory: str):
    """
    A copy of a scene with what the OpenGL viewer draws written into it
    as glTF says it, so Blender's importer reads it the same way.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to copy.
    count : int
      Frames to render, where one is the current pose rather than the
      animation's first frame.
    fps : float
      Frames per second of animation.
    directory : str
      Where diffuse transmission textures are saved.

    Returns
    ------------
    copied : trimesh.Scene
      Smooth normals split at sharp edges, the viewer's default material
      where there is none, glass as the transparency the viewer draws,
      and one animation keyed on every frame which loops each animation
      past its own end as the viewer plays them.
    translucent : dict
      The `DIFFUSE_TRANSMISSION` of each material by name, with
      textures as the paths of PNG files.
    """
    from ..base import Trimesh
    from ..exchange.gltf import DIFFUSE_TRANSMISSION
    from ..scene.animation import Animation
    from ..viewer.conversion import MaterialUniforms
    from ..visual.color import srgb_to_linear, to_float
    from ..visual.material import PBRMaterial
    from ..visual.texture import TextureVisuals

    scene = scene.copy()
    times = np.arange(count) / fps
    # a later animation of an edge replaces an earlier as in `Scene.animate`
    poses = {}
    for animation in scene.animations:
        looped = times
        if animation.duration > 0.0:
            looped = times % animation.duration
        poses.update(zip(animation.edges, np.swapaxes(animation.at(looped), 0, 1)))
    scene.animations = []
    if count > 1 and len(poses) > 0:
        scene.animations.append(
            Animation(list(poses), times, np.stack(list(poses.values()), axis=1))
        )

    translucent = {}
    # textures by their pixels to where they were saved, as geometry
    # sharing a texture has been copied apart
    saved = {}
    for name, geometry in list(scene.geometry.items()):
        try:
            smooth = geometry.smooth_shaded
        except AttributeError:
            # points and paths, which the template draws
            continue
        if smooth.visual.kind == "face":
            # glTF colors vertices, so every triangle gets its own to keep
            # its color, with the same smooth normals
            corners = smooth.faces.reshape(-1)
            flat = Trimesh(
                vertices=smooth.vertices[corners],
                faces=np.arange(len(corners)).reshape((-1, 3)),
                vertex_normals=smooth.vertex_normals[corners],
                process=False,
            )
            flat.visual.vertex_colors = np.repeat(smooth.visual.face_colors, 3, axis=0)
            smooth = flat
        if smooth.visual.kind == "texture":
            material = smooth.visual.material.to_pbr().copy()
        else:
            # the viewer draws a mesh without a material as matte plastic
            # tinted by its sRGB colors, which glTF stores as linear
            colors = to_float(smooth.visual.vertex_colors)
            colors[:, :3] = srgb_to_linear(colors[:, :3])
            plastic = MaterialUniforms()
            material = PBRMaterial(
                metallicFactor=plastic.metallic, roughnessFactor=plastic.roughness
            )
            if (colors[:, 3] < 1.0).any():
                material.alphaMode = "BLEND"
            smooth.visual = TextureVisuals(material=material)
            smooth.visual.vertex_attributes["color"] = np.round(colors * 255).astype(
                np.uint8
            )
        if material.transmissionFactor is not None:
            # EEVEE's refraction draws thin glass opaque, so glass is the
            # transparency the viewer draws which `ALPHA_MODES` in
            # `trimesh.viewer.conversion` explains
            color = np.ones(4)
            if material.baseColorFactor is not None:
                color = to_float(material.baseColorFactor)
            color[3] *= 1.0 - material.transmissionFactor
            material.baseColorFactor = color
            material.alphaMode = "BLEND"
            material.transmissionFactor = None
        if material.diffuseTransmissionFactor is not None:
            # Blender's importer ignores the extension so the template is
            # passed it under a name Blender keeps whole, which is unique by
            # its index and still says what it was
            unique = f"translucent_{len(translucent)}_{material.name or ''}"
            material.name = unique.encode("utf-8")[:NAME_BYTES].decode(
                "utf-8", errors="ignore"
            )
            extension = {key: getattr(material, key) for key in DIFFUSE_TRANSMISSION}
            for key in DIFFUSE_TRANSMISSION[2:]:
                image = extension[key]
                if image is None:
                    continue
                pixels = (image.mode, image.size, image.tobytes())
                if pixels not in saved:
                    saved[pixels] = os.path.join(directory, f"texture_{len(saved)}.png")
                    image.save(saved[pixels])
                extension[key] = saved[pixels]
            translucent[material.name] = extension
        smooth.visual.material = material
        scene.geometry[name] = smooth
    return scene, translucent


def build(
    scene, studio, directory: str, count: int, fps: Number, samples: int, blend=None
) -> None:
    """
    Run `resources/templates/blender_scene.py` in Blender, which imports
    a GLB of the scene and either renders it to `frame_*.png` files in
    `directory` or saves it as a `.blend`.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to build, through `scene.camera` at `scene.camera.resolution`.
    studio : trimesh.scene.metadata.Studio
      The environment, floor, occlusion, exposure and line and point sizes the
      OpenGL viewer draws, which aren't in a glTF.
    directory : str
      Where the GLB, settings and any frames go.
    count : int
      One frame of the current pose, or this many frames of
      `scene.animations` every `1 / fps` seconds.
    fps : float
      Frames per second of animation.
    samples : int
      Antialiasing samples per pixel, which is where the time goes.
    blend : str or None
      Save a `.blend` to this path rather than rendering.
    """
    if not exists:
        raise ValueError("No blender available!")
    found = version()
    if found < MINIMUM_VERSION:
        raise ValueError(
            f"rendering needs Blender >= {'.'.join(map(str, MINIMUM_VERSION))}, "
            + f"found {'.'.join(map(str, found))} at `{_blender_executable}`"
        )
    scene, translucent = viewer_glb(scene, count=count, fps=fps, directory=directory)
    width, height = scene.camera.resolution
    path = os.path.join(directory, "scene.glb")
    scene.export(path, include_normals=True)
    # a glTF light without a `range` reaches everything, which is at most
    # across the bounds of the geometry and the lights in any frame, posed
    # at each frame's time which `viewer_glb` keyed exactly
    points = []
    for time in np.arange(count) / fps:
        scene.animate(time)
        points.append(scene.bounds)
        points.extend(scene.graph[light.name][0][:3, 3] for light in scene.lights)
    reach = np.linalg.norm(np.ptp(np.vstack(points), axis=0))
    # the importer converts seconds to frames with an integer `fps`
    # divided by `fps_base`, which is at least one
    whole_fps = max(round(fps), 1)
    settings = {
        "input": path,
        "output": os.path.join(directory, "frame_"),
        "blend": blend,
        # the camera to look through, by the node it's on, and its exact
        # fields of view in degrees
        "camera": scene.camera.name,
        "fov": scene.camera.fov,
        "fps": whole_fps,
        "fps_base": whole_fps / fps,
        "count": count,
        "width": width,
        "height": height,
        "samples": samples,
        "environment": asdict(studio.environment),
        "ground": studio.ground,
        "ground_light": studio.ground_light,
        "occlusion_radius": studio.occlusion_radius,
        "show_environment": studio.show_environment,
        "exposure": studio.exposure,
        # lines and points are this many pixels across, and a pixel is
        # this much bigger per unit of depth in front of the camera
        "line_width": studio.line_width,
        "point_size": studio.point_size,
        "pixel": 2.0 * np.tan(np.radians(scene.camera.fov[1]) / 2.0) / height,
        "ranges": {light.name: light.radius for light in scene.lights if light.radius},
        "cutoff": reach * CUTOFF_PER_REACH,
        "spot_blend_scale": spot_blend_scale(),
        "translucent": translucent,
    }
    settings_path = os.path.join(directory, "settings.json")
    with open(settings_path, "w") as file_obj:
        # numpy arrays and scalars as lists and numbers
        file_obj.write(util.jsonify(settings))
    script = os.path.join(directory, "blender_scene.py")
    with open(script, "w") as file_obj:
        file_obj.write(get_string("templates/blender_scene.py"))
    # without `--python-exit-code` an error in the script still exits zero
    # with its log and errors in the order Blender wrote them
    result = subprocess.run(
        [_blender_executable, "--background", "--python-exit-code", "1"]
        + ["--python", script, "--", settings_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"`{_blender_executable}` failed:\n{result.stdout}")


def render(scene, studio, count: int = 1, fps: Number = 30.0, samples: int = 64):
    """
    Render a scene with Blender's EEVEE through its own camera.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to render at `scene.camera.resolution`.
    studio : trimesh.scene.metadata.Studio
      The environment, floor and occlusion the OpenGL viewer draws,
      which aren't in a glTF so Blender is told them directly.
    count : int
      One frame of the current pose, or this many frames of
      `scene.animations` every `1 / fps` seconds.
    fps : float
      Frames per second of animation.
    samples : int
      Antialiasing samples per pixel, which is where the time goes.

    Returns
    ------------
    frames : generator
      (height, width, 4) uint8 RGBA images over white, read one at a
      time from the files Blender wrote.
    """
    from PIL import Image

    with tempfile.TemporaryDirectory() as directory:
        build(scene, studio, directory, count=count, fps=fps, samples=samples)
        paths = sorted(glob.glob(os.path.join(directory, "frame_*.png")))
        if len(paths) != count:
            raise RuntimeError(f"blender rendered {len(paths)} of {count} frames")
        # the background was transparent so put it over white
        white = Image.new("RGBA", tuple(scene.camera.resolution), (255,) * 4)
        for path in paths:
            with Image.open(path) as image:
                yield np.asarray(Image.alpha_composite(white, image.convert("RGBA")))


def save_blend(
    scene, studio, count: int = 1, fps: Number = 30.0, samples: int = 64
) -> bytes:
    """
    The scene as a `.blend` which renders what `render` does when opened
    in Blender, with its animation, camera, lights, environment and
    every image packed in.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to save, through `scene.camera` at `scene.camera.resolution`.
    studio : trimesh.scene.metadata.Studio
      The environment, floor and occlusion the OpenGL viewer draws.
    count : int
      Frames in the file's frame range.
    fps : float
      Frames per second of animation.
    samples : int
      Antialiasing samples per pixel for rendering.

    Returns
    ------------
    blend : bytes
      The `.blend` file.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "scene.blend")
        build(scene, studio, directory, count=count, fps=fps, samples=samples, blend=path)
        with open(path, "rb") as file_obj:
            return file_obj.read()
