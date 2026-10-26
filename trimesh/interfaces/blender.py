import glob
import os
import platform
import re
import subprocess
import tempfile

import numpy as np

from .. import util
from ..constants import log
from ..resources import get_string
from ..typed import BooleanOperationType, Iterable, Number
from .generic import HIDDEN_CONSOLE, MeshScript

if platform.system() == "Windows":
    # try to find Blender install on Windows
    # split existing path by delimiter
    _search_path = [i for i in os.environ.get("PATH", "").split(";") if len(i) > 0]
    for pf in [r"C:\Program Files", r"C:\Program Files (x86)"]:
        pf = os.path.join(pf, "Blender Foundation")
        if os.path.exists(pf):
            for p in os.listdir(pf):
                if "Blender" in p:
                    _search_path.append(os.path.join(pf, p))
    _search_path = ";".join(set(_search_path))
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
    _search_path = ":".join(set(_search_path))
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
            timeout=60.0,
            **HIDDEN_CONSOLE,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as E:
        raise RuntimeError(f"`{_blender_executable}` didn't run: {E}") from E
    match = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", output)
    if match is None:
        raise RuntimeError(f"`{_blender_executable}` isn't Blender:\n{output}")
    found = tuple(int(part) for part in match.groups())
    _BLENDER_VERSION[_blender_executable] = found
    return found


def build(
    scene,
    directory: str,
    count: int,
    fps: Number,
    samples: int,
    blend=None,
    **settings,
):
    """
    Run `resources/templates/blender_scene.py` in Blender on a GLB of the
    scene, which renders `frame_*.png` files or saves a `.blend`.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene from `trimesh.viewer.render.stage`.
    directory : str
      Where the GLB, settings and any frames go.
    count : int
      Frames of `scene.animations` every `1 / fps` seconds.
    fps : float
      Frames per second of animation.
    samples : int
      Antialiasing samples per pixel.
    blend : str or None
      Save a `.blend` to this path rather than rendering.
    settings : dict
      Changes to `trimesh.viewer.conversion.RENDER_DEFAULTS`, of which
      Blender reads what glTF can't say: the exposure, if the environment
      is seen, and how wide lines and points are.
    """
    from ..viewer.conversion import RENDER_DEFAULTS

    unknown = sorted(set(settings).difference(RENDER_DEFAULTS))
    if len(unknown) > 0:
        raise TypeError(f"unknown render settings {unknown}")
    settings = {**RENDER_DEFAULTS, **settings}
    if not exists:
        raise ValueError("No blender available!")
    found = version()
    if found < MINIMUM_VERSION:
        raise ValueError(
            f"rendering needs Blender >= {'.'.join(map(str, MINIMUM_VERSION))}, "
            + f"found {'.'.join(map(str, found))} at `{_blender_executable}`"
        )
    for geometry in scene.geometry.values():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        if getattr(material, "diffuseTransmissionFactor", None) is not None:
            log.warning("only the OpenGL viewer renders diffuse transmission")
            break

    path = os.path.join(directory, "scene.glb")
    scene.export(path, include_normals=True)
    # the importer converts seconds to frames with an integer `fps`
    # divided by `fps_base`, which is at least one
    whole_fps = max(round(fps), 1)
    width, height = scene.camera.resolution
    keys = ("exposure", "show_environment", "line_width", "point_size")
    settings = {key: settings[key] for key in keys}
    settings.update(
        input=path,
        output=os.path.join(directory, "frame_"),
        blend=blend,
        camera=scene.camera.name,
        fps=whole_fps,
        fps_base=whole_fps / fps,
        count=count,
        width=width,
        height=height,
        samples=samples,
        # a pixel is about the scene's scale over the image's height
        pixel=scene.scale / height,
    )
    settings_path = os.path.join(directory, "settings.json")
    with open(settings_path, "w") as file_obj:
        file_obj.write(util.jsonify(settings))
    script = os.path.join(directory, "blender_scene.py")
    with open(script, "w") as file_obj:
        file_obj.write(get_string("templates/blender_scene.py"))
    # without `--python-exit-code` an error in the script still exits zero
    result = subprocess.run(
        [_blender_executable, "--background", "--python-exit-code", "1"]
        + ["--python", script, "--", settings_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **HIDDEN_CONSOLE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"`{_blender_executable}` failed:\n{result.stdout}")


def render(scene, count: int = 1, fps: Number = 30.0, samples: int = 64, **settings):
    """
    Render a scene with Blender's EEVEE through its own camera.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene from `trimesh.viewer.render.stage` to render at
      `scene.camera.resolution`.
    count : int
      Frames of `scene.animations` every `1 / fps` seconds.
    fps : float
      Frames per second of animation.
    samples : int
      Antialiasing samples per pixel.
    settings : dict
      Changes to `trimesh.viewer.conversion.RENDER_DEFAULTS`.

    Returns
    ------------
    frames : generator
      (height, width, 4) uint8 RGBA images over white.
    """
    from PIL import Image

    with tempfile.TemporaryDirectory() as directory:
        build(scene, directory, count, fps, samples, **settings)
        paths = sorted(glob.glob(os.path.join(directory, "frame_*.png")))
        if len(paths) != count:
            raise RuntimeError(f"blender rendered {len(paths)} of {count} frames")
        # the background was transparent so put it over white
        white = Image.new("RGBA", tuple(scene.camera.resolution), (255,) * 4)
        for path in paths:
            with Image.open(path) as image:
                yield np.asarray(Image.alpha_composite(white, image.convert("RGBA")))


def save_blend(
    scene,
    resolution=None,
    duration: Number | None = None,
    fps: Number = 30.0,
    samples: int = 64,
    **settings,
) -> bytes:
    """
    The scene as a `.blend` which renders the frames `render` does.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to save, which is not changed.
    resolution : (2,) int or None
      Render size in pixels, defaults to `scene.camera.resolution`.
    duration : float or None
      Seconds in the frame range, defaults to `scene.duration` or
      a turntable if it isn't animated.
    fps : float
      Frames per second of animation.
    samples : int
      Antialiasing samples per pixel for rendering.
    settings : dict
      Changes to `trimesh.viewer.conversion.RENDER_DEFAULTS`.

    Returns
    ------------
    blend : bytes
      The `.blend` file with every image packed in.
    """
    from ..viewer.render import stage

    scene, count = stage(scene, resolution=resolution, duration=duration, fps=fps)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "scene.blend")
        build(scene, directory, count, fps, samples, blend=path, **settings)
        with open(path, "rb") as file_obj:
            return file_obj.read()
