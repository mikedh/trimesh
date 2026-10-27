import os
import platform
import subprocess

from .. import util
from ..constants import log
from ..resources import get_string
from ..typed import BooleanOperationType, Integer, Iterable, Number
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


def build(
    scene,
    directory: str,
    count: Integer,
    fps: Number,
    samples: Integer = 64,
    **settings,
) -> None:
    """
    Run `resources/templates/blender_scene.py` in Blender on a GLB of the
    scene, which renders `frame_*.png` files.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene from `trimesh.viewer.render.stage`.
    directory : str
      Where the GLB and any frames go.
    count, fps : int, float
      Frames of `scene.animations` every `1 / fps` seconds.
    samples : int
      Antialiasing samples per pixel.
    settings : dict
      Changes to `trimesh.viewer.conversion.RENDER_DEFAULTS`.
    """
    from ..viewer.conversion import RENDER_DEFAULTS

    unknown = sorted(set(settings).difference(RENDER_DEFAULTS))
    if len(unknown) > 0:
        raise TypeError(f"unknown render settings {unknown}")
    if not exists:
        raise ValueError("No blender available!")

    path = os.path.join(directory, "scene.glb")
    scene.export(path, include_normals=True)
    # the settings glTF can't say, like exposure and line widths, where a
    # pixel is about the scene's scale over the image's height
    arguments = {
        **RENDER_DEFAULTS,
        **settings,
        "input": path,
        "output": os.path.join(directory, "frame_"),
        "camera": scene.camera.name,
        "fps": fps,
        "count": count,
        "resolution": scene.camera.resolution,
        "samples": samples,
        "pixel": scene.scale / scene.camera.resolution[1],
    }
    # without `--python-exit-code` an error in the script still exits zero
    command = [_blender_executable, "--background", "--python-exit-code", "1"]
    command += ["--python-expr", get_string("templates/blender_scene.py")]
    command += ["--", util.jsonify(arguments)]
    result = subprocess.run(command, capture_output=True, text=True, **HIDDEN_CONSOLE)
    if result.returncode != 0:
        output = result.stdout + result.stderr
        raise RuntimeError(f"`{_blender_executable}` failed:\n{output}")
