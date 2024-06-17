import os
import platform

from .. import resources, util
from ..constants import log
from ..typed import Iterable
from .generic import MeshScript

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


def boolean(
    meshes: Iterable,
    operation: str = "difference",
    use_exact: bool = False,
    use_self: bool = False,
    debug: bool = False,
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

    Returns
    ----------
    result
      The result of the boolean operation on the provided meshes.
    """
    if not exists:
        raise ValueError("No blender available!")
    operation = str.upper(operation)
    if operation == "INTERSECTION":
        operation = "INTERSECT"

    if use_exact:
        solver_options = "EXACT"
    else:
        solver_options = "FAST"
    # get the template from our resources folder
    template = resources.get_string("templates/blender_boolean.py.tmpl")
    script = template.replace("$OPERATION", operation)
    script = script.replace("$SOLVER_OPTIONS", solver_options)
    script = script.replace("$USE_SELF", f"{use_self}")

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
    template = resources.get_string("templates/blender_unwrap.py.template")
    script = template.replace("$ANGLE_LIMIT", f"{angle_limit:.6f}").replace(
        "$ISLAND_MARGIN", f"{island_margin:.6f}"
    )

    with MeshScript(meshes=[mesh], script=script, exchange="obj", debug=debug) as blend:
        result = blend.run(_blender_executable + " --background --python $SCRIPT")

    for m in util.make_sequence(result):
        # blender returns actively incorrect face normals
        m.face_normals = None

    return result
