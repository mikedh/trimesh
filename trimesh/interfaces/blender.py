import os
import platform

from .. import util
from ..resources import get_string
from ..typed import BooleanOperationType, Callable, Iterable, Union
from .generic import MeshScript


def _search_path() -> str:
    """
    Construct the search path for places Blender might be located.
    """
    if platform.system() == "Windows":
        # try to find Blender install on Windows
        # split existing path by delimiter
        values = [i for i in os.environ.get("PATH", "").split(";") if len(i) > 0]
        for pf in [r"C:\Program Files", r"C:\Program Files (x86)"]:
            pf = os.path.join(pf, "Blender Foundation")
            if os.path.exists(pf):
                for p in os.listdir(pf):
                    if "Blender" in p:
                        values.append(os.path.join(pf, p))
        return ";".join(set(values))

    elif platform.system() == "Darwin":
        # try to find Blender on Mac OSX
        values = [i for i in os.environ.get("PATH", "").split(":") if len(i) > 0]
        values.extend(
            [
                "/Applications/blender.app/Contents/MacOS",
                "/Applications/Blender.app/Contents/MacOS",
                "/Applications/Blender/blender.app/Contents/MacOS",
            ]
        )
        return ":".join(set(values))

    return os.environ.get("PATH", "")


class LazyWhich:
    def __init__(self, name: str, path: Union[str, None, Callable] = None):
        """
        Construct a lazy evaluator for `shutil.which(name, path=path)`.
        """
        self.name = name
        self.path = path

        self._evaluated = None
        self._evaluated_once = False

    def _value(self):
        if not self._evaluated_once:
            if callable(self.path):
                # path may be a lazy evaluated function
                path = self.path()
            else:
                path = self.path

            self._evaluated_once = True
            self._evaluated = util.which(self.name, path=path)
        return self._evaluated

    def __bool__(self) -> bool:
        return self._value() is not None

    def __str__(self) -> str:
        return self._value()


_blender_executable = LazyWhich(name="blender", path=_search_path)
exists = _blender_executable

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
