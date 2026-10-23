import os
import platform

from .. import util
from ..constants import log
from ..resources import get_string
from ..typed import BooleanOperationType, Iterable, Number
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


def viewer_glb(scene, count: int):
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

    Returns
    ------------
    copied : trimesh.Scene
      Smooth normals split at sharp edges, the viewer's default material
      where there is none, glass as the transparency the viewer draws,
      and every animation as one which plays them all at once.
    """
    import numpy as np

    from ..base import Trimesh
    from ..visual.color import srgb_to_linear, to_float
    from ..visual.material import PBRMaterial
    from ..visual.texture import TextureVisuals

    scene = scene.copy()
    if count == 1:
        scene.animations = []
    # the viewer plays every animation at once where Blender plays one
    # glTF animation, which animations sharing a name export as
    for animation in scene.animations:
        animation.name = scene.animations[0].name
    for name, geometry in list(scene.geometry.items()):
        if not hasattr(geometry, "smooth_shaded"):
            continue
        smooth = geometry.smooth_shaded
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
        if smooth.visual.kind != "texture":
            # the viewer draws a mesh without a material as matte plastic
            # tinted by its sRGB colors, which glTF stores as linear
            colors = to_float(smooth.visual.vertex_colors)
            colors[:, :3] = srgb_to_linear(colors[:, :3])
            material = PBRMaterial(metallicFactor=0.0, roughnessFactor=0.5)
            if (colors[:, 3] < 1.0).any():
                material.alphaMode = "BLEND"
            smooth.visual = TextureVisuals(material=material)
            smooth.visual.vertex_attributes["color"] = np.round(colors * 255).astype(
                np.uint8
            )
        elif smooth.visual.material.to_pbr().transmissionFactor is not None:
            # EEVEE's refraction draws thin glass opaque, so glass is the
            # transparency the viewer draws which `ALPHA_MODES` in
            # `trimesh.viewer.conversion` explains
            glass = smooth.visual.material.to_pbr().copy()
            color = np.ones(4)
            if glass.baseColorFactor is not None:
                color = to_float(glass.baseColorFactor)
            color[3] *= 1.0 - glass.transmissionFactor
            glass.baseColorFactor = color
            glass.alphaMode = "BLEND"
            glass.transmissionFactor = None
            smooth.visual.material = glass
        scene.geometry[name] = smooth
    return scene


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
      The environment, floor, occlusion and line and point sizes the
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
    import json
    import subprocess
    from dataclasses import asdict

    import numpy as np

    if not exists:
        raise ValueError("No blender available!")
    scene = viewer_glb(scene, count=count)
    width, height = scene.camera.resolution
    path = os.path.join(directory, "scene.glb")
    scene.export(path, include_normals=True)
    settings = {
        "input": path,
        "output": os.path.join(directory, "frame_"),
        "blend": blend,
        # the camera to look through, by the node it's on, and its exact
        # fields of view in degrees
        "camera": scene.camera.name,
        "fov": scene.camera.fov.tolist(),
        # the importer converts seconds to frames with an integer `fps`
        # divided by `fps_base`
        "fps": round(fps),
        "fps_base": round(fps) / fps,
        "count": count,
        "width": int(width),
        "height": int(height),
        "samples": samples,
        "environment": asdict(studio.environment),
        "ground": None,
        "ground_light": studio.ground_light,
        "occlusion_radius": studio.occlusion_radius,
        "show_environment": studio.show_environment,
        # lines and points are this many pixels across, and a pixel is
        # this much bigger per unit of depth in front of the camera
        "line_width": studio.line_width,
        "point_size": studio.point_size,
        "pixel": 2.0 * np.tan(np.radians(scene.camera.fov[1]) / 2.0) / height,
    }
    if studio.ground is not None:
        settings["ground"] = studio.ground.tolist()
    settings_path = os.path.join(directory, "settings.json")
    with open(settings_path, "w") as file_obj:
        json.dump(settings, file_obj)
    script = os.path.join(directory, "blender_scene.py")
    with open(script, "w") as file_obj:
        file_obj.write(get_string("templates/blender_scene.py"))
    # without `--python-exit-code` an error in the script still exits zero
    result = subprocess.run(
        [_blender_executable, "--background", "--python-exit-code", "1"]
        + ["--python", script, "--", settings_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"blender failed:\n{result.stdout}{result.stderr}")


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
    frames : list
      (height, width, 4) uint8 RGBA images over white.
    """
    import glob
    import tempfile

    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as directory:
        build(scene, studio, directory, count=count, fps=fps, samples=samples)
        paths = sorted(glob.glob(os.path.join(directory, "frame_*.png")))
        if len(paths) != count:
            raise RuntimeError(f"blender rendered {len(paths)} of {count} frames")
        # the background was transparent so put it over white
        width, height = scene.camera.resolution
        white = Image.new("RGBA", (int(width), int(height)), (255, 255, 255, 255))
        frames = []
        for frame in paths:
            with Image.open(frame) as image:
                frames.append(
                    np.asarray(Image.alpha_composite(white, image.convert("RGBA")))
                )
        return frames


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
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "scene.blend")
        build(scene, studio, directory, count=count, fps=fps, samples=samples, blend=path)
        with open(path, "rb") as file_obj:
            return file_obj.read()
