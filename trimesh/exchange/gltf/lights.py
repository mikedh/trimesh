"""
lights.py
--------------

Read and write `KHR_lights_punctual` and `EXT_lights_image_based`.
Lights are named after their node.
"""

import numpy as np

from ...constants import log, tol
from ...scene import lighting
from ...scene.transforms import DEFAULT_BASE_FRAME
from ...transformations import quaternion_from_matrix, quaternion_matrix
from ...visual import color
from .animation import WXYZ_FROM_XYZW, XYZW_FROM_WXYZ

NAME = "KHR_lights_punctual"
CLASSES = {
    "directional": lighting.DirectionalLight,
    "point": lighting.PointLight,
    "spot": lighting.SpotLight,
}
ENVIRONMENT = "EXT_lights_image_based"
# Babylon.js `LinearEncodePowerApprox`
RGBD_GAMMA = 2.2


def parse_lights(header, names, scene_index, images, graph):
    """
    Load the punctual lights of every node and the scene environment.

    Parameters
    ------------
    header : dict
      Parsed GLTF header.
    names : dict
      {node index : node name}
    scene_index : int
      Index of the loaded scene.
    images : list or None
      `PIL.Image.Image` by index into `header["images"]`.
    graph : collections.deque
      `SceneGraph.update` kwargs, appended to in place.

    Returns
    ----------
    lights : list
      `trimesh.scene.lighting.Light` objects.
    """
    extensions = header.get("extensions", {})
    stored = extensions.get(NAME, {}).get("lights", [])
    lights = []
    for index, node in enumerate(header.get("nodes", [])):
        reference = node.get("extensions", {}).get(NAME, {}).get("light")
        if reference is None or index not in names:
            continue
        try:
            entry = stored[reference]
            lights.append(
                CLASSES[entry["type"]](
                    name=names[index],
                    intensity=entry.get("intensity", 1.0),
                    radius=entry.get("range"),
                    color=np.array(entry.get("color", [1.0, 1.0, 1.0]), dtype=np.float64),
                    **entry.get("spot", {}),
                )
            )
        except (KeyError, IndexError, TypeError, ValueError):
            if tol.strict:
                raise
            log.warning(f"unable to load the light of node `{index}`", exc_info=True)

    scene = header.get("scenes", [{}])[scene_index].get("extensions", {}).get(ENVIRONMENT)
    if scene is None:
        return lights
    try:
        entry = extensions[ENVIRONMENT]["lights"][scene["light"]]
        faces = [images[index].convert("RGBA") for index in entry["specularImages"][0]]
        environment = lighting.EnvironmentLight(
            name=entry.get("name"),
            intensity=entry.get("intensity"),
            # the spec flips each face horizontally
            cubemap=rgbd_unpack([np.asarray(face)[:, ::-1] for face in faces]),
        )
        if environment.name not in names.values():
            rotation = np.array(entry.get("rotation", [0.0, 0.0, 0.0, 1.0]))
            rotation = rotation[WXYZ_FROM_XYZW]
            graph.append(
                {
                    "frame_from": DEFAULT_BASE_FRAME,
                    "frame_to": environment.name,
                    "matrix": quaternion_matrix(rotation),
                }
            )
        lights.append(environment)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        if tol.strict:
            raise
        log.warning("unable to load environment light", exc_info=True)
    return lights


def append_lights(scene, tree, node_index, append_image):
    """
    Write the lights of a scene into a glTF tree in place.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to export.
    tree : dict
      GLTF header with `nodes` populated.
    node_index : dict
      {node name : index in `tree["nodes"]`}
    append_image : function
      Stores a `PIL.Image.Image` and returns its index.
    """
    types = {value: key for key, value in CLASSES.items()}
    stored, environments = [], []
    for light in scene.lights:
        if light.name not in node_index:
            log.warning(f"light `{light.name}` has no node!")
        elif isinstance(light, lighting.EnvironmentLight):
            environments.append(light)
        elif type(light) not in types:
            if tol.strict:
                raise ValueError(f"`{type(light).__name__}` has no glTF light type")
            log.warning(f"can't export `{light.name}`: no glTF light type")
        else:
            entry = {
                "name": light.name,
                "color": color.to_float(light.color)[:3].tolist(),
                "intensity": float(light.intensity),
                "type": types[type(light)],
            }
            if light.radius is not None:
                entry["range"] = float(light.radius)
            if entry["type"] == "spot":
                entry["spot"] = {
                    key: float(getattr(light, key))
                    for key in ("innerConeAngle", "outerConeAngle")
                }
            # copy so the source extensions aren't mutated
            node = tree["nodes"][node_index[light.name]]
            node["extensions"] = {
                **node.get("extensions", {}),
                NAME: {"light": len(stored)},
            }
            stored.append(entry)
    if len(stored) > 0:
        tree.setdefault("extensions", {})[NAME] = {"lights": stored}
    if len(environments) == 0:
        return
    from PIL import Image

    if len(environments) > 1:
        log.warning("exporting the first of several environment lights")
    light, levels = environments[0], lighting.prefilter(environments[0].cubemap)
    rotation = quaternion_from_matrix(scene.graph[light.name][0])[XYZW_FROM_WXYZ]
    entry = {
        "name": light.name,
        "intensity": float(light.intensity),
        "rotation": rotation.tolist(),
        "irradianceCoefficients": lighting.spherical_harmonics(light.cubemap).tolist(),
        "specularImageSize": len(levels[0][0]),
        # the spec flips each face horizontally
        "specularImages": [
            [append_image(Image.fromarray(rgbd_pack(face)[:, ::-1])) for face in level]
            for level in levels
        ],
    }
    tree.setdefault("extensions", {})[ENVIRONMENT] = {"lights": [entry]}
    tree["scenes"][0]["extensions"] = {ENVIRONMENT: {"light": 0}}


def rgbd_pack(radiance):
    """
    Pack linear radiance up to 255 as RGBD bytes.

    Parameters
    ------------
    radiance : (..., 3) float
      Linear RGB radiance.

    Returns
    ------------
    packed : (..., 4) uint8
      RGBD pixels.
    """
    radiance = np.clip(np.asarray(radiance, dtype=np.float64), 0.0, None)
    decoded = radiance
    for _ in range(2):
        # second pass from the decoded values so repacking is stable
        brightest = np.maximum(decoded.max(axis=-1, keepdims=True), 1.0)
        alpha = np.floor(255.0 / brightest * (1.0 + np.finfo(np.float32).eps))
        alpha = np.clip(alpha, 1.0, 255.0)
        encoded = np.minimum(radiance * alpha / 255.0, 1.0) ** (1.0 / RGBD_GAMMA)
        packed = np.concatenate((np.round(encoded * 255.0), alpha), axis=-1)
        decoded = rgbd_unpack(packed)
    return packed.astype(np.uint8)


def rgbd_unpack(packed):
    """
    Unpack RGBD pixels to linear radiance.

    Parameters
    ------------
    packed : (..., 4) uint8
      RGBD pixels.

    Returns
    ------------
    radiance : (..., 3) float
      Linear RGB radiance.
    """
    packed = np.asarray(packed, dtype=np.float64) / 255.0
    return packed[..., :3] ** RGBD_GAMMA / packed[..., 3:]
