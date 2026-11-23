"""
lights.py
--------------

Read and write `KHR_lights_punctual` and `EXT_lights_image_based`.
Lights are named after their node.
"""

from collections.abc import Callable

import numpy as np

from ...constants import log, tol
from ...scene import lighting
from ...scene.transforms import DEFAULT_BASE_FRAME
from ...transformations import quaternion_from_matrix, quaternion_matrix
from ...typed import ArrayLike, Integer, NDArray
from ...visual import color
from .animation import WXYZ_FROM_XYZW, XYZW_FROM_WXYZ

NAME = "KHR_lights_punctual"
CLASSES = {
    "directional": lighting.DirectionalLight,
    "point": lighting.PointLight,
    "spot": lighting.SpotLight,
}
TYPES = {value: key for key, value in CLASSES.items()}
SPOT = ("innerConeAngle", "outerConeAngle")
ENVIRONMENT = "EXT_lights_image_based"
# Babylon.js `LinearEncodePowerApprox`
RGBD_GAMMA = 2.2


def parse_lights(header: dict, names: dict) -> list:
    """
    Load the punctual lights of the loaded nodes.

    Parameters
    ------------
    header : dict
      Parsed GLTF header.
    names : dict
      {node index : node name}

    Returns
    ----------
    lights : list
      `trimesh.scene.lighting.Light` objects.
    """
    # {node name : its `KHR_lights_punctual`}
    nodes = {
        names[index]: node["extensions"][NAME]
        for index, node in enumerate(header.get("nodes", []))
        if index in names and NAME in node.get("extensions", {})
    }
    try:
        stored = header.get("extensions", {}).get(NAME, {}).get("lights", [])
        entries = {name: stored[node["light"]] for name, node in nodes.items()}
        return [
            CLASSES[entry["type"]](
                name=name,
                # a light without a color is white
                color=entry.get("color", [1.0] * 3),
                intensity=entry.get("intensity"),
                radius=entry.get("range"),
                **entry.get("spot", {}),
            )
            for name, entry in entries.items()
        ]
    except (KeyError, IndexError, TypeError, ValueError):
        if tol.strict:
            raise
        log.warning("unable to load lights", exc_info=True)
        return []


def parse_environment(
    header: dict, names: dict, scene_index: Integer, parse_images: Callable
) -> tuple[list, list]:
    """
    Load the `EXT_lights_image_based` light of the loaded scene.

    Parameters
    ------------
    header : dict
      Parsed GLTF header.
    names : dict
      {node index : node name}
    scene_index : int
      Index of the loaded scene.
    parse_images : function
      Returns `PIL.Image.Image` by index into `header["images"]`.

    Returns
    ----------
    lights : list
      The `EnvironmentLight` if the scene has one.
    edges : list
      `SceneGraph.update` kwargs placing it, unless a node of its name does.
    """
    scene = header.get("scenes", [{}])[scene_index].get("extensions", {})
    if ENVIRONMENT not in scene:
        return [], []
    try:
        entry = header["extensions"][ENVIRONMENT]["lights"][scene[ENVIRONMENT]["light"]]
        images = parse_images()
        faces = [images[index].convert("RGBA") for index in entry["specularImages"][0]]
        light = lighting.EnvironmentLight(
            name=entry.get("name"),
            intensity=entry.get("intensity"),
            # the spec flips each face horizontally
            cubemap=rgbd_unpack([np.asarray(face)[:, ::-1] for face in faces]),
        )
        rotation = np.array(entry.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        edge = {
            "frame_from": DEFAULT_BASE_FRAME,
            "frame_to": light.name,
            "matrix": quaternion_matrix(rotation[WXYZ_FROM_XYZW]),
        }
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        if tol.strict:
            raise
        log.warning("unable to load environment light", exc_info=True)
        return [], []
    if light.name in names.values():
        # a node of the same name already places it
        return [light], []
    return [light], [edge]


def append_lights(scene, tree: dict, node_index: dict, append_image: Callable) -> None:
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
    # lights on an exported node of a type glTF has
    kinds = (*TYPES, lighting.EnvironmentLight)
    placed = [i for i in scene.lights if i.name in node_index and type(i) in kinds]
    skipped = [i.name for i in scene.lights if i not in placed]
    if len(skipped) > 0:
        log.warning(f"can't export lights {skipped}: no node or glTF light type")
    environments = [i for i in placed if isinstance(i, lighting.EnvironmentLight)]
    punctual = [light for light in placed if type(light) in TYPES]
    # `KHR_lights_punctual` entries without the fields a light doesn't have
    stored = [
        {
            key: value
            for key, value in {
                "name": light.name,
                "type": TYPES[type(light)],
                "color": color.to_float(light.color)[:3].tolist(),
                "intensity": float(light.intensity),
                "range": light.radius,
                "spot": {k: float(getattr(light, k)) for k in SPOT if hasattr(light, k)},
            }.items()
            if value is not None and value != {}
        }
        for light in punctual
    ]
    for index, light in enumerate(punctual):
        # copy so the source extensions aren't mutated
        node = tree["nodes"][node_index[light.name]]
        node["extensions"] = {**node.get("extensions", {}), NAME: {"light": index}}
    if len(stored) > 0:
        tree.setdefault("extensions", {})[NAME] = {"lights": stored}
    if len(environments) == 0:
        return
    from PIL import Image

    if len(environments) > 1:
        log.warning("exporting the first of several environment lights")
    light = environments[0]
    rotation = quaternion_from_matrix(scene.graph[light.name][0])[XYZW_FROM_WXYZ]
    entry = {
        "name": light.name,
        "intensity": float(light.intensity),
        "rotation": rotation.tolist(),
        "irradianceCoefficients": light.irradiance.tolist(),
        "specularImageSize": len(light.levels[0][0]),
        # the spec flips each face horizontally
        "specularImages": [
            [append_image(Image.fromarray(rgbd_pack(face)[:, ::-1])) for face in level]
            for level in light.levels
        ],
    }
    tree.setdefault("extensions", {})[ENVIRONMENT] = {"lights": [entry]}
    tree["scenes"][0]["extensions"] = {ENVIRONMENT: {"light": 0}}


def rgbd_pack(radiance: ArrayLike) -> NDArray:
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


def rgbd_unpack(packed: ArrayLike) -> NDArray:
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
