"""
lights.py
--------------

Read and write the `KHR_lights_punctual` glTF extension, and an
`EnvironmentLight` as the `EXT_lights_image_based` light of the scene.
Every light is named after the node carrying it, so `Scene.lights` and
the scene graph refer to the same thing.
"""

import numpy as np

from ...constants import log, tol
from ...scene import lighting
from ...scene.transforms import DEFAULT_BASE_FRAME
from ...transformations import quaternion_from_matrix, quaternion_matrix
from ...visual import color

# the extension key, and the trimesh light class of each glTF light type
NAME = "KHR_lights_punctual"
CLASSES = {
    "directional": lighting.DirectionalLight,
    "point": lighting.PointLight,
    "spot": lighting.SpotLight,
}
# the extension key of the image based light a scene can have
ENVIRONMENT = "EXT_lights_image_based"
# RGBD stores radiance to this power, Babylon.js's `LinearEncodePowerApprox`
RGBD_GAMMA = 2.2


def parse_lights(header, names, scene_index, images, graph):
    """
    Collect the `KHR_lights_punctual` lights nodes carry, one per node
    as two nodes may reference one entry of the document's array, then
    the `EXT_lights_image_based` light of the loaded scene, which is its
    mirror level as the rest is recomputed from it.

    Parameters
    ------------
    header : dict
      Parsed GLTF header.
    names : dict
      Mapping of {node index in the file : node name in the scene}
    scene_index : int
      Which of the header's scenes is loaded.
    images : list or None
      `PIL.Image.Image` by index into `header["images"]`.
    graph : collections.deque
      `SceneGraph.update` kwargs of the loaded scene, which gets a node at
      the extension's `rotation` for an environment no node carries.

    Returns
    ----------
    lights : list
      One `trimesh.scene.lighting.Light` per node reference in file
      order, then the `EnvironmentLight` if the scene has one.
    """
    extensions = header.get("extensions", {})
    stored = extensions.get(NAME, {}).get("lights", [])
    lights = []
    for index, node in enumerate(header.get("nodes", [])):
        reference = node.get("extensions", {}).get(NAME, {}).get("light")
        if reference is None:
            continue
        try:
            entry = stored[reference]
            lights.append(
                CLASSES[entry["type"]](
                    name=names[index],
                    intensity=entry.get("intensity", 1.0),
                    radius=entry.get("range"),
                    # glTF lights are linear RGB, white unless they say otherwise
                    color=np.array(entry.get("color", [1.0, 1.0, 1.0]), dtype=np.float64),
                    # only a spot has a cone, defaulting as the extension does
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
            # each face is flipped about its vertical axis as the spec says
            cubemap=rgbd_unpack([np.asarray(face)[:, ::-1] for face in faces]),
        )
        if environment.name not in {edge["frame_to"] for edge in graph}:
            # glTF quaternions are `x, y, z, w` and trimesh's `w, x, y, z`
            rotation = np.roll(entry.get("rotation", [0.0, 0.0, 0.0, 1.0]), 1)
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
    Write the lights of a scene into a tree in place: the light array of
    `KHR_lights_punctual` and every node's reference into it, and the
    first `EnvironmentLight` as the `EXT_lights_image_based` light of the
    tree's scene with its node's world rotation as `rotation`.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene whose lights are its own.
    tree : dict
      GLTF header, mutated in place, with `nodes` already populated.
    node_index : dict
      Mapping of {node name : index in `tree["nodes"]`}
    append_image : function
      Stores a `PIL.Image.Image` and returns its index in `tree["images"]`.
    """
    from PIL import Image

    types = {value: key for key, value in CLASSES.items()}
    stored, environments = [], []
    for light in scene.lights:
        if light.name not in node_index:
            log.warning(f"light `{light.name}` has no node!")
        elif isinstance(light, lighting.EnvironmentLight):
            environments.append(light)
        elif type(light) in types:
            entry = {
                "name": light.name,
                # the extension stores linear RGB in the 0.0 - 1.0 range
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
            # the node holding the light points at it, merged into a copy of
            # any extensions the node already has so the scene isn't changed
            node = tree["nodes"][node_index[light.name]]
            node["extensions"] = {
                **node.get("extensions", {}),
                NAME: {"light": len(stored)},
            }
            stored.append(entry)
    if len(stored) > 0:
        tree["extensions"] = {**tree.get("extensions", {}), NAME: {"lights": stored}}
    if len(environments) == 0:
        return
    if len(environments) > 1:
        log.warning("exporting the first of several environment lights")
    light, levels = environments[0], lighting.prefilter(environments[0].cubemap)
    # glTF quaternions are `x, y, z, w` and trimesh's `w, x, y, z`
    rotation = np.roll(quaternion_from_matrix(scene.graph[light.name][0]), -1)
    entry = {
        "name": light.name,
        "intensity": float(light.intensity),
        "rotation": rotation.tolist(),
        "irradianceCoefficients": lighting.spherical_harmonics(light.cubemap).tolist(),
        "specularImageSize": len(levels[0][0]),
        # each face flipped about its vertical axis as the spec says
        "specularImages": [
            [append_image(Image.fromarray(rgbd_pack(face)[:, ::-1])) for face in level]
            for level in levels
        ],
    }
    tree["extensions"] = {**tree.get("extensions", {}), ENVIRONMENT: {"lights": [entry]}}
    tree["scenes"][0]["extensions"] = {ENVIRONMENT: {"light": 0}}


def rgbd_pack(radiance):
    """
    Pack linear radiance up to 255 as RGBD bytes: alpha `D` is
    `floor(255 / max(1, brightest))` and each channel is
    `(radiance * D / 255) ** (1 / RGBD_GAMMA)`, with `D` picked again from
    the first packing decoded so repacking what was unpacked is the same.

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
        # the most alpha the brightest channel fits a byte with, which
        # float32 rounding of a decoded `255 / alpha` doesn't lower
        brightest = np.maximum(decoded.max(axis=-1, keepdims=True), 1.0)
        alpha = np.floor(255.0 / brightest * (1.0 + np.finfo(np.float32).eps))
        alpha = np.clip(alpha, 1.0, 255.0)
        encoded = np.minimum(radiance * alpha / 255.0, 1.0) ** (1.0 / RGBD_GAMMA)
        packed = np.concatenate((np.round(encoded * 255.0), alpha), axis=-1)
        decoded = rgbd_unpack(packed)
    return packed.astype(np.uint8)


def rgbd_unpack(packed):
    """
    Linear radiance from RGBD pixels, the inverse of `rgbd_pack`.

    Parameters
    ------------
    packed : (..., 4) uint8
      RGBD pixels, where LDR pixels have an alpha of 255.

    Returns
    ------------
    radiance : (..., 3) float
      Linear RGB radiance.
    """
    packed = np.asarray(packed, dtype=np.float64) / 255.0
    return packed[..., :3] ** RGBD_GAMMA / packed[..., 3:]
