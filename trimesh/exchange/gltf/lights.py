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
from .animation import WXYZ_FROM_XYZW, XYZW_FROM_WXYZ

# the extension key, and trimesh light class name to GLTF light
# type with the reverse for loading
NAME = "KHR_lights_punctual"
# the extension key of the image based light a scene can have
ENVIRONMENT = "EXT_lights_image_based"
# RGBD stores radiance to this power, Babylon.js's `LinearEncodePowerApprox`
RGBD_GAMMA = 2.2
_TYPES = {
    "DirectionalLight": "directional",
    "PointLight": "point",
    "SpotLight": "spot",
}
_CLASSES = {value: key for key, value in _TYPES.items()}


def parse_lights(header, names):
    """
    Collect the `KHR_lights_punctual` lights nodes carry, one per node
    as two nodes may reference one entry of the document's array.

    Parameters
    ------------
    header : dict
      Parsed GLTF header.
    names : dict
      Mapping of {node index in the file : node name in the scene}

    Returns
    ----------
    lights : list
      One `trimesh.scene.lighting.Light` per node reference, in file order.
    """
    from ...scene import lighting
    from ...visual import color

    stored = header.get("extensions", {}).get(NAME, {}).get("lights", [])

    lights = []
    for index, node in enumerate(header.get("nodes", [])):
        reference = node.get("extensions", {}).get(NAME, {}).get("light")
        if reference is None:
            continue
        if not isinstance(reference, int) or reference >= len(stored):
            log.warning("node references a missing light!")
            continue

        entry = stored[reference]
        constructor = getattr(lighting, _CLASSES.get(entry.get("type"), ""), None)
        if constructor is None:
            log.warning(f"unsupported light type `{entry.get('type')}`!")
            continue

        kwargs = {
            "name": names[index],
            "intensity": entry.get("intensity", 1.0),
            "radius": entry.get("range"),
            # glTF lights are white unless they say otherwise
            "color": color.to_rgba(
                np.array(entry.get("color", [1.0, 1.0, 1.0]), dtype=np.float64)
            ),
        }
        if constructor is lighting.SpotLight:
            spot = entry.get("spot", {})
            # outer first as the inner setter validates against it
            kwargs["outerConeAngle"] = spot.get("outerConeAngle", np.pi / 4.0)
            kwargs["innerConeAngle"] = spot.get("innerConeAngle", 0.0)

        lights.append(constructor(**kwargs))

    return lights


def append_lights(lights, tree, node_index):
    """
    Write the light array and every node's reference into it into a
    tree in place, skipping any `EnvironmentLight` as glTF has no such light.

    Parameters
    ------------
    lights : list
      Sequence of `trimesh.scene.lighting.Light`.
    tree : dict
      GLTF header, mutated in place, with `nodes` already populated.
    node_index : dict
      Mapping of {node name : index in `tree["nodes"]`}
    """
    from ...visual import color

    stored = []
    for light in lights:
        kind = _TYPES.get(type(light).__name__)
        if kind is None:
            continue
        index = node_index.get(light.name)
        if index is None:
            log.warning(f"light `{light.name}` has no node!")
            continue
        node = tree["nodes"][index]
        entry = {
            "name": light.name,
            # the extension stores linear RGB in the 0.0 - 1.0 range
            "color": color.to_float(light.color)[:3].tolist(),
            "intensity": float(light.intensity),
            "type": kind,
        }
        if light.radius is not None:
            entry["range"] = float(light.radius)
        if entry["type"] == "spot":
            # only a spot has a cone, and the extension defaults differ
            # from ours so always write both rather than guessing
            entry["spot"] = {
                "innerConeAngle": float(light.innerConeAngle),
                "outerConeAngle": float(light.outerConeAngle),
            }
        # the node holding the light points at it, merged into a copy of
        # any extensions the node already has so the scene isn't changed
        node["extensions"] = {**node.get("extensions", {}), NAME: {"light": len(stored)}}
        stored.append(entry)

    if len(stored) > 0:
        tree["extensions"] = {**tree.get("extensions", {}), NAME: {"lights": stored}}


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


def append_environment(scene, tree, append_image):
    """
    Write the `EnvironmentLight` of a scene as the `EXT_lights_image_based`
    light of a tree's scene in place, its node's world rotation as `rotation`.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene whose lights are its own.
    tree : dict
      GLTF header, mutated in place.
    append_image : function
      Stores a `PIL.Image.Image` and returns its index in `tree["images"]`.
    """
    from PIL import Image

    from ...scene.lighting import EnvironmentLight, prefilter, spherical_harmonics
    from ...transformations import quaternion_from_matrix

    lights = [light for light in scene.lights if isinstance(light, EnvironmentLight)]
    if len(lights) > 1 and tol.strict:
        raise ValueError("a glTF scene has at most one environment light!")
    elif len(lights) > 1:
        log.warning("exporting the first of several environment lights")
    if len(lights) == 0 or lights[0].name not in scene.graph.nodes:
        return
    light, levels = lights[0], prefilter(lights[0].cubemap)
    entry = {
        "name": light.name,
        "intensity": float(light.intensity),
        "rotation": quaternion_from_matrix(scene.graph[light.name][0])[
            XYZW_FROM_WXYZ
        ].tolist(),
        "irradianceCoefficients": spherical_harmonics(light.cubemap).tolist(),
        "specularImageSize": len(levels[0][0]),
        # each face flipped about its vertical axis as the spec says
        "specularImages": [
            [append_image(Image.fromarray(rgbd_pack(face)[:, ::-1])) for face in level]
            for level in levels
        ],
    }
    tree["extensions"] = {**tree.get("extensions", {}), ENVIRONMENT: {"lights": [entry]}}
    tree["scenes"][0]["extensions"] = {ENVIRONMENT: {"light": 0}}


def parse_environment(header, scene_index, images):
    """
    The `EXT_lights_image_based` light of a scene of a GLTF header, which
    is its mirror level as the rest is recomputed from it.

    Parameters
    ------------
    header : dict
      Parsed GLTF header.
    scene_index : int
      Which of the header's scenes is loaded.
    images : list or None
      `PIL.Image.Image` by index into `header["images"]`.

    Returns
    ----------
    light : trimesh.scene.lighting.EnvironmentLight or None
      The scene's environment, None if it has none or it is broken.
    rotation : (4, 4) float
      The extension's `rotation` as a transform.
    """
    from ...scene.lighting import EnvironmentLight
    from ...transformations import quaternion_matrix

    light, entry = None, {}
    try:
        scene = header["scenes"][scene_index].get("extensions", {}).get(ENVIRONMENT)
        if scene is not None:
            entry = header["extensions"][ENVIRONMENT]["lights"][scene["light"]]
            faces = [
                images[index].convert("RGBA") for index in entry["specularImages"][0]
            ]
            light = EnvironmentLight(
                name=entry.get("name"),
                intensity=entry.get("intensity"),
                cubemap=rgbd_unpack(
                    np.array([np.asarray(face)[:, ::-1] for face in faces])
                ),
            )
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        if tol.strict:
            raise
        log.warning("unable to load environment light", exc_info=True)
    rotation = np.asarray(entry.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    return light, quaternion_matrix(rotation[WXYZ_FROM_XYZW])
