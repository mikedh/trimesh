"""
lights.py
--------------

Read and write the `KHR_lights_punctual` glTF extension. Every light is
named after the node carrying it, so `Scene.lights` and the scene graph
refer to the same thing.
"""

import numpy as np

from ...constants import log

# the extension key, and trimesh light class name to GLTF light
# type with the reverse for loading
NAME = "KHR_lights_punctual"
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
    tree in place.

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
    for light_index, light in enumerate(lights):
        entry = {
            "name": light.name,
            "type": _TYPES[type(light).__name__],
            # the extension stores linear RGB in the 0.0 - 1.0 range
            "color": color.to_float(light.color)[:3].tolist(),
            "intensity": float(light.intensity),
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
        stored.append(entry)
        # the node holding the light points at it, merged into a copy of
        # any extensions the node already has so the scene isn't changed
        index = node_index.get(light.name)
        if index is None:
            log.warning(f"light `{light.name}` has no node!")
            continue
        node = tree["nodes"][index]
        node["extensions"] = {**node.get("extensions", {}), NAME: {"light": light_index}}

    tree["extensions"] = {**tree.get("extensions", {}), NAME: {"lights": stored}}
